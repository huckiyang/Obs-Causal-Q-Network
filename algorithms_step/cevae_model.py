#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the Q-network used by the agents
modified from ucaiado

Created on 10/23/2018
causal variational model for deep q network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


'''
Begin help functions
'''


'''
End help functions
'''


# class QNetwork(nn.Module):
#     '''Actor (Policy) Model.'''

#     def __init__(self, state_size, action_size, seed):
#         '''
#         Initialize parameters and build model.

#         :param state_size: int. Dimension of each state
#         :param action_size: int. Dimension of each action
#         :param seed: int. Random seed
#         '''
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_size)

#     def forward(self, state):
#         '''Build a network that maps state -> action values.'''
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, use_dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            use_dueling (bool): if 'True' use dueling agent
        """
        super(QNetwork, self).__init__()
        self.name = 'QNetwork'
        self.seed = torch.manual_seed(seed)
        self.use_dueling = use_dueling

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.state_value = nn.Linear(fc2_units, 1)

    def forward(self, state, t=None):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.use_dueling:
            # advantage values + state value
            return {'py': self.fc3(x) + self.state_value(x)}
        else:
            return {'py': self.fc3(x)}

########## CEVAE ##########
class encoder_ce(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, z_units=64):
        super(encoder_ce, self).__init__()

        # q(t|x)
        self.logits_t = nn.Sequential(nn.Linear(state_size, fc1_units),
                                      nn.ReLU(),
                                      nn.Linear(fc1_units, fc2_units),
                                      nn.ReLU(),
                                      nn.Linear(fc2_units, 1),
                                      nn.Sigmoid())
        # q(y|x, t)
        self.hqy = nn.Sequential(nn.Linear(state_size, fc1_units),
                                 nn.ReLU(),
                                 nn.Linear(fc1_units, fc2_units),
                                 nn.ReLU())
        self.qy_t0 = nn.Linear(fc2_units, action_size)
        self.qy_t1 = nn.Linear(fc2_units, action_size)

        # q(z|x, t, y)
        self.hqz = nn.Sequential(nn.Linear(state_size + action_size, fc1_units),
                                 nn.ReLU())
        self.muq_t0 = nn.Linear(fc1_units, z_units, bias=False)
        self.muq_t1 = nn.Linear(fc1_units, z_units, bias=False)
        self.sigmaq_t0 = nn.Linear(fc1_units, z_units)
        self.sigmaq_t1 = nn.Linear(fc1_units, z_units)
        
        
    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def forward(self, state, _t):
        # q(t|x)
#         print("T:",_t, end = "\n")
        logits_t = self.logits_t(state)
        qt = dist.bernoulli.Bernoulli(logits_t)

        # q(y||x, t)
        hqy = self.hqy(state)
        qy_t0 = self.qy_t0(hqy)
        qy_t1 = self.qy_t1(hqy)

        if self.training:
            qy = qy_t1 * _t + (1 - _t) * qy_t0
        else:
            qt_sample = qt.sample()
            qy = qt_sample * qy_t1 + (1. - qt_sample) * qy_t0

        # q(z|x, t, y)
        hqz = self.hqz(torch.cat([state, qy], dim=-1))
        #hqz = self.hqz(state)
        muq_t0, sigmaq_t0 = self.muq_t0(hqz), F.softplus(self.sigmaq_t0(hqz))
        muq_t1, sigmaq_t1 = self.muq_t1(hqz), F.softplus(self.sigmaq_t1(hqz))

        if self.training:
            z = self.reparameterize_normal(_t * muq_t1 + (1 - _t) * muq_t0,
                                           _t * sigmaq_t1 + (1 - _t) * sigmaq_t0)
        else:
            z = self.reparameterize_normal(qt_sample * muq_t1 + (1. - qt_sample) * muq_t0,
                                           qt_sample * sigmaq_t1 + (1. - qt_sample) * sigmaq_t0)
        return z, qt, qy


class decoder_ce(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, z_units=64):
        super(decoder_ce, self).__init__()
        self.reoconstruct = nn.Sequential(nn.Linear(z_units, state_size))

        self.logits_t = nn.Sequential(nn.Linear(z_units, 1),
                                      nn.Sigmoid())
        self.py_t0 = nn.Sequential(nn.Linear(z_units, z_units),
                                   nn.ReLU(),
                                   nn.Linear(z_units, action_size))
        self.py_t1 = nn.Sequential(nn.Linear(z_units, z_units),
                                   nn.ReLU(),
                                   nn.Linear(z_units, action_size))

    def forward(self, z, _t):
        # p(x|z)
        state = self.reoconstruct(z)

        # p(t|z)
        logits_t = self.logits_t(z)
        t = dist.bernoulli.Bernoulli(logits_t)

        # p(y|t, z)
        py_t0 = self.py_t0(z)
        py_t1 = self.py_t1(z)

        if self.training:
            y = _t * py_t1 + (1 - _t) * py_t0
        else:
            t_sample = t.sample()
            y = t_sample * py_t1 + (1. - t_sample) * py_t0

        return y, state, t

########## NEW CEVAE ##########
class encoder_ce_fusion(nn.Module):
    def __init__(self, state_size, action_size, num_treatment=2, fc1_units=64, fc2_units=64, z_units=64):
        super(encoder_ce_fusion, self).__init__()

        # q(t|x)
        # Now q(t|x) is a classification task
        self.num_treatment = num_treatment
        self.qt = nn.Sequential(nn.Linear(state_size, fc1_units),
                                nn.ReLU(),
                                nn.Linear(fc1_units, fc2_units),
                                nn.ReLU(),
                                nn.Linear(fc2_units, num_treatment))
        self.embed_t = nn.Sequential(nn.Linear(num_treatment, num_treatment * 16))

        # q(y|x, t)
        self.hqy = nn.Sequential(nn.Linear(state_size + num_treatment * 16, fc1_units),
                                 nn.ReLU(),
                                 nn.Linear(fc1_units, action_size),
                                 nn.ReLU())
        
        # q(z|x, t, y)
        self.hqz = nn.Sequential(nn.Linear(state_size + action_size + num_treatment * 16, fc1_units),
                                 nn.ReLU())
        self.muq = nn.Linear(fc1_units, z_units, bias=False)
        self.sigmaq = nn.Linear(fc1_units, z_units)
        
        
    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def forward(self, state, _t):
        # q(t|x)
        
        qt = self.qt(state)
        if self.training:
            onehot_t = torch.zeros(qt.shape[0], self.num_treatment).type(qt.type())
            onehot_t.scatter(1, _t.long(), 1)
            embed_t = self.embed_t(onehot_t)
        else:
            onehot_t = torch.zeros(qt.shape[0], self.num_treatment).type(qt.type())
            onehot_t.scatter(1, qt.topk(1, 1, True, True)[1], 1)
            embed_t = self.embed_t(onehot_t)

        # q(y||x, t)
        qy = self.hqy(torch.cat([state, embed_t], dim=-1))
        
        # q(z|x, t, y)
        hqz = self.hqz(torch.cat([state, qy, embed_t], dim=-1))
        #hqz = self.hqz(state)
        muq, sigmaq = self.muq(hqz), F.softplus(self.sigmaq(hqz))
        
        z = self.reparameterize_normal(muq, sigmaq)
        return z, qt, qy


class decoder_ce_fusion(nn.Module):
    def __init__(self, state_size, action_size, num_treatment=2, fc1_units=64, fc2_units=64, z_units=64):
        super(decoder_ce_fusion, self).__init__()
        self.reoconstruct = nn.Sequential(nn.Linear(z_units, state_size))

        self.num_treatment = num_treatment
        self.pt = nn.Sequential(nn.Linear(z_units, num_treatment))
        self.embed_t = nn.Sequential(nn.Linear(num_treatment, num_treatment * 16))
        
        self.py = nn.Sequential(nn.Linear(z_units + num_treatment * 16, z_units),
                                nn.ReLU(),
                                nn.Linear(z_units, action_size))
        
    def forward(self, z, _t):
        # p(x|z)
        state = self.reoconstruct(z)
        # p(t|z)
        pt = self.pt(z)
        if self.training:
            onehot_t = torch.zeros(pt.shape[0], self.num_treatment).type(pt.type())
            onehot_t.scatter(1, _t.long(), 1)
            embed_t = self.embed_t(onehot_t)

        else:
            onehot_t = torch.zeros(pt.shape[0], self.num_treatment).type(pt.type())
            onehot_t.scatter(1, pt.topk(1, 1, True, True)[1], 1)
            embed_t = self.embed_t(onehot_t)

        # p(y|t, z)
        py = self.py(torch.cat([z, embed_t], dim=-1))
        
        return py, state, pt


########## VAE ##########
class encoder(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, z_units=64):
        super(encoder, self).__init__()

        # q(z|x)
        self.hqz = nn.Sequential(nn.Linear(state_size, fc1_units),
                                 nn.ReLU())
        self.muq = nn.Linear(fc1_units, z_units, bias=False)
        self.sigmaq = nn.Linear(fc1_units, z_units)
        
    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def forward(self, state):
        # q(z|x)
        hqz = self.hqz(state)
        muq, sigmaq = self.muq(hqz), F.softplus(self.sigmaq(hqz))

        z = self.reparameterize_normal(muq, sigmaq)
        return z


class decoder(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, z_units=64):
        super(decoder, self).__init__()
        self.reoconstruct = nn.Sequential(nn.Linear(z_units, state_size))

        self.py = nn.Sequential(nn.Linear(z_units, z_units),
                                nn.ReLU(),
                                nn.Linear(z_units, action_size))

    def forward(self, z, _t=None):
        # p(x|z)
        state = self.reoconstruct(z)
 
        # p(y|z)
        y = self.py(z)
        return y, state




class CEVAE_QNetwork(nn.Module):
    """ QNetwork with Causality reasoning
    """ 
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, z_units=64, use_dueling=False):
        super(CEVAE_QNetwork, self).__init__()
        self.name = 'CEVAE_QNetwork'
        self.seed = torch.manual_seed(seed)
        self.use_dueling = use_dueling

        self.encoder = encoder_ce(state_size, action_size, fc1_units, fc2_units, z_units)
        self.decoder = decoder_ce(state_size, action_size, fc1_units, fc2_units, z_units)

    def forward(self, state, t=None):
        x = state
        z, qt, qy = self.encoder(x, t)
        py, px, pt = self.decoder(z, t)

        return {'z': z,
                'qt': qt,
                'qy': qy,
                'py': py,
                'px': px,
                'pt': pt}


class New_CEVAE_QNetwork(nn.Module):
    """ New QNetwork with Causality reasoning
    """ 
    def __init__(self, state_size, action_size, seed, num_treatment=2, fc1_units=64, fc2_units=64, z_units=64, use_dueling=False):
        super(New_CEVAE_QNetwork, self).__init__()
        self.name = 'New_CEVAE_QNetwork'
        self.seed = torch.manual_seed(seed)
        self.use_dueling = use_dueling

        self.encoder = encoder_ce_fusion(state_size, action_size, num_treatment, fc1_units, fc2_units, z_units)
        self.decoder = decoder_ce_fusion(state_size, action_size, num_treatment, fc1_units, fc2_units, z_units)

    def forward(self, state, t=None):
        x = state
        z, qt, qy = self.encoder(x, t)
        py, px, pt = self.decoder(z, t)

        return {'z': z,
                'qt': qt,
                'qy': qy,
                'py': py,
                'px': px,
                'pt': pt}


class VAE_QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, z_units=64, use_dueling=False):
        super(VAE_QNetwork, self).__init__()
        self.name = 'VAE_QNetwork'
        self.seed = torch.manual_seed(seed)
        self.use_dueling = use_dueling

        self.encoder = encoder(state_size, action_size, fc1_units, fc2_units, z_units)
        self.decoder = decoder(state_size, action_size, fc1_units, fc2_units, z_units)

    def forward(self, state, t=None):
        x = state
        z = self.encoder(x)
        py, px = self.decoder(z)

        return {'z': z,
                'py': py,
                'px': px}
