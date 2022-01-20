import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class AbstractDQN(nn.Module):
    def __init__(self, state_size=4, action_size=2, fc1_units=64, fc2_units=64):
        super(AbstractDQN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(state_size, fc1_units),
                                     nn.ReLU(),
                                     nn.Linear(fc1_units, fc2_units),
                                     nn.ReLU())

    def _forward(self, data):
        raise NotImplementedError

    def forward(self, data):
        data['z'] = [self.encoder(s) for s in data['state']]
        out = self._forward(data)
        return out


class SimpleQNetwork(AbstractDQN):
    """ using $step frame and treatment. using prev_action when predict interference 
        attention：if using prev_action during training, the samples of data will only learn t, not learning y 
    """
    def __init__(self, state_size=4, action_size=2, fc1_units=64, fc2_units=64, step=2, num_treatment=2):
        super(SimpleQNetwork, self).__init__(state_size, action_size, fc1_units, fc2_units)
        self.name = 'SimpleQNetwork'
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        # self.fc3_units = fc3_units
        self.step = step
        self.logits_t = nn.Sequential(nn.Linear(fc2_units, fc2_units // 2),
                                      nn.ReLU(),
                                      nn.Linear(fc2_units // 2, num_treatment))

        self.fc = nn.Sequential(nn.Linear(fc2_units * step, fc2_units * step // 2),
                                nn.ReLU(),
                                nn.Linear(fc2_units * step // 2, fc2_units * step // 2),
                                nn.ReLU(),
                                nn.Linear(fc2_units * step // 2, action_size))

    def _forward(self, data):
        out = {}
        z = data['z'][-self.step:]
        t = [self.logits_t(_z) for _z in z]

        z = torch.cat(z, dim=-1)
        z = F.pad(z, pad=(self.fc2_units * self.step - z.shape[-1], 0)) # pad zeros to the left to fit in fc layer
        y = self.fc(z)
        
        onehot = torch.zeros(y.shape).type(y.type())
        data['onehot_prev_action'] = onehot.scatter(1, data['prev_action'].long(), 1)
            
        if self.training:
            y = torch.where(data['t'][-1] == 1, data['onehot_prev_action'], y)
        else:
            _t = t[-1].topk(1, 1, True, True)[1]
            y = torch.where(_t == 1, data['onehot_prev_action'], y)

        out['t'] = t[-1]
        out['y'] = y
        return out


class CEQNetwork_1(AbstractDQN):
    """ using $step frame and treatment, concat together, using fc to predict Q
    """ 
    def __init__(self, state_size=4, action_size=2, fc1_units=64, fc2_units=64, step=2, num_treatment=2):
        super(CEQNetwork_1, self).__init__(state_size, action_size, fc1_units, fc2_units)
        self.name = 'CEQNetwork_1'
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.num_treatment = num_treatment
        self.step = step
        self.logits_t = nn.Sequential(nn.Linear(fc2_units, fc2_units // 2),
                                      nn.ReLU(),
                                      nn.Linear(fc2_units // 2, num_treatment))

        self.fc = nn.Sequential(nn.Linear((fc2_units + num_treatment) * step , (fc2_units + num_treatment) * step // 2),
                                nn.ReLU(),
                                nn.Linear((fc2_units + num_treatment) * step // 2, (fc2_units + num_treatment) * step // 2),
                                nn.ReLU(),
                                nn.Linear((fc2_units + num_treatment) * step // 2, action_size))

    def _forward(self, data):
        out = {}
        z = data['z'][-self.step:]
        t = [self.logits_t(_z) for _z in z]

        z = torch.cat(z, dim=-1)
        z = F.pad(z, pad=(self.fc2_units * self.step - z.shape[-1], 0)) # pad zeros to the left to fit in fc layer
        t_stack = torch.stack(t, dim=1)
        
        if self.training:
            _t = torch.stack(data['t'][-self.step:], dim=1)
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, _t.long(), 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)
        else:
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, t_stack.topk(1, 2, True, True)[1], 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)

        onehot_t = F.pad(onehot_t, pad=(self.num_treatment * self.step - onehot_t.shape[-1], 0))
        y = self.fc(torch.cat([z, onehot_t], dim=-1))
        
        out['t'] = t[-1]
        out['y'] = y
        return out


class CEQNetwork_2(AbstractDQN):
    """ this model using $step frame and treatment, concat together. using eihter fc_t0 / fc_t1 to estimate Q based on the prediction result of t4
    """
    def __init__(self, state_size=4, action_size=2, fc1_units=64, fc2_units=64, step=2, num_treatment=2):
        super(CEQNetwork_2, self).__init__(state_size, action_size, fc1_units, fc2_units)
        self.name = 'CEQNetwork_2'
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.num_treatment = num_treatment
        self.step = step
        self.logits_t = nn.Sequential(nn.Linear(fc2_units, fc2_units // 2),
                                      nn.ReLU(),
                                      nn.Linear(fc2_units // 2, num_treatment))
        
        self.fc_t0 = nn.Sequential(nn.Linear((fc2_units + num_treatment) * step , (fc2_units + num_treatment) * step // 2),
                                   nn.ReLU(),
                                   nn.Linear((fc2_units + num_treatment) * step // 2, (fc2_units + num_treatment) * step // 2),
                                   nn.ReLU(),
                                   nn.Linear((fc2_units + num_treatment) * step // 2, action_size))
        self.fc_t1 = nn.Sequential(nn.Linear((fc2_units + num_treatment) * step , (fc2_units + num_treatment) * step // 2),
                                   nn.ReLU(),
                                   nn.Linear((fc2_units + num_treatment) * step // 2, action_size))

    def _forward(self, data):
        out = {}
        z = data['z'][-self.step:]
        t = [self.logits_t(_z) for _z in z]

        z = torch.cat(z, dim=-1)
        z = F.pad(z, pad=(self.fc2_units * self.step - z.shape[-1], 0)) # pad zeros to the left to fit in fc layer
        t_stack = torch.stack(t, dim=1)
        
        if self.training:
            _t = torch.stack(data['t'][-self.step:], dim=1)
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, _t.long(), 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)
            
            t_bernoulli = data['t'][-1]
        else:
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, t_stack.topk(1, 2, True, True)[1], 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)
            t_bernoulli = t[-1].topk(1, 1)[1].float() ## no sampling
            # t_bernoulli = dist.bernoulli.Bernoulli(F.softmax(torch.sigmoid(t[-1]), dim=-1)[:, 1])
            # t_bernoulli = t_bernoulli.sample()

        onehot_t = F.pad(onehot_t, pad=(self.num_treatment * self.step - onehot_t.shape[-1], 0))
        
        y_t0 = self.fc_t0(torch.cat([z, onehot_t], dim=-1))
        y_t1 = self.fc_t1(torch.cat([z, onehot_t], dim=-1))
        y = t_bernoulli * y_t1 + (1 - t_bernoulli) * y_t0
        
        out['t'] = t[-1]
        out['y'] = y
        return out
