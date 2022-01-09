#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement Deep Q-leaning, DDQN and Prioritized experience replay


"""

import numpy as np
import random
from collections import namedtuple, deque

from .model import SimpleQNetwork, CEQNetwork_1, CEQNetwork_2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''
Begin help functions
'''

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 2        # how often to update the network
PER_ALPHA = 0.6         # importance sampling exponent
PER_BETA = 0.4          # prioritization exponent
REPLAY_STEP = 4         # 指定replay step數目 (default=4 by OuYang)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
End help functions
'''


def weighted_mse_loss(input, target, weights):
    '''
    Return the weighted mse loss to be used by Prioritized experience replay

    :param input: torch.Tensor.
    :param target: torch.Tensor.
    :param weights: torch.Tensor.
    :return loss:  torch.Tensor.
    '''
    # source: http://
    # forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
    out = (input-target)**2
    out = out * weights.expand_as(out)
    loss = out.mean(0)  # or sum over whatever dimensions
    return loss


class DQNAgent():
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, seed, use_dueling=False, use_double=False, select_network=0):
        '''Initialize an DQNAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param seed: int. random seed
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.loss_record = torch.zeros(1)

        # Q-Network
        # self.qnet = QNetwork(state_size, action_size, seed).to(device)
        # self.qnet_target = QNetwork(state_size, action_size, seed).to(device)
        # # TODO: test RMSprop here
        # self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)

        # select Q-Network
        if select_network == 0:
            Net = SimpleQNetwork
        elif select_network == 1:
            Net = CEQNetwork_1
        elif select_network == 2:
            Net = CEQNetwork_2
        else:
            raise ValueError("Network type not supported.")
        self.qnetwork_local = Net(state_size, action_size, fc1_units=64, fc2_units=64, step=REPLAY_STEP, num_treatment=2).to(device)
        self.qnetwork_target = Net(state_size, action_size, fc1_units=64, fc2_units=64, step=REPLAY_STEP, num_treatment=2).to(device)
        
        print("Using {}. Number of Params: {}".format(self.qnetwork_local.name, sum(p.numel() for p in self.qnetwork_local.parameters() if p.requires_grad)))
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
	
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, replay_step=REPLAY_STEP)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, treatment_i):
#         print("t_i: ", self.treatment_his)
        self.memory.add(state, action, reward, next_state, done, treatment_i)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            # and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def step_causal(self, state, action, reward, next_state, done, treatment_i):
        self.memory.add(state, action, reward, next_state, done, treatment_i)

    def act(self, state, eps=0.):
        '''Returns actions for given state as per current policy.

        :param state: array_like. current state
        :param eps: float. epsilon, for epsilon-greedy action selection
        '''
        # 
        replay = self.memory.recall()
        if replay == []: # memory empty, pick random action as prev_action
            prev_action = torch.LongTensor([[random.choice(np.arange(self.action_size)).astype(np.int32)]]).long().to(device)
        else:
            prev_action = torch.LongTensor([[replay[-1].action.astype(np.int32)]]).long().to(device) 
        prev_states = [r.state for r in replay]
        states = prev_states + [state]
        states = [torch.from_numpy(s).float().unsqueeze(0).to(device) for s in states]
        item = {'state': states, 'prev_action': prev_action}

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(item)['y']
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().data.numpy()
        else:
            return random.choice(np.arange(self.action_size)), action_values.cpu().data.numpy()

    def act_causal(self, state, eps=0.):
        '''Returns actions for given state as per current policy.

        :param state: array_like. current state
        :param eps: float. epsilon, for epsilon-greedy action selection
        '''
        # 
        replay = self.memory.recall()
        if replay == []: # memory empty, pick random action as prev_action
            prev_action = torch.LongTensor([[random.choice(np.arange(self.action_size)).astype(np.int32)]]).long().to(device)
        else:
            prev_action = torch.LongTensor([[replay[-1].action.astype(np.int32)]]).long().to(device) 
        prev_states = [r.state for r in replay]
        states = prev_states + [state]
        states = [torch.from_numpy(s).float().unsqueeze(0).to(device) for s in states]
        item = {'state': states, 'prev_action': prev_action}

        self.qnetwork_local.eval()
        with torch.no_grad():
            out = self.qnetwork_local(item)
        
        action_values = out['y']
        q_value = np.max(out['y'].cpu().data.numpy())

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().data.numpy(), {'z': out['z'].cpu().data.numpy(), 'q': q_value}
        else:
            return random.choice(np.arange(self.action_size)), action_values.cpu().data.numpy(), {'z': out['z'].cpu().data.numpy(), 'q': q_value}

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.

        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states, actions, rewards, next_states, dones, treatments = experiences
        #rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards
        
#         print("treatments", treatments.to(device).shape)

        # it is optimized to use in gpu
        # Get max predicted Q values (for next states) from target model
        # max(1) := np.max(array, axis=1)
        # unsqueeze : = array.reshape(-1, 1)
        # max_a' \hat{Q}(\phi(s_{t+1}, a'; θ^{-}))
        # max_Qhat = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # y_i = r + γ * maxQhat
        # # y_i = r, if done
        # Q_target = rewards_ + (gamma * max_Qhat * (1 - dones))

        # # Q(\phi(s_t), a_j; \theta)
        # Q_expected = self.qnet(states).gather(1, actions)

        # # perform gradient descent step on on (y_i - Q)**2
        # loss = F.mse_loss(Q_expected, Q_target)
        # self.optimizer.zero_grad()  # Clear the gradients
        # loss.backward()
        # self.optimizer.step()

        # # ------------------- update target network ------------------- #
        # self.soft_update(self.qnet, self.qnet_target, TAU)
                # Get max predicted Q values (for next states) from target model
        # next_states are not yet attacked, so treatments are always zero
        next_batch = {'state': next_states, 'prev_action': actions[-1], 't': treatments[1:] + [torch.zeros_like(treatments[-1])]}
        curr_batch = {'state': states,      'prev_action': actions[-2], 't': treatments}
        if self.use_double:
            action_values = self.qnetwork_local(next_batch)['y']
            indices = torch.argmax(action_values.detach(),1)
            action_values = self.qnetwork_target(next_batch)['y']
            Q_targets_next = action_values.detach().gather(1,indices.unsqueeze(1))
        else:
            # target & local --> local only (?)
            # 單一模型學比較快
            action_values = self.qnetwork_local(next_batch)['y']
            Q_targets_next = action_values.detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        model_out = self.qnetwork_local(curr_batch)
        action_values = model_out['y']
        Q_expected = action_values.gather(1, actions[-1].long())

        # Compute loss
        t_onehot = torch.zeros(model_out['t'].shape).type(model_out['t'].type())
        t_onehot = t_onehot.scatter(1, treatments[-1].long(), 1)
        t_loss = nn.BCEWithLogitsLoss()(model_out['t'], t_onehot)
        loss = F.mse_loss(Q_expected, Q_targets) + t_loss
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_record = loss # update the loss
#         self.treatment_his = []
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: PyTorch model. weights will be copied from
        :param target_model: PyTorch model. weights will be copied to
        :param tau: float. interpolation parameter
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def get_treatment(self, states):
        """ get treatment from states
            TODO: treatment based on having adversarial attack
        """
        batch_size = states.shape[0]
#         print("Treatment", self.treatment_his)
        t = torch.ones(batch_size).float()
#         print('T_o: ', len(t_o), end = "\n")
#         t = torch.tensor(self.treatment_his).float()
#         print('T: ', len(t), end = "\n")
        if states.is_cuda:
            t = t.cuda()
        return t

    def check_loss(self):
        return self.loss_record.cpu().detach().numpy()

def compute_kl(mu):
    mu_2 = torch.pow(z, 2)
    loss = torch.mean(mu_2)
    return loss

class DDQNAgent(DQNAgent):
    '''
    Implementation of a DDQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, seed):
        '''Initialize an DoubleDQNAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param seed: int. random seed
        '''
        super(DDQNAgent, self).__init__(state_size, action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.

        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states, actions, rewards, next_states, dones, treatments = experiences
        rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards
#         treatments = self.get_treatment(states)

#         # arg max_{a} \hat{Q}(s_{t+1}, a, θ_t)
#         print("Q-Net(next_state): ",self.qnetwork_local(next_states))
#         argmax_actions = self.qnetwork_local(next_states)[0].detach().max(1)[1].unsqueeze(1)
#         # max_Qhat :=  \hat{Q}(s_{t+1}, argmax_actions, θ^−)
#         max_Qhat = self.qnetwork_target(next_states).gather(1, argmax_actions)
        if self.use_double:
            action_values = self.qnetwork_local(next_states)['py']
            indices = torch.argmax(action_values.detach(),1)
            action_values = self.qnetwork_target(next_states)['py']
            Q_targets_next = action_values.detach().gather(1,indices.unsqueeze(1))
        else:
            action_values = self.qnetwork_target(next_states)['py']
            Q_targets_next = action_values.detach().max(1)[0].unsqueeze(1)
        # y_i = r + γ * maxQhat
        # y_i = r, if done
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        model_out = self.qnetwork_local(states)
        action_values = model_out['py']
        Q_expected = action_values.gather(1, actions)
#         # Q(\phi(s_t), a_j; \theta)
#         Q_expected = self.qnetwork_local(states).gather(1, actions)

        # perform gradient descent step on on (y_i - Q)**2
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # CEVAE Loss
        if self.qnetwork_local.name == 'CEVAE_QNetwork':
            qy = model_out['qy'].gather(1, actions)
            z, qt, px, pt = model_out['z'], model_out['qt'], model_out['px'], model_out['pt']
            
            loss_px = F.mse_loss(px, states)
            loss_pt = - torch.mean(torch.sum(pt.log_prob(treatments), dim=1))
            loss_z  = torch.pow(z, 2).mean()
            loss_qt = - torch.mean(torch.sum(qt.log_prob(treatments), dim=1))
            loss_qy = F.mse_loss(qy, Q_targets)
            loss = loss + loss_px + loss_pt + loss_z + loss_qt + loss_qy
        
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class DDQNPREAgent(DQNAgent):
    '''
    Implementation of a DDQN agent that used priritized experience replay and
    interacts with and learns from the environment
    '''

    def __init__(self, state_size, action_size, seed, alpha=PER_ALPHA,
                 max_t=1000, initial_beta=PER_BETA):
        '''Initialize an DDQNPREAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param seed: int. random seed
        '''
        super(DDQNPREAgent, self).__init__(state_size, action_size, seed)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size,
                                              BUFFER_SIZE,
                                              BATCH_SIZE,
                                              seed,
                                              alpha)
        # get_is_weights(self, current_beta):
        # update_priorities(self, td_errors):
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.alpha = alpha
        self.initial_beta = initial_beta
        self.max_t = max_t
        self.t_step = 0

    def get_beta(self, t):
        '''
        Return the current exponent β based on its schedul. Linearly anneal β
        from its initial value β0 to 1, at the end of learning.

        :param t: integer. Current time step in the episode
        :return current_beta: float. Current exponent beta
        '''
        f_frac = min(float(t) / self.max_t, 1.0)
        current_beta = self.initial_beta + f_frac * (1. - self.initial_beta)
        return current_beta

    def learn(self, experiences, gamma, t=1000):
        '''Update value parameters using given batch of experience tuples.

        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states, actions, rewards, next_states, dones, treatments = experiences
        rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards
#         treatments = self.get_treatment(states)

#         # arg max_{a} \hat{Q}(s_{t+1}, a, θ_t)
#         print("Q-Net(next_state): ",self.qnetwork_local(next_states))
#         argmax_actions = self.qnetwork_local(next_states)[0].detach().max(1)[1].unsqueeze(1)
#         # max_Qhat :=  \hat{Q}(s_{t+1}, argmax_actions, θ^−)
#         max_Qhat = self.qnetwork_target(next_states).gather(1, argmax_actions)
        if self.use_double:
            action_values = self.qnetwork_local(next_states)['py']
            indices = torch.argmax(action_values.detach(),1)
            action_values = self.qnetwork_target(next_states)['py']
            Q_targets_next = action_values.detach().gather(1,indices.unsqueeze(1))
        else:
            action_values = self.qnetwork_target(next_states)['py']
            Q_targets_next = action_values.detach().max(1)[0].unsqueeze(1)
        # y_i = r + γ * maxQhat
        # y_i = r, if done
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        action_values = self.qnetwork_local(states)['py']
        Q_expected = action_values.gather(1, actions)

        # compute importance-sampling weight wj
        f_currbeta = self.get_beta(t) # param t: integer. Current time step in the episode
        weights = self.memory.get_is_weights(current_beta=f_currbeta)

        # compute TD-error δj and update transition priority pj
        td_errors = Q_targets - Q_expected
        self.memory.update_priorities(td_errors)

        # perform gradient descent step
        # Accumulate weight-change ∆←∆+wj x δj x ∇θQ(Sj−1,Aj−1)
        # loss = F.mse_loss(Q_expected*weights, Q_target*weights)
        loss = weighted_mse_loss(Q_expected, Q_targets, weights)
        # loss = F.mse_loss(Q_expected, Q_target)*weights
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class ReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed, replay_step=4):
        '''Initialize a ReplayBuffer object.

        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done", "treatment_i"])
        self.seed = random.seed(seed)
        self.replay_step = replay_step

    def add(self, state, action, reward, next_state, done, treatment_i):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done, treatment_i)
        self.memory.append(e)

    def recall(self):
        ''' get recent (step - 1) replaybuffer'''
        step = int(min(self.replay_step - 1, len(self.memory)))
        if step == 0:
            return []
        return [self.memory[-i] for i in range(step, 0, -1)]

    def sample(self):
        '''Randomly sample a batch of experiences from memory.'''
        experience_idxs = random.sample(range(len(self.memory)), k=self.batch_size)

        states  = [torch.from_numpy(np.vstack([self.memory[i - j].state  for i in experience_idxs if self.memory[i - j] is not None])).float().to(device) for j in range(self.replay_step - 1, -1, -1)]
        actions = [torch.from_numpy(np.vstack([self.memory[i - j].action for i in experience_idxs if self.memory[i - j] is not None])).float().to(device) for j in range(self.replay_step - 1, -1, -1)]
        next_states = [torch.from_numpy(np.vstack([self.memory[i - j].next_state  for i in experience_idxs if self.memory[i - j] is not None])).float().to(device) for j in range(self.replay_step - 1, -1, -1)]
        
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in experience_idxs
                                              if self.memory[i] is not None])).float().to(device)
        
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in experience_idxs
                                            if self.memory[i] is not None]).astype(np.uint8)).float().to(device)
        
        treatments =  [torch.from_numpy(np.vstack([self.memory[i - j].treatment_i for i in experience_idxs if self.memory[i - j] is not None])).float().to(device) for j in range(self.replay_step - 1, -1, -1)]
                    
        return (states, actions, rewards, next_states, dones, treatments)
    
    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)


class PrioritizedReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        '''Initialize a ReplayBuffer object.

        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size: int: size of each training batch
        :param seed: int: random seed
        :param alpha: float: 0~1 indicating how much prioritization is used
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done", "treatment_i"])
        self.seed = random.seed(seed)
        # specifics
        self.alpha = max(0., alpha)  # alpha should be >= 0
        self.priorities = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self.cum_priorities = 0.
        self.eps = 1e-6
        self._indexes = []
        self.max_priority = 1.**self.alpha

    def add(self, state, action, reward, next_state, done, treatment_i):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done, treatment_i)
        self.memory.append(e)
        # exclude the value that will be discareded
        if len(self.priorities) >= self._buffer_size:
            self.cum_priorities -= self.priorities[0]
        # include the max priority possible initialy
        self.priorities.append(self.max_priority)  # already use alpha
        # accumulate the priorities abs(td_error)
        self.cum_priorities += self.priorities[-1]

    def sample(self):
        '''
        Sample a batch of experiences from memory according to importance-
        sampling weights

        :return. tuple[torch.Tensor]. Sample of past experiences
        '''
        i_len = len(self.memory)
        na_probs = None
        if self.cum_priorities:
            na_probs = np.array(self.priorities)/self.cum_priorities
        l_index = np.random.choice(i_len,
                                   size=min(i_len, self.batch_size),
                                   p=na_probs)
        self._indexes = l_index

        experiences = [self.memory[ii] for ii in l_index]

        states = torch.from_numpy(np.vstack([e.state for e in experiences
                                  if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences
                                   if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences
                                   if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state
                                                  for e in experiences
                                                  if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences
                                            if e is not None]).astype(np.uint8)).float().to(device)

        treatments = torch.from_numpy(np.vstack([e.treatment_i for e in experiences
                                  if e is not None])).float().to(device)
        return (states, actions, rewards, next_states, dones, treatments)

    def _calculate_is_w(self, f_priority, current_beta, max_weight, i_n):
        #  wi= ((N x P(i)) ^ -β)/max(wi)
        f_wi = (i_n * f_priority/self.cum_priorities)
        return (f_wi ** -current_beta)/max_weight

    def get_is_weights(self, current_beta):
        '''
        Return the importance sampling (IS) weights of the current sample based
        on the beta passed

        :param current_beta: float. fully compensates for the non-uniform
            probabilities P(i) if β = 1
        '''
        # calculate P(i) to what metters
        i_n = len(self.memory)
        max_weight = (i_n * min(self.priorities) / self.cum_priorities)
        max_weight = max_weight ** -current_beta

        this_weights = [self._calculate_is_w(self.priorities[ii],
                                             current_beta,
                                             max_weight,
                                             i_n)
                        for ii in self._indexes]
        return torch.tensor(this_weights,
                            device=device,
                            dtype=torch.float).reshape(-1, 1)

    def update_priorities(self, td_errors):
        '''
        Update priorities of sampled transitions
        inspiration: https://bit.ly/2PdNwU9

        :param td_errors: tuple of torch.tensors. TD-Errors of last samples
        '''
        for i, f_tderr in zip(self._indexes, td_errors):
            f_tderr = float(f_tderr)
            self.cum_priorities -= self.priorities[i]
            # transition priority: pi^α = (|δi| + ε)^α
            self.priorities[i] = ((abs(f_tderr) + self.eps) ** self.alpha)
            self.cum_priorities += self.priorities[i]
        self.max_priority = max(self.priorities)
        self._indexes = []

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)
