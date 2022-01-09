import torch
import torch.nn.functional as F
import numpy as np
import random

class atk_model():
    """Interacts with and learns from the environment."""

    def __init__(self, C_atk = False, Beta = 0.25, epsilon = 0.3, Ftype = 1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            use_dueling (bool): if 'True' use dueling agent
            use_double (bool): if 'True' use double DDQN agent
        """
        self.Beta = Beta
        # self.act_vals = act_vals
        self.C_atk = C_atk
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.epsilon = epsilon
        self.Ftype = Ftype
         
        self.gamma = 0.99 # discount factor for DQN agent
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def Atck_F(self, state, prev_state, atk = False, t_i = 0 ):
        if atk == True:
            s_size = len(state)
#             # zero is too strong... disable now
            if self.Ftype == 1:
                state = prev_state # replay the state "(lag)", "frozen"
            elif self.Ftype == 2: ### adding random noise with 0.3 SNR
                state = state + np.random.normal(0., np.var(state[0:s_size]), s_size)
                state = np.array(state)
            else:
                state[0:s_size-1] = 0
                state = np.array(state) 
            t_i = 1
        return state, t_i

    def Atck_C(self, act_vals):
        C_atk = self.C_atk
        T_cst = -1. # Beta = 0.35 Refer to Lin et al. 2017
        max_a = np.argmax(act_vals)
        min_a = np.argmin(act_vals)
        Q = np.array(act_vals)
        C = np.exp(Q[0][max_a]/T_cst)/sum(np.exp(Q[0]/T_cst)) - np.exp(Q[0][min_a]/T_cst)/sum(np.exp(Q[0])/T_cst)
        if C > self.Beta:
            C_atk = True # Enable Timing Attack
        return C_atk
    
    def Atck_Adv(self, state, action, reward, next_state, done, attack_network, target_network, atk=False):
        # some environmental setup
        if atk:
            target_network.eval()
            # numpy to torch
            state = torch.from_numpy(state[np.newaxis, :]).float().to(self.device)
            next_state = torch.from_numpy(next_state[np.newaxis, :]).float().to(self.device)
            
            # Compute target Q for current state from target model
            action_value = target_network(next_state)['py']
            Q_target_next = action_value.detach().max()
            Q_target = reward + (self.gamma * Q_target_next * (1 - done))
    
            # Compute expected Q from attack model
            state.requires_grad = True
            action_value = attack_network(state)['py']
            Q_expected = action_value[0, action]
            
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_target)
            attack_network.zero_grad()
    
            # Calculate gradients of model in backward pass
            loss.backward()
    
            data_grad = state.grad.data
            # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_state = state + self.epsilon * sign_data_grad
            # Adding clipping to maintain [0,1] range
            perturbed_state = torch.clamp(perturbed_state, 0, 1)
            # zero grads
            attack_network.zero_grad()
            target_network.zero_grad()
            target_network.train()
        
            return perturbed_state.detach().cpu().numpy(), 1
        else:
            return state, 0
