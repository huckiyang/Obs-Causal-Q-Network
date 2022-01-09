import os
import time as ti

data_prefix = './data/'+ti.strftime("%m%d-%H%M")
s_currentpath = os.getcwd()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type = float, default= 0.40, help="for Atk_C theshold")
parser.add_argument("--baseline", type = int, default= 0, help= " 0 - with an attack or 1 - yes, use baseline model " )
parser.add_argument("--Atktype", type = int, default= 1, help= "choose attack (treatment) type, 1 - attack_F, 2 = adversarial")
parser.add_argument("--causal", type = int, default= 1, help= "0 - Not use treatment info, 1 - use treatment info")
parser.add_argument("--network", type = int, default=1, help="0 - Safe DQN, 1 - CIQ type1, 2 - CIQ type2")
parser.add_argument("--history", type = int, default= 0, help= "0 - not show treatment history")
parser.add_argument("--attack_network_path", type = str, default="", help="path to attack model, must be specified if use adversarial attack.")
parser.add_argument("--evaluate", type = int, default= 1, help= "1 - yes, evaluate the trained agent, 0 - no" )
parser.add_argument("--Ftype", type = int, default= 0, help= "type of F: 1 - Lag & Frozen Screen, 2 - Random Noise, else 0 - Zerout" )
parser.add_argument("--ratio", type = float, default= 0.2, help= "ratio of random time frame select" )
parser.add_argument("--shellratio", type = int, default= 1, help= "noise ratio for shell" )
parser.add_argument("--num_eval", type = int, default= 5, help= "number of evaluatel" )
args = parser.parse_args()

env_type = "fig_result"
fig_prefix = env_type + '/reply_' + ti.strftime("%m%d-%H%M") + '_Trt'+ str(args.Ftype)

if args.baseline == 1:
    if args.causal == 0:
        fig_prefix = fig_prefix + "_base_Net_" +str(args.network)+ "_0."+str(args.shellratio)# + env_type
    else:
        fig_prefix = fig_prefix + "_base_Net_wC"  +str(args.network)+ "_0."+str(args.shellratio)
else:
    if args.causal == 0:
        fig_prefix = fig_prefix + "_Atk_"+ str(args.Atktype)+ '_Net_' +str(args.network)+"_0."+ str(args.shellratio)
    else:
        fig_prefix = fig_prefix + "_Atk_wC_"+ str(args.Atktype)+ '_Net_' +str(args.network)+ "_0."+str(args.shellratio)

import sys
sys.path.append("../")  # include the root directory as the main
import eda
import pandas as pd
import numpy as np
import gym
from collections import deque
import pickle
import random
import torch
from collections import deque
from algorithms_step.dqn_agent import DQNAgent, DDQNAgent, DDQNPREAgent
from algorithms_step.model import SimpleQNetwork
from algorithms_step.attacker import atk_model

env = gym.make('CartPole-v0')
env.seed(0)

print('observation space:', env.observation_space)
print('action space:', env.action_space)

# reset the environment
# env.reset()
# number of actions
action_size = 2
state_size = 4
print("Timing Atk Ratio: " + str(10*args.shellratio) + "%" )


eval_num = args.num_eval
n_episodes = 3000
eps_start = 1.
eps_end=0.01
eps_decay=0.998
s_model = 'dqn'

# initialize
agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
atker = atk_model(Beta = args.beta, Ftype = args.Ftype)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.attack_network_path != "":
    attack_network = SimpleQNetwork(state_size=state_size, action_size=action_size, seed=0)
    state = torch.load(args.attack_network_path)
    attack_network.load_state_dict(state)
    attack_network = attack_network.to(device)
    attack_network.eval()

print( " Interference Type: " + str(args.Atktype), " Use baseline: ", str(args.baseline), "use CGM: ", str(args.causal), end = "\n")

loss_book = []
scores = []
scores_std = []                    # List containing the std dev of the last 100 episodes
scores_avg = []                    # List containing the mean of the last 100 episodes
scores_atkratio = []
scores_window = deque(maxlen=60)  # last n scores
eps = eps_start                    # initialize epsilon

#######
def eval_test():
    state = env.reset()
    eva_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
    # load the weights from file
    eva_log = data_prefix + 'checkpoint_' + s_model +'.pth' # '0626-1919checkpoint_dqn.pth'
    eva_agent.qnetwork_local.load_state_dict(torch.load(eva_log))
    eva_score = 0.
    for i in range(5):
        while True:
    #             print(state,"shape:" ,state.shape)
            action, act_vals = eva_agent.act(state, eps=0.00)
            next_state, reward, done, _ = env.step(action)  
            state = next_state 
            eva_score += reward 
            state = next_state
            if done:
                break
        env_info = env.reset()
    print("### Evaluation Phase & Report DQNs Test Score "+": {}".format(eva_score/float(5)))
    return eva_score/float(5)
#######

for i_episode in range(n_episodes):
    cnt = 0.
    total_cnt = 0.
    state = env.reset() # reset the environment # get the current state
    prev_state = state
    score = 0                                          # initialize the score
    while True:
        action, act_vals = agent.act(state, eps)
        if args.baseline == 0:
            if random.randint(0,100) < args.shellratio * 10:
                C_atk = True
            else:
                C_atk = False
        else:
            C_atk = False
        next_state, reward, done, _ = env.step(action)                  # see if episode has finished
        
        if C_atk == True:
            if args.Atktype == 1: ## Zero-out(0), Lag Attack (1), Gaussain Attack (2)
                fake_state, t_i = atker.Atck_F(state, prev_state, C_atk)
            elif args.Atktype == 2: ## Adversarial Attack
                fake_state, t_i = atker.Atck_Adv(state, action, reward, next_state, done, attack_network, agent.qnetwork_target, C_atk)
            else:
                raise ValueError("Attack type not supported")

            if args.causal == 1:
                agent.step(fake_state, action, reward, next_state, done, t_i) # add the experience to the agents replay memory
            else:
                agent.step(fake_state, action, reward, next_state, done, 0.) # no treatment diff
            cnt+=1.

        else:
            t_i = 0.
            agent.step(state, action, reward, next_state, done, t_i)
        score += reward
        if C_atk == False: ## Keep frozen the state - by not updating previous state
            prev_state = state                                # update the score
        state = next_state                             # roll over the state to next time step
        eps = max(eps_end, eps_decay*eps)              # decrease epsilon
        total_cnt +=1.
        if done:                                       # exit loop if episode finished
            break

    if i_episode == 0:
        torch.save(agent.qnetwork_local.state_dict(), '%scheckpoint_%s.pth' % (data_prefix, s_model))
    else:
        if score >= max(scores):
            torch.save(agent.qnetwork_local.state_dict(), '%scheckpoint_%s.pth' % (data_prefix, s_model))        
        # else:
        #     if score == max(scores):
        #         if agent.check_loss() < min(loss_book):
        #             torch.save(agent.qnetwork_local.state_dict(), '%scheckpoint_%s.pth' % (data_prefix, s_model))
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    scores_std.append(np.std(scores_window)) # save most recent std dev
    scores_avg.append(np.mean(scores_window)) # save most recent std dev
    scores_atkratio.append(round(cnt/total_cnt,2)) # save the attack ratio
    loss_book.append(agent.check_loss())    # else:
        #     agent.qnetwork_local.load_state_dict(torch.load(tmp_log))
    if i_episode % 20 == 0:
        print("With: ",round(100*cnt/total_cnt,2),"% timing attack", end = "\n")
        print('\rEpisode {}   Score: {:.2f}, Average Score: {:.2f}, Loss: {:.2f}'.format(i_episode, score, np.mean(scores_window), agent.check_loss()))
    if np.mean(scores_window)>=150.0:
        eval_test_score = eval_test()
        if eval_test_score >= 150:
            if np.mean(scores_window)>=195.0:
                torch.save(agent.qnetwork_local.state_dict(), '%scheckpoint_%s.pth' % (data_prefix, s_model))
                if eval_test_score >= 190:
                    s_msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                    print(s_msg.format(i_episode, np.mean(scores_window)))        
                    break


# save data to use latter
d_data = {'episodes': i_episode,
          'scores': scores,
          'scores_std': scores_std,
          'scores_avg': scores_avg,
          'scores_window': scores_window, 'atk_ratio': scores_atkratio, 'loss':loss_book}
pickle.dump(d_data, open('%ssim-data-%s.data' % (data_prefix, s_model), 'wb'))

d_data = pickle.load(open(data_prefix+'sim-data-dqn.data', 'rb'))
s_msg = 'Environment solved in {:d} episodes!\tAverage Score: {:.2f} +- {:.2f}'
print(s_msg.format(d_data['episodes'], np.mean(d_data['scores_window']), np.std(d_data['scores_window'])))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#recover data
na_raw = np.array(d_data['scores'])
na_mu = np.array(d_data['scores_avg'])
na_sigma = np.array(d_data['scores_std'])
record_attack = np.array(d_data['atk_ratio'])
loss_r = np.array(d_data['loss'])

# plot the scores
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# plot the sores by episode
ax1.plot(np.arange(len(na_raw)), na_raw)
ax1.set_xlim(0, len(na_raw)+1)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode # solved in ' + str(i_episode))
ax1.set_title('raw scores, red-solved 195')
ax1.axhline(y=195., xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.9, alpha=0.9)
ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax12.set_ylabel('atk ratio', color=color)  # we already handled the x-label with ax1
ax12.plot(np.arange(len(record_attack)), record_attack, color=color)
ax12.tick_params(axis='y', labelcolor=color)
# plot the average of these scores
ax2.plot(np.arange(len(na_mu)), na_mu)
ax2.fill_between(np.arange(len(na_mu)), na_mu+na_sigma, na_mu-na_sigma, facecolor='gray', alpha=0.1)
ax22 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

## Note: in Deep Q Network, the relation between loss and performance is still not very clear. Feel free if you want to see the loss of CIQs

# ax22.set_ylabel('loss')  # we already handled the x-label with ax1
# ax22.plot(np.arange(len(loss_r)), loss_r, 'c')
# ax22.tick_params(axis='y')

if args.evaluate == 1:
    # reset the environment
    print('############# Basic Evaluate #############', end = '\n')
        # get the default brain
    state = env.reset()
    eva_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
    # load the weights from file
    eva_log = data_prefix + 'checkpoint_' + s_model +'.pth'# '0626-1919checkpoint_dqn.pth'
    eva_agent.qnetwork_local.load_state_dict(torch.load(eva_log))
    eva_score = 0.
    for i in range(eval_num):
        while True:
#             print(state,"shape:" ,state.shape)
            action, act_vals = eva_agent.act(state, eps=0.00)
            next_state, reward, done, _ = env.step(action)  
            state = next_state 
            eva_score += reward 
            state = next_state
            if done:
                break
        env_info = env.reset()

print("Evaluate Score "+": {}".format(eva_score/float(eval_num)))
    
ax2.axhline(y=eva_score/float(eval_num), xmin=0.0, xmax=1.0, color='green', linestyle='--', linewidth=0.9, alpha=0.9)

if args.evaluate == 1:
    # reset the environment
    print('############# Noise Evaluate #############', end = '\n')
        # get the default brain
    state = env.reset()
    eva_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
    # load the weights from file
    eva_log = data_prefix + 'checkpoint_' + s_model +'.pth' # '0626-1919checkpoint_dqn.pth'
    eva_agent.qnetwork_local.load_state_dict(torch.load(eva_log))
    robust_score = 0.
    for i in range(eval_num):
        while True:
            if random.randint(0,100) < args.shellratio * 10:
                state, _ = atker.Atck_F(state, True)
#                 print(state,"shape:" ,state.shape)
            action, act_vals = eva_agent.act(state, eps=0.00)
            next_state, reward, done, _ = env.step(action) 
            state = next_state
            robust_score += reward
            state = next_state
            if done:
                break
        env_info = env.reset()

print("Robust Score " +": {}".format(robust_score/float(eval_num)))
ax2.axhline(y=robust_score/float(eval_num), xmin=0.0, xmax=1.0, color='black', linestyle='-.', linewidth=0.9, alpha=0.9)

ax2.set_ylabel('Scores')
ax2.set_xlabel('Episode solved in '+ str(i_episode))
ax2.set_title('avg-blue | validation-green:' + str(eva_score/float(eval_num))+'| robust:'+ str(robust_score/float(eval_num)))
f.tight_layout()

f.savefig(fig_prefix + '_dqn.pdf', format='pdf')


env.close()
