import os
import time as ti
data_prefix = './data/'+ti.strftime("%m%d-%H%M")
s_currentpath = os.getcwd()
from unityagents import UnityEnvironment
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type = int, default= 0, help= " 0 - with an attack or 1 - yes, use baseline model " )
parser.add_argument("--Atktype", type = int, default= 1, help= "choose attack (treatment) type, 1 - attack_F, 2 = adversarial")
parser.add_argument("--causal", type = int, default= 1, help= "0 - Not use treatment info, 1 - use treatment info")
parser.add_argument("--network", type = int, default=1, help="0 - DQN, 1 - CEVAE, 2 - VAE, 3 - New CEVAE")
parser.add_argument("--history", type = int, default= 0, help= "0 - not show treatment history")
parser.add_argument("--attack_network_path", type = str, default="", help="path to attack model, must be specified if use adversarial attack.")
parser.add_argument("--evaluate", type = int, default= 1, help= "1 - yes, evaluate the trained agent, 0 - no" )
parser.add_argument("--Ftype", type = int, default= 0, help= "type of F: 1 - reverse, 2 - Random Noise, else - Zerout" )
parser.add_argument("--ratio", type = float, default= 0.2, help= "ratio of random time frame select" )
parser.add_argument("--shellratio", type = int, default= 1, help= "noise ratio for shell" )
parser.add_argument("--num_eval", type = int, default= 5, help= "number of evaluatel" )
parser.add_argument("--workid", type = int, default= 10, help= "worker id for Unity Rendering")

args = parser.parse_args()

env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64", worker_id= args.workid )


env_type = "fig_result" # path to save the result images
fig_prefix = "banana_"+ env_type  + ti.strftime("%m%d-%H%M") + '_Trt'+ str(args.Ftype)

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
from algorithms_step.banana_agent import DQNAgent, DDQNAgent, DDQNPREAgent
from algorithms_step.banana_model import SimpleQNetwork
from algorithms_step.attacker import atk_model



# Unity Environments contain brains which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[6]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space 
state = env_info.vector_observations[0]
# prev_state = state
state_size = len(state)

print("Timing Atk Ratio: " + str(10*args.shellratio) + "%" )

# And finally, we are going to train the model. We will consider that this environment is solved if the agent is able to receive an average reward (over 100 episodes) of at least +13.

eval_num = args.num_eval
n_episodes = 1000
eps_start = 1.
eps_end=0.01
eps_decay=0.998
max_t = 2000
s_model = 'dqn'
# ratio_t = int(max_t * args.ratio)
# ratio_t = int((max_t/10) * args.shellratio) 

# initialize
agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
atker = atk_model(Beta = 0.4, Ftype = args.Ftype)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.attack_network_path != "":
    attack_network = SimpleQNetwork(state_size=state_size, action_size=action_size, seed=0)
    state = torch.load(args.attack_network_path)
    attack_network.load_state_dict(state)
    attack_network = attack_network.to(device)
    attack_network.eval()

print("Unity Worker id:", args.workid, " T: " + str(args.Atktype), " Use baseline: ", str(args.baseline), " CEVAE: ", str(args.causal), end = "\n")

loss_book = []
scores = []
scores_std = []                    # List containing the std dev of the last 100 episodes
scores_avg = []                    # List containing the mean of the last 100 episodes
scores_atkratio = []
scores_window = deque(maxlen=20)  # last n scores
eps = eps_start                    # initialize epsilon

for i_episode in range(n_episodes):
    # treat_his = np.ones(max_t)
    cnt = 0.
    total_cnt = 0.
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0] # reset the environment # get the current state
    prev_state = state
    score = 0                                          # initialize the score
    # atk_list = np.random.choice(max_t, ratio_t)
    #while True: 
    for t in range(max_t):
        action, act_vals = agent.act(state, eps)
        if args.baseline == 0:
            if random.randint(0,100) < args.shellratio * 10:
                C_atk = True
            else:
                C_atk = False
        else:
            C_atk = False
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        
        if C_atk == True:
            if   args.Atktype == 1: ### (F = 0: Zerout) (F=1: Lag) (F=2 Gaussain)
                fake_state, t_i = atker.Atck_F(state, prev_state, C_atk)
            elif args.Atktype == 2: ### Adversarial need to put eplison to 0.3 || need check C&W
                fake_state, t_i = atker.Atck_Adv(state, action, reward, next_state, done, attack_network, agent.qnetwork_target, C_atk)
            else:
                raise ValueError("Attack type not supported")

            if args.causal == 1:
                agent.step(fake_state, action, reward, next_state, done, t_i) # add the experience to the agents replay memory
            else:
                agent.step(fake_state, action, reward, next_state, done, 0.) # no treatment diff
            
            #treat_his[t] = t_i # if t_1 = 0 update into treatment history 
            cnt+=1.

        else:
            # print("ok: ", t)
            t_i = 0.
            agent.step(state, action, reward, next_state, done, t_i)
        score += reward
        if C_atk == False: #keep frozen the state - by stop updating prev_state
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
        #     tmp_log = data_prefix + 'checkpoint_' + s_model +'.pth'# '0626-1919checkpoint_dqn.pth'
        #     agent.qnetwork_local.load_state_dict(torch.load(tmp_log))
    if i_episode % 5 == 0:
        print("With: ",round(100*cnt/total_cnt,2),"% timing attack", end = "\n")
        print('\rEpisode {}   Score: {:.2f}, Average Score: {:.2f}'.format(i_episode, score, np.mean(scores_window), ))
    if np.mean(scores_window)>=12.5:
        s_msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
        print(s_msg.format(i_episode, np.mean(scores_window)))        
        break


# save data to use latter
d_data = {'episodes': i_episode,
          'scores': scores,
          'scores_std': scores_std,
          'scores_avg': scores_avg,
          'scores_window': scores_window, 'atk_ratio': scores_atkratio,}
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

# plot the scores
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# plot the sores by episode
ax1.plot(np.arange(len(na_raw)), na_raw)
ax1.set_xlim(0, len(na_raw)+1)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode # solved in ' + str(i_episode))
ax1.set_title('raw scores, red-solved 195')
ax1.axhline(y=10., xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.9, alpha=0.9)
ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax12.set_ylabel('atk ratio', color=color)  # we already handled the x-label with ax1
ax12.plot(np.arange(len(record_attack)), record_attack, color=color)
ax12.tick_params(axis='y', labelcolor=color)
# plot the average of these scores
ax2.plot(np.arange(len(na_mu)), na_mu)
ax2.fill_between(np.arange(len(na_mu)), na_mu+na_sigma, na_mu-na_sigma, facecolor='gray', alpha=0.1)

# You need some tuning for the clean environment when using network 3 for Linux banana environments on banana_model.py
# if args.evaluate == 1:
#     # reset the environment
#     print('############# Basic Evaluate #############', end = '\n')
#         # get the default brain
#     brain_name = env.brain_names[0]
#     brain = env.brains[brain_name]
#     env_info = env.reset(train_mode=False)[brain_name]
#     # examine the state space 
#     state = env_info.vector_observations[0]

#     eva_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
#     # load the weights from file
#     eva_log = data_prefix + 'checkpoint_' + s_model +'.pth'# '0626-1919checkpoint_dqn.pth'
#     eva_agent.qnetwork_local.load_state_dict(torch.load(eva_log))
#     eva_score = 0.
#     for i in range(eval_num):
#         while True:
# #             print(state,"shape:" ,state.shape)
#             action, act_vals = eva_agent.act(state, eps=0.00)
#             env_info = env.step(action)[brain_name]  
#             next_state = env_info.vector_observations[0]   # geti the next state
#             reward = env_info.rewards[0]                   # get the reward
#             done = env_info.local_done[0]  
#             state = next_state 
#             eva_score += reward 
#             state = next_state
#             if done:
#                 break
#         env_info = env.reset(train_mode=False)[brain_name]

# print("Evaluate Score "+": {}".format(eva_score/float(eval_num)))
    
# ax2.axhline(y=eva_score/float(eval_num), xmin=0.0, xmax=1.0, color='green', linestyle='--', linewidth=0.9, alpha=0.9)

if args.evaluate == 1:
    # reset the environment
    print('############# Noise Evaluate #############', end = '\n')
        # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    # examine the state space 
    state = env_info.vector_observations[0]
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
            env_info = env.step(action)[brain_name]  
            next_state = env_info.vector_observations[0]   # geti the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0] 
            state = next_state 
            robust_score += reward 
            state = next_state
            if done:
                break
        env_info = env.reset()

print("Robust Score " +": {}".format(robust_score/float(eval_num)))
ax2.axhline(y=robust_score/float(eval_num), xmin=0.0, xmax=1.0, color='black', linestyle='-.', linewidth=0.9, alpha=0.9)

ax2.set_ylabel('Scores')
ax2.set_xlabel('Episode  solved in '+ str(i_episode))
ax2.set_title('avg-blue | validation-green:' + str(eva_score/float(eval_num))+'| robust:'+ str(robust_score/float(eval_num)))
    
f.tight_layout()

f.savefig( env_type +'/' + fig_prefix + '_dqn.pdf', format='pdf')

env.close()
