import os
import time as ti

fig_prefix = 'results/'+ti.strftime("%m%d-%H%M")
data_prefix = 'data/'+ti.strftime("%m%d-%H%M")
s_currentpath = os.getcwd()

from unityagents import UnityEnvironment
import sys
import os
import pandas as pd
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, default='dqn', help="dqn, ddqn, pre")
parser.add_argument("--workid", type = int, default= 0, help= "worker id for Unity Rendering")
args = parser.parse_args()


env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64", worker_id = args.workid )


# Unity Environments contain brains which are responsible for deciding the actions of their 

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)


# And finally, we are going to train the model. We will consider that this environment is 
# solved if the agent is able to receive an average reward (over 100 episodes) of at least +13.


import gym
import pickle
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import DQNAgent, DDQNAgent, DDQNPREAgent

n_episodes = 1000
eps_start = 1.
eps_end=0.01
eps_decay=0.995
max_t = 1000


if args.model == 'dqn':
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)
elif args.model == 'ddqn':
    agent = DDQNAgent(state_size=state_size, action_size=action_size, seed=0)
else:
    agent = DDQNPREAgent(state_size=state_size, action_size=action_size, seed=0)

scores = []                        # list containing scores from each episode
scores_std = []                    # List containing the std dev of the last 100 episodes
scores_avg = []                    # List containing the mean of the last 100 episodes
scores_window = deque(maxlen=100)  # last 100 scores
eps = eps_start                    # initialize epsilon

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    for t in range(max_t):
        # action = np.random.randint(action_size)        # select an action
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.step(state, action, reward, next_state, done)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    scores_std.append(np.std(scores_window)) # save most recent std dev
    scores_avg.append(np.mean(scores_window)) # save most recent std dev
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=10.0:
        s_msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
        print(s_msg.format(i_episode, np.mean(scores_window)))
        torch.save(agent.qnet.state_dict(), '%scheckpoint_%s.pth' % (data_prefix, args.model))
        break

env.close()
        
# save data to use latter
d_data = {'episodes': i_episode,
          'scores': scores,
          'scores_std': scores_std,
          'scores_avg': scores_avg,
          'scores_window': scores_window}
pickle.dump(d_data, open('%ssim-data-%s.data' % (data_prefix, args.model), 'wb'))


import pickle

d_data = pickle.load(open(data_prefix+'sim-data-'+ args.model + '.data', 'rb'))
s_msg = 'Environment solved in {:d} episodes!\tAverage Score: {:.2f} +- {:.2f}'
print(s_msg.format(d_data['episodes'], np.mean(d_data['scores_window']), 
np.std(d_data['scores_window'])))


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#recover data
na_raw = np.array(d_data['scores'])
na_mu = np.array(d_data['scores_avg'])
na_sigma = np.array(d_data['scores_std'])

# plot the scores
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# plot the sores by episode
ax1.plot(np.arange(len(na_raw)), na_raw)
ax1.set_xlim(0, len(na_raw)+1)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode #')
ax1.set_title('raw scores')

# plot the average of these scores
ax2.axhline(y=10., xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9)
ax2.plot(np.arange(len(na_mu)), na_mu)
ax2.fill_between(np.arange(len(na_mu)), na_mu+na_sigma, na_mu-na_sigma, facecolor='gray', alpha=0.1)
ax2.set_ylabel('Average Score')
ax2.set_xlabel('Episode #')
ax2.set_title('average scores')

f.tight_layout()


# In[17]:

# f.savefig(fig_prefix + 'dqn.eps', format='eps', dpi=1200)
f.savefig(fig_prefix + args.model+ '.pdf', format='pdf')

