import os
import time as ti

# fig_prefix = 'figures/'+ti.strftime("%m%d-%H%M")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type = float, default= 0.40, help="for Atk_C theshold")
parser.add_argument("--baseline", type = int, default= 0, help= " 0 - with an attack or 1 - yes, use baseline model " )
# parser.add_argument("-v", "--verbosity", action="count", default=0)
parser.add_argument("--Atktype", type = int, default= 1, help= "choose attack (treatment) type, 1 - attack_F, 2 = adversarial")
parser.add_argument("--causal", type = int, default= 1, help= "0 - Not use treatment info, 1 - use treatment info")
parser.add_argument("--network", type = int, default=0, help="0 - DQN, 1 - CEVAE, 2 - VAE, 3 - New CEVAE")
parser.add_argument("--history", type = int, default= 0, help= "0 - not show treatment history")
parser.add_argument("--attack_network_path", type = str, default="", help="path to attack model, must be specified if use adversarial attack.")
parser.add_argument("--evaluate", type = int, default= 1, help= "1 - yes, evaluate the trained agent, 0 - no" )
parser.add_argument("--Ftype", type = int, default= 0, help= "type of F: 1 - Lag & Frozen Screen, 2 - Random Noise, else 0 - Zerout" )
parser.add_argument("--ratio", type = float, default= 0.2, help= "ratio of random time frame select" )
parser.add_argument("--shellratio", type = int, default= 1, help= "noise ratio for shell" )
parser.add_argument("--num_eval", type = int, default= 5, help= "number of evaluatel" )
args = parser.parse_args()
# answer = args.x**args.y
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

# env_type = "fig_test"
# fig_prefix = env_type + '/reply_' + ti.strftime("%m%d-%H%M") + '_Trt'+ str(args.Ftype)
data_prefix = './data/'+ti.strftime("%m%d-%H%M")
s_currentpath = os.getcwd()

action_size = 2
state_size = 4
print("Causal Analysis on Atk Ratio: " + str(10*args.shellratio) + "%" )

# And finally, we are going to train the model. We will consider that this environment is solved if the agent is able to receive an average reward (over 100 episodes) of at least +13.

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
atker = atk_model(Beta = args.beta, Ftype = args.Ftype)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eval_num = args.num_eval

## Init

df = {'Z{}'.format(i): [] for i in range(256)}
df['y'] = []
df['v'] = []
#df['r'] = []

s_model = '0921-2314checkpoint_dqn'
if args.evaluate == 1:
    # reset the environment
    print('############# Causal Data Generation #############', end = '\n')
        # get the default brain
    state = env.reset()
    prev_state = state
    eva_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, select_network=args.network)
    # load the weights from file
    #eva_log = data_prefix + 'checkpoint_' + s_model +'.pth'# '0626-1919checkpoint_dqn.pth'
    eva_log = 'weights/0921-2314checkpoint_dqn.pth'
    eva_agent.qnetwork_local.load_state_dict(torch.load(eva_log))
    robust_score = 0.

    for i in range(eval_num):
        while True:
            Atk_test = False
            if random.randint(0,100) < args.shellratio * 10:
                Atk_test = True
                state, _ = atker.Atck_F(state, prev_state, Atk_test)
#                 print(state,"shape:" ,state.shape)
            action, act_vals, causal = eva_agent.act_causal(state, eps=0.00)
            next_state, reward, done, _ = env.step(action)  
            eva_agent.step_causal(state, action, reward, next_state, done, Atk_test)
            df['v'].append(Atk_test)
            # |max_a Q(z, a) - reward|
            df['y'].append(abs(causal['q'] - reward))
            for j in range(256):
                df['Z{}'.format(j)].append(causal['z'][0][j])
            
            if Atk_test == False: ## if attack frozen previous state - not update previous 
                prev_state = state
            state = next_state 
            robust_score += reward 
            state = next_state
            if done:
                break
        env_info = env.reset()

### Save the date
import logging
import dowhy
from dowhy.do_why import CausalModel
import dowhy.datasets

df = pd.DataFrame(data=df)
data = dowhy.datasets.linear_dataset(beta=10,
        num_common_causes=0,
        num_instruments=256,
        num_samples=1000,
        treatment_is_binary=True)

cevae_graph = 'graph[directed 1node[ id "v" label "v"]node[ id "y" label "y"] edge[source "v" target "y"]'
cevae_graph += ''.join(['node[ id "Z{}" label "Z{}"] edge[ source "Z{}" target "v"]'.format(i, i, i) for i in range(256)])
cevae_graph += ''.join(['edge[ source "Z{}" target "y"]'.format(i, i, i) for i in range(256)])
cevae_graph += ']'

data["gml_graph"] = cevae_graph

model=CausalModel(
        data = df,
        treatment = 'v',
        outcome = 'y',
        graph = cevae_graph,
        instruments = data["instrument_names"],
        logging_level = logging.INFO
        )

identified_estimand = model.identify_effect()

causal_estimate_reg = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)

print("Causal Estimate is " + str(causal_estimate_reg.value))
causal_estimate = causal_estimate_reg

print("Robust Score " +": {}".format(robust_score/float(eval_num)))
ax2.axhline(y=robust_score/float(eval_num), xmin=0.0, xmax=1.0, color='black', linestyle='-.', linewidth=0.9, alpha=0.9)

# ax2.set_ylabel('Scores')
# ax2.set_xlabel('Episode # solved in '+ str(i_episode))
# ax2.set_title('avg-blue | validation-green:' + str(eva_score/float(eval_num))+'| robust:'+ str(robust_score/float(eval_num)))
    
# f.tight_layout()
