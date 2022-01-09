#   Obs-Causal-Q-Network


AAAI 2022 - Training a Resilient Q-Network against Observational Interference


[Preprint](https://arxiv.org/pdf/2102.09677.pdf) | [Slides](https://docs.google.com/presentation/d/1WOgnMKszZ6eYwxiR0jLZjrj7XbmKpEf9sNrbI8poSMg/edit?usp=sharing)



### Example for Causal Inference Q-Network Training

- Run Causal Inference Q-Network Training

```
python 0-cartpole-main.py --network 1
```

- Output Logs

```python
observation space: Box(4,)
action space: Discrete(2)
Timing Atk Ratio: 10%
Using CEQNetwork_1. Number of Params: 41872
 Interference Type: 1  Use baseline:  0 use CGM:  1
With:  10.42 % timing attack
Episode 0   Score: 48.00, Average Score: 48.00, Loss: 1.71
With:  0.0 % timing attack
Episode 20   Score: 15.00, Average Score: 18.71, Loss: 30.56
With:  3.57 % timing attack
Episode 40   Score: 28.00, Average Score: 19.83, Loss: 36.36
With:  8.5 % timing attack
Episode 60   Score: 200.00, Average Score: 43.65, Loss: 263.29
With:  9.0 % timing attack
Episode 80   Score: 200.00, Average Score: 103.53, Loss: 116.35
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 193.4
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 164.2
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 147.8
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 193.4
With:  9.5 % timing attack
Episode 100   Score: 200.00, Average Score: 163.20, Loss: 77.38
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 198.4
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 197.8
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 197.6
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 198.6
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 199.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 186.8
Using CEQNetwork_1. Number of Params: 41872
### Evaluation Phase & Report DQNs Test Score : 200.0

Environment solved in 114 episodes!     Average Score: 195.55
Environment solved in 114 episodes!     Average Score: 195.55 +- 25.07
############# Basic Evaluate #############
Using CEQNetwork_1. Number of Params: 41872
Evaluate Score : 200.0
############# Noise Evaluate #############
Using CEQNetwork_1. Number of Params: 41872
Robust Score : 200.0
```

## Reference

This fun work was initialzed when Danny and I first read the Causal Variational Model between 2018 to 2019. 

Please consider to reference the paper if you find this work helpful or relative to your research. 

- A non-archival and preliminary venue was presented in ICLR 2021 Self-supervision for Reinforcement Learning Workshop, spotlight, long contributed talk

```bib
@article{yang2021causal,
  title={Causal Inference Q-Network: Toward Resilient Reinforcement Learning},
  author={Yang, Chao-Han Huck and Hung, I and Danny, Te and Ouyang, Yi and Chen, Pin-Yu},
  journal={arXiv preprint arXiv:2102.09677},
  year={2021}
}
```
