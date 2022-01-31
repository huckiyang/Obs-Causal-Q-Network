#   Obs-Causal-Q-Network-Demo



[Preprint](https://arxiv.org/pdf/2102.09677.pdf) | [Slides](https://docs.google.com/presentation/d/1WOgnMKszZ6eYwxiR0jLZjrj7XbmKpEf9sNrbI8poSMg/edit?usp=sharing) | [Colab Demo](https://colab.research.google.com/drive/1W0muo9IQMsQUIc4nLbB5VOKg7aXuWFR2?usp=sharing) | 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

### Environment Setup

- option 1 (from conda .yml under `conda 10.2` and `python 3.6`)

```shell
conda env create -f obs-causal-q-conda.yml 
```

- option 2 (from a clean python 3.6 and please follow the [setup](https://github.com/udacity/deep-reinforcement-learning#dependencies) of [UnityAgent 3D](https://github.com/Unity-Technologies/ml-agents) environment for Banana Navigator )

```shell
pip install torch torchvision torchaudio
pip install dowhy
pip install gym
```

### 1. Example of Training Causal Inference Q-Network (CIQ) on Cartpole

- Run Causal Inference Q-Network Training (`--network 1` for Treatment Inference Q-network)

```shell
python 0-cartpole-main.py --network 1
```

- Causal Inference Q-Network Architecture

<img src="https://github.com/huckiyang/Obs-Causal-Q-Network/blob/main/imgs/ciq_cartpole.png" width="500">

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


### 2. Example of Training a "Variational" Causal Inference Q-Network on Unity 3D Banana Navigator

- Run **Variational** Causal Inference Q-Networks (VCIQs) Training (`--network 3` for Causal **Variational** Inference)

```shell
python 1-banana-navigator-main.py --network 3
```

- Variational Causal Inference Q-Network Architecture

<img src="https://github.com/huckiyang/Obs-Causal-Q-Network/blob/main/imgs/variational_ciq.png" width="500">

- Output Logs

```python
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :

Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
Timing Atk Ratio: 10%
Using CEVAE_QNetwork.
Unity Worker id: 10  T: 1  Use baseline:  0  CEVAE:  1
With:  9.67 % timing attack
Episode 0   Score: 0.00, Average Score: 0.00
With:  11.0 % timing attack
Episode 5   Score: 1.00, Average Score: 0.17
With:  11.33 % timing attack
Episode 10   Score: 0.00, Average Score: 0.36
With:  10.33 % timing attack
Episode 15   Score: 0.00, Average Score: 0.56
...
Episode 205   Score: 10.00, Average Score: 9.25
With:  9.33 % timing attack
Episode 210   Score: 9.00, Average Score: 9.70
With:  9.0 % timing attack
Episode 215   Score: 10.00, Average Score: 11.10
With:  8.33 % timing attack
Episode 220   Score: 14.00, Average Score: 10.85
With:  12.33 % timing attack
Episode 225   Score: 19.00, Average Score: 11.70
With:  11.0 % timing attack
Episode 230   Score: 18.00, Average Score: 12.10
With:  7.67 % timing attack
Episode 235   Score: 21.00, Average Score: 11.60
With:  9.67 % timing attack
Episode 240   Score: 16.00, Average Score: 12.05

Environment solved in 242 episodes!     Average Score: 12.50
Environment solved in 242 episodes!     Average Score: 12.50 +- 4.87
############# Basic Evaluate #############
Using CEVAE_QNetwork.
Evaluate Score : 12.6
############# Noise Evaluate #############
Using CEVAE_QNetwork.
Robust Score : 12.5

```

## Reference

This fun work was initialzed when [Danny](https://www.linkedin.com/in/danny-hung/) and I first read the Causal Variational Model between 2018 to 2019 with the helps from Dr. [Yi Ouyang](https://scholar.google.com/citations?hl=en&user=dw_Sj_YAAAAJ) and Dr. [Pin-Yu Chen](https://scholar.google.com/citations?user=jxwlCUUAAAAJ&hl=en).

Please consider to reference the paper if you find this work helpful or relative to your research. 

- A non-archival and preliminary venue was presented in ICLR 2021 [Self-supervision for Reinforcement Learning Workshop](https://sslrlworkshop.github.io/), spotlight, long contributed talk. We appreciate the research community and the orgainzers.

```bib
@article{yang2021causal,
  title={Causal Inference Q-Network: Toward Resilient Reinforcement Learning},
  author={Yang, Chao-Han Huck and Hung, I and Danny, Te and Ouyang, Yi and Chen, Pin-Yu},
  journal={arXiv preprint arXiv:2102.09677},
  year={2021}
}
```
