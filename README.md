# Encouraging Human-Like Gameplay in Deep Reinforcement Learning Hanabi Agents

### Abstract:
The card game of Hanabi has become a benchmark for multi-agent reinforcement learning and machine cooperation due to its unique blend of cooperative gameplay and imperfect information. While recent models have excelled in self-play, there has been slower progress in developing agents that can effectively collaborate with agents with whom they have not trained, especially humans. In this paper, I modify the Rainbow RL agent and implement the Other-Play algorithm [3] and an intrinsic rewards model. These additions enhance the agents by looking for more robust strategies that exploit the presence of known symmetries in the problem as well as guiding the agent’s training according to known good-play principles. I show that the modified agent performs equivalently well in self-play, and it performs significantly better in cross-play settings. In future work, I hope to train the agent more to further improve the model and evaluate the modified agent’s performance with human players.

Please refer to the hanabi-learning-environment: https://github.com/deepmind/hanabi-learning-environment

The instructions to run this locally follow from above:

### Getting started
Install the learning environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```
Run the examples:
```
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```
Go to the Rainbow agent:
```
cd hanabi_learning_environment/agents/rainbow/configs
```
Edit the ```hanabi_rainbow.gin``` file to change the parameters and toggle both the Intrinsic Rewards (IR) and Other-Play card shuffling (OP).

### To enable IR:
Set ```DQNAgent.goir``` to ```True```

### To enable OP:
Set ```create_environment.color_shuffle``` to ```True```

To train the agent, from ```hanabi-learning-environment/hanabi_learning_environment/agents/rainbow```, run:
```
python -um train \
  --base_dir=/tmp/hanabi_rainbow \
  --gin_files='configs/hanabi_rainbow.gin'
```

The ```base_dir``` flag is the directory in which to store the logs and checkpoints. 
