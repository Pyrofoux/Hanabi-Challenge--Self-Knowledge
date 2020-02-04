# Hanabi Project

This repository is based on the hanabi-learning-environment repository of deepmind.

## Installation

In order to install the environment, please follow these commands:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```

## Our implementation

- You can find our `Knowledge` class in the file of the same name, in the directory `hanabi_learning_environment/agents/`.
- `RedRanger` is the default class that uses the knowledge structure
- `RedRanger` simply updates its self-knowledge but does not use it for the moment
- It's also the default agent created in the `game.py` file

## Run a game

To run our agent, or start a game with a custom configuration, go to `hanabi_learning_environment/agents/`:
- Choose the game parameters in  `game.py`, in the last line of the file
- Run the game with `$python game.py`
