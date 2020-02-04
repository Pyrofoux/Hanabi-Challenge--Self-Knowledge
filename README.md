# The Hanabi Project

Repository containing the code from our AI semester final project at the ENSEIRB and ENSC engineering schools. The long term goal is to build an agent capable of playing the Hanabi game, as it requires valuable skills for an artificial intelligence system. Our team focused on representing and updating knowledge in the game in a way that can be used to guess a player's own hand (its **self-knowledge**) but also its estimations about what other teammates know about their own hand (their own self-knowledge).

More details on the context and design can be found in the [project report](Report.pdf).

Team :
- Achraf El Khamsi (ENSEIRB)
- Jean-Marie Saindon (ENSEIRB)
- Younès Rabii (ENSC)

This repository is based on Deepmind's [`hanabi-learning-environment repository`](https://github.com/deepmind/hanabi-learning-environment).

## Installation

To install the environment, please follow these commands:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```

## Model Implementation

- The `Knowledge` class is located in the directory `hanabi_learning_environment/agents/`.
It be used by an agent to automatically update the Knowledge Representation model we designed.

- `RedRanger` is an agent using `Knowledge` to gather and update information about its own hand.
It's also the default agent created in the `game.py` script.

## Run Demo

To try `Redranger`, or start a game with a custom configuration:
- Go to `hanabi_learning_environment/agents/`
- Choose the game parameters in `game.py` last line
- Run the game with `$python game.py`

## Sample Screenshots
