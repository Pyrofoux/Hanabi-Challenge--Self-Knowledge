"""Red Ranger."""

import random
from hanabi_learning_environment.rl_env import Agent
import knowledge as kn


class RedRanger(Agent):
  """Agent that tries to estimate it's own hand with
  a probability matrix."""

  myKnowledge = None

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    # print(config)
    self.myKnowledge = kn.Knowledge(config)


  def STOP(self):
      raise NameError("\n [MANUAL STOP]")

  def act(self, observation):
    """Act based on an observation."""

    self.myKnowledge.update(observation)

    if observation.cur_player_offset() == 0:
        #Plays a random legal move
      self.STOP()
      return random.choice(observation.legal_moves())
    else:
      return None
