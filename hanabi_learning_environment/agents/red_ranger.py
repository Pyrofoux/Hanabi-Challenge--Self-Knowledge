"""Red Ranger."""

import random
from hanabi_learning_environment.rl_env import Agent
from Knowledge import Knowledge
from hanabi_learning_environment.pyhanabi import HanabiMoveType

class RedRanger(Agent):
    """Agent that tries to estimate it's own hand with
    a probability matrix."""

    knowledge = None

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        game = args[0]
        playerIndex = args[1]
        self.knowledge = Knowledge(config, game, playerIndex)


    def act(self, observation):
        """Act based on an observation."""
        self.knowledge.update(observation)

        if observation.cur_player_offset() == 0:
            move = random.choice(observation.legal_moves())
            move_type = move.type()
            if (move_type == HanabiMoveType.PLAY or move_type == HanabiMoveType.DISCARD):
                self.knowledge.initialize_new_card(move.card_index())
            return move
        else:
          return None
