# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code demonstrating the Python Hanabi interface."""

from __future__ import print_function
from bcolors import bcolors
import numpy as np
from hanabi_learning_environment import pyhanabi
import hanabi_learning_environment.agents as rdcustom

from red_ranger import RedRanger
#from rdcustom import  random_agent_custom
#from hanabi_learning_environment.agents import random_agent_custom
#import hanabi_learning_environment.agents.random_agent_custom


def run_game(config):
    """Play a game, selecting random actions."""

    def print_state(state):
        """Print some basic information about the state."""
        print("")
        print("Current player: {}".format(state.cur_player()))
        print(state)

        # Example of more queries to provide more about this state. For
        # example, bots could use these methods to to get information
        # about the state in order to act accordingly.
        print("### Information about the state retrieved separately ###")
        print("### Information tokens: {}".format(state.information_tokens()))
        print("### Life tokens: {}".format(state.life_tokens()))
        print("### Fireworks: {}".format(state.fireworks()))
        print("### Deck size: {}".format(state.deck_size()))
        print("### Discard pile: {}".format(str(state.discard_pile())))
        print("### Player hands: {}".format(str(state.player_hands())))
        print("")

    def print_observation(observation):
        """Print some basic information about an agent observation."""
        print("--- Observation ---")
        print(observation)

        print("### Information about the observation retrieved separately ###")
        print("### Current player, relative to self: {}".format(
            observation.cur_player_offset()))
        print("### Observed hands: {}".format(observation.observed_hands()))
        print("### Card knowledge: {}".format(observation.card_knowledge()))
        print("### Discard pile: {}".format(observation.discard_pile()))
        print("### Fireworks: {}".format(observation.fireworks()))
        print("### Deck size: {}".format(observation.deck_size()))
        move_string = "### Last moves:"
        for move_tuple in observation.last_moves():
          move_string += " {}".format(move_tuple)
        print(move_string)
        print("### Information tokens: {}".format(observation.information_tokens()))
        print("### Life tokens: {}".format(observation.life_tokens()))
        print("### Legal moves: {}".format(observation.legal_moves()))
        print("--- EndObservation ---")

    def print_encoded_observations(encoder, state, num_players):
        print("--- EncodedObservations ---")
        print("Observation encoding shape: {}".format(encoder.shape()))
        print("Current actual player: {}".format(state.cur_player()))
        for i in range(num_players):
          print("Encoded observation for player {}: {}".format(
              i, encoder.encode(state.observation(i))))
        print("--- EndEncodedObservations ---")




    ### Creating an instance of the Hanabi game
    game = pyhanabi.HanabiGame(config)
    #print(game.parameter_string(), end="")
    obs_encoder = pyhanabi.ObservationEncoder(
      game,
      enc_type=pyhanabi.ObservationEncoderType.CANONICAL
    )


    ### Initialize players
    players_number = config['players']
    players = [RedRanger(config, game, playerIndex) for playerIndex in range(players_number)]


    ### Initialize the state of the Hanabi Game
    state = game.new_initial_state()


    ### Main loop of the game
    nb_round = 1
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
          state.deal_random_card()
          continue

        active_player = players[state.cur_player()]

        ##print
        print(bcolors.BACKGROUNDGREEN + "------------------------------ ROUND " + str(nb_round) + " --------------------------------" + bcolors.WHITE)
        nb_round += 1
        print("")
        print(bcolors.LIGHTGREEN + "ACTIVE PLAYER: " + str(state.cur_player()) + bcolors.WHITE)
        print("")
        ##

        observation = state.observation(state.cur_player())
        legal_moves = state.legal_moves()
        move = active_player.act(observation)

        ##print
        print(bcolors.LIGHTGREEN + "PLAYER " + str(state.cur_player()) + " HAND" + bcolors.WHITE)
        print(state.player_hands()[state.cur_player()])
        print("")
        print(bcolors.LIGHTGREEN + "GAME STATE" + bcolors.WHITE)
        print(bcolors.LIGHTRED + "Fireworks: " + bcolors.WHITE + str(state.fireworks()) + "   (number of cards in each stack)")
        print(bcolors.LIGHTRED + "Player Hands: " + bcolors.WHITE + str(state.player_hands()))
        print(bcolors.LIGHTRED + "Discard Pile: " + bcolors.WHITE + str(state.discard_pile()))
        print(bcolors.LIGHTRED + "Info Tokens: " + bcolors.WHITE + str(state.information_tokens()))
        print(bcolors.LIGHTRED + "Life Tokens: " + bcolors.WHITE + str(state.life_tokens()))
        print("")
        print(bcolors.LIGHTGREEN + "MOVE PLAYED" + bcolors.WHITE)
        print("Chose random legal move: {}".format(move))
        print("")
        ##

        state.apply_move(move)

    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    run_game({"players": 3, "random_start_player": False, "colors":3, "ranks":3, "hand_size":3})
