from __future__ import division
import numpy as np
from bcolors import bcolors



class Knowledge:
    """This is the class that manages the knowledge of the player about his hand.
    The knowledge about each card in the player's hand is represented by two probability vectors:
        - A probability vector for the color of the card   (indexed in this order: RYGWB)
        - A probability vector for the rank of the card


    We keep the count of unknown cards in two arrays:
        - An array of unknown colors  (indexed by the number of colors in the game)
        - An array of unknonw ranks   (indexed by the number of ranks in the game)

        Each element in these arrays represents the number of cards of a particular color or rank
        that are not yet revealed to us. We calculate this number based on:
        - The cards that are visible to us
            - stacked cards (fireworks)
            - discard pile
            - other player's hands
        - The cards in our hand with a revealed rank or color (revelation by hint or by deduction)


    The hints about each card in the payer's hand is stored in an array as well
    """

    def __init__(self, config, game, playerIndex):
        #configuration
        self.config = config
        self.players = config['players'] if 'players' in config else 5
        self.colors = config['colors'] if 'colors' in config else 5
        self.ranks = config['ranks'] if 'ranks' in config else 5
        self.hand_size = config['ranks'] if 'ranks' in config else 5
        self.game = game
        self.index = playerIndex #0-based player index

        #knowledge
        self.unknown_cards = UnknownCards(self) #unknown_colors & unknown_ranks
        self.proba_vectors = [ProbaVectors(self) for cardIndex in range(self.hand_size)]  #indexed by the player's hand
        self.hints = [[] for card in range(self.hand_size)]  #indexed by the player's hand


    def update(self, observation):
        """ Main update function of our knowledge
            it orchestrates between all the update functions
        """

        self.update_proba_vectors_v1(observation)
        self.update_unknown_cards(observation)
            #update_proba_vectors_v2
                #update_unknown_cards
                    #update_proba_vectors_v2
                        # ... (until the information propagates)

        self.print_knowledge()
        return


    def update_proba_vectors_v1(self, observation):
        """ Update the probability vectors of our cards according
        to the hints given to us during the last rounds.

        For example, if our probability vector for the rank of the first card
        in our hand is [0.5, 0.25, 0.25] and that a teammate gives us a hint
        about the rank of this card (this card being of rank 1). The probability
        vector will become [1, 0, 0]
        """

        lastMoves = observation.last_moves()

        ##print
        print(bcolors.LIGHTGREEN + "LAST MOVES " + bcolors.WHITE)
        print(lastMoves)
        print("")
        ##

        for i in range(len(lastMoves)):
            card_indexes_revealed = lastMoves[i].card_info_newly_revealed()
            if (card_indexes_revealed):
                move = lastMoves[i].move()
                player_offset = lastMoves[i].player() #offset of the player of this move
                target_offset = move.target_offset()
                target_index = (self.index + player_offset + target_offset) % self.players
                if (target_index == self.index):
                    color_index = move.color()
                    rank_index = move.rank()
                    #direct hints
                    for card_index in card_indexes_revealed:
                        hint = Hint(color_index=color_index, rank_index=rank_index)
                        self.hints[card_index].append(hint)
                        self.proba_vectors[card_index].apply_hint(hint)

                    #indirect hints
                    generator = (card_index for card_index in range(self.hand_size) if card_index not in card_indexes_revealed)  #generator expression
                    for card_index in generator:
                        hint = Hint(not_color_index=color_index, not_rank_index=rank_index)
                        self.hints[card_index].append(hint)
                        self.proba_vectors[card_index].apply_hint(hint)




    def update_proba_vectors_v2(self, observation):
        """ Update the probability vectors of our cards according
        to the number of unknown cards in the game.

        We just update the probability vectors that are not specific, which means
        the probability vectors that don't have 1 in a single component and 0 in
        all others.

        For example, if the array of the unknown ranks is [2, 1, 0]:
            - if the proba vector corresponding to the rank of a card is
              [0.33, 0.33, 0.33], it will become [0.66, 0.33, 0].
            - if the proba vector corresponding to the rank of a card is
              [0, 1, 0], it will not change.

        <-> If the rank or the color of a card is revealed for the first time,
            we call the update_unknown_cards function to update the unknown_cards
            arrays.
        """

        flag = False  #whether we should call update_unknown_cards or not
        sum_unknown_colors = self.unknown_cards.sum_unknown_colors()
        sum_unknown_ranks = self.unknown_cards.sum_unknown_ranks()

        for proba_vector in self.proba_vectors:
            color_index = self.specific_proba_vector(proba_vector.proba_color)
            rank_index = self.specific_proba_vector(proba_vector.proba_rank)

            if (color_index < 0):
                proba_vector.proba_color = np.array(
                    [ self.unknown_cards.unknown_colors[color_index] / sum_unknown_colors
                        for color_index in range(self.colors)
                    ]
                )
                if (not(flag) and self.specific_proba_vector(proba_vector.proba_color) >= 0):
                    flag = True
            if (rank_index < 0):
                proba_vector.proba_rank = np.array(
                    [ self.unknown_cards.unknown_ranks[rank_index] / sum_unknown_ranks
                        for rank_index in range(self.ranks)
                    ]
                )
                if (not(flag) and self.specific_proba_vector(proba_vector.proba_rank) >= 0):
                    flag = True

        # Call (on condition) update_unknown_cards
        if (flag):
            self.update_unknown_cards(observation)


    def update_unknown_cards(self, observation):
        """ Update the unknown cards table based on the number of revealed cards
            on the game, which includes:

                - The visible information:
                    - Stacked Cards (fireworks)
                    - Discard Pile
                    - Other player's hands

                - Cards in our hand that we are sure of their rank or color

            The premise is that we will count the number of revealed cards for
            each rank and color, and subtract it from the total number of cards
            for each rank and color.

            <-> If there is a difference between the previous state of the unknown_cards
                arrays and the new, we call the update_proba_vectors_v2 function to
                update the probability vectors.
        """

        old_unknown_cards = self.unknown_cards
        new_unknown_cards = UnknownCards(self)
        fireworks = observation.fireworks()
        discard_pile = observation.discard_pile()
        observed_hands = observation.observed_hands()

        ## stacked cards (fireworks)
        for color_index in range(len(fireworks)):
            n_cards = fireworks[color_index]
            if (n_cards > 0):
                new_unknown_cards.subtract_n_color(color_index, n_cards)
                for rank_index in range(n_cards):
                    new_unknown_cards.subtract_n_rank(rank_index, 1)

        ## discard pile
        for card in discard_pile:
            new_unknown_cards.subtract_n_color(card.color(), 1)
            new_unknown_cards.subtract_n_rank(card.rank(), 1)

        ## hands of other players
        for hand in observed_hands:
            for card in hand:
                new_unknown_cards.subtract_n_color(card.color(), 1)
                new_unknown_cards.subtract_n_rank(card.rank(), 1)

        ## Our own cards
        # Based on proba_vectors
        for proba_vector in self.proba_vectors:
            color_index = self.specific_proba_vector(proba_vector.proba_color)
            rank_index = self.specific_proba_vector(proba_vector.proba_rank)

            if (color_index >= 0):
                new_unknown_cards.subtract_n_color(color_index, 1)
            if (rank_index >= 0):
                new_unknown_cards.subtract_n_rank(rank_index, 1)
        # Based on Hints
        # for card_index in range(self.hand_size):
        #     for hint in self.hints[card_index]:
        #         if (hint.color_index != -1):
        #             new_unknown_cards.subtract_n_color(hint.color_index, 1)
        #         elif (hint.rank_index != -1):
        #             new_unknown_cards.subtract_n_rank(hint.rank_index, 1)

        # Call (on condition) update_proba_vectors_v2
        unknown_colors_comp = compare_arrays(old_unknown_cards.unknown_colors, new_unknown_cards.unknown_colors)
        unknown_ranks_comp = compare_arrays(old_unknown_cards.unknown_ranks, new_unknown_cards.unknown_ranks)
        if (unknown_colors_comp == False or unknown_ranks_comp == False):
            self.unknown_cards = new_unknown_cards
            self.update_proba_vectors_v2(observation)


    def initialize_new_card(self, card_index):
        """
        Update our probability vectors because the card card_index was played or discarded
        The proba_vectors elements will probably be shifted and a new probaVectors will be
        initialized at the end of the player's hand
        """

        for i in range(card_index, self.hand_size-1):
            self.proba_vectors[i] = self.proba_vectors[i+1]
            self.hints[i] = self.hints[i+1]

        self.proba_vectors[self.hand_size-1] = ProbaVectors(self)
        self.hints[self.hand_size-1] = []


    def specific_proba_vector(self, proba_vector):
        """ Check if the probability vector has 1 in a single component and 0 in all others
            - in which case, it returns the index of the color or the rank corresponding to
            the probability value 1

            - otherwise, it returns -1
        """

        for index in range(len(proba_vector)):
            if (proba_vector[index] == 1):
                return index
        return -1


    def print_knowledge(self):
        print(bcolors.LIGHTGREEN + "PLAYER " + str(self.index) + " KNOWLEDGE" + bcolors.WHITE)
        print(bcolors.LIGHTRED + "  Unknown Colors" + bcolors.WHITE)
        print("    " + str(self.unknown_cards.unknown_colors))
        print(bcolors.LIGHTRED + "  Unknown Ranks" + bcolors.WHITE)
        print("    " + str(self.unknown_cards.unknown_ranks))
        print(bcolors.LIGHTRED + "  Probability Vectors" + bcolors.WHITE)
        for i in range(len(self.proba_vectors)) :
            print("         Card " + str(i))
            print(bcolors.CYAN + "           proba_color" + bcolors.WHITE)
            print("             " + str(self.proba_vectors[i].proba_color))
            print(bcolors.CYAN + "           proba_rank" + bcolors.WHITE)
            print("             " + str(self.proba_vectors[i].proba_rank))
            print(bcolors.CYAN + "           Hints" + bcolors.WHITE)
            print("             " + str(self.hints[i]))
            print("")

    def __str__(self):
        return ""

    def __repr__(self):
        return self.__str__()


class UnknownCards:
    """ The number of unknown cards per color and rank

        We keep the count of unknown cards in two arrays:
            - An array of unknown colors  (indexed by the number of colors in the game)
            - An array of unknonw ranks   (indexed by the number of ranks in the game)

        Each element in these arrays represents the number of cards of a particular color or rank
        that are not yet revealed to us.
    """

    def __init__(self, knowledge):
        self.unknown_colors = np.array([self.cards_per_color(knowledge) for i in range(knowledge.colors)])
        self.unknown_ranks  = np.array([ self.cards_per_rank(knowledge, rank) for rank in range(knowledge.ranks)])

    def cards_per_color(self, knowledge):
        """ Calculate the total number of cards per color """
        cards_per_color = 0
        for rank in range(knowledge.ranks):
            cards_per_color += knowledge.game.num_cards(0, rank)
        return cards_per_color

    def cards_per_rank(self, knowledge, rank):
        """ Calculate the total number of cards per rank """
        return knowledge.colors * knowledge.game.num_cards(0, rank)

    def sum_unknown_colors(self):
        """ Calculate the total number of unknown color cards """
        sum=0
        for n in self.unknown_colors:
            sum += n
        return sum

    def sum_unknown_ranks(self):
        """ Calculate the total number of unknown rank cards """
        sum=0
        for n in self.unknown_ranks:
            sum += n
        return sum

    def subtract_n_color(self, color_index, n):
        """ subtract n from the count of unknown cards related to the color color_index """

        if (color_index != -1):
            self.unknown_colors[color_index] -= 1

    def subtract_n_rank(self, rank_index, n):
        """ subtract n from the count of unknown cards related to the rank rank_index """

        if (rank_index != -1):
            self.unknown_ranks[rank_index] -= 1



class ProbaVectors:

    """ The knowledge and estimations about the color and rank of a card.
        the estimations are represented by two probability vectors:
            - A probability vector for the color of the card   (indexed in this order: RYGWB)
            - A probability vector for the rank of the card
    """

    def __init__(self, knowledge):
        sum_unknown_colors = knowledge.unknown_cards.sum_unknown_colors()
        sum_unknown_ranks = knowledge.unknown_cards.sum_unknown_ranks()

        self.proba_color = np.array(
            [ knowledge.unknown_cards.unknown_colors[color_index] / sum_unknown_colors
                for color_index in range(knowledge.colors)
            ]
        )
        self.proba_rank = np.array(
            [ knowledge.unknown_cards.unknown_ranks[rank_index] / sum_unknown_ranks
                for rank_index in range(knowledge.ranks)
            ]
        )
        self.colors = knowledge.colors
        self.ranks = knowledge.ranks

    def apply_hint(self, hint):
        """ Apply the hint given to the card linked to these two proba vectors """

        if (hint.color_index != -1):
            self.set_to_color(hint.color_index)
        elif (hint.rank_index != -1):
            self.set_to_rank(hint.rank_index)
        elif (hint.not_color_index != -1):
            self.set_to_not_color(hint.not_color_index)
        elif (hint.not_rank_index != -1):
            self.set_to_not_rank(hint.not_rank_index)
        else:
            pass

    def set_to_color(self, color_index):
        """ set proba_color to 100% for a particular color_index (0-based) """
        self.proba_color[color_index] = 1
        for i in range(self.colors):
            if (i != color_index):
                self.proba_color[i] = 0

    def set_to_rank(self, rank_index):
        """ set proba_rank to 100% for a particular rank_index (0-based) """
        self.proba_rank[rank_index] = 1
        for i in range(self.ranks):
            if (i != rank_index):
                self.proba_rank[i] = 0

    def set_to_not_color(self, not_color_index):
        #get the indexes of non-zero values
        if (self.proba_color[not_color_index] == 0):
            return

        non_zero_indexes = []
        generator = (color_index for color_index in range(len(self.proba_color)) if color_index != not_color_index)  #generator expression
        for color_index in generator:
            if (self.proba_color[color_index] != 0):
                non_zero_indexes.append(color_index)

        if (len(non_zero_indexes) != 0):
            self.proba_color[not_color_index] = 0
            proba = self.proba_color[not_color_index] / len(non_zero_indexes)
            for color_index in non_zero_indexes:
                self.proba_color[color_index] = proba

        # print("SET_TO_NON_COLOR " + str(not_color_index) + " " + str(non_zero_indexes))

    def set_to_not_rank(self, not_rank_index):
        #get the indexes of non-zero values
        if (self.proba_rank[not_rank_index] == 0):
            return

        non_zero_indexes = []
        generator = (rank_index for rank_index in range(len(self.proba_rank)) if rank_index != not_rank_index)  #generator expression
        for rank_index in generator:
            if (self.proba_rank[rank_index] != 0):
                non_zero_indexes.append(rank_index)

        if (len(non_zero_indexes) != 0):
            self.proba_rank[not_rank_index] = 0
            proba = self.proba_rank[not_rank_index] / len(non_zero_indexes)
            for rank_index in non_zero_indexes:
                self.proba_rank[rank_index] = proba

        # print("SET_TO_NON_RANK" + str(not_rank_index) + " " + str(non_zero_indexes))

    def getProbaMatrix(self):
        return np.outer(self.proba_color, self.proba_rank)

    def getProbaCard(self, color_index, rank_index):
        return self.proba_color[color_index]*self.proba_rank[rank_index]


class Hint:
    """ A hint about the color or rank of a card
        A card will be associated with many instances of this class
    """

    def __init__(self, color_index=-1, rank_index=-1, not_color_index=-1, not_rank_index=-1):
        """ color_index for the hints that concerns a color
            rank_index for the hints that concerns a rank
        """
        self.color_index = color_index
        self.rank_index = rank_index
        self.not_color_index = not_color_index
        self.not_rank_index = not_rank_index

    def get_type(self):
        """ Return 0 if the hint concerns the color, and 1 if the hint concerns the rank """
        if (self.color_index != -1):
            return 0
        elif (self.rank_index != -1):
            return 1

    def __str__(self):
        if (self.color_index >= 0):
            return "Color Index: " + str(self.color_index)
        elif (self.rank_index >= 0):
            return "Rank Index: " + str(self.rank_index)
        elif (self.not_color_index >= 0):
            return "Not Color Index: " + str(self.not_color_index)
        elif (self.not_rank_index >= 0):
            return "Not Rank Index: " + str(self.not_rank_index)
        else:
            return "Invalid Hint"

    def __repr__(self):
        return self.__str__()


#################################
##### AUXIALIARY FUNCTIONS ######
#################################

def compare_arrays(tab1, tab2):
    res = tab1 == tab2
    for bool in res:
        if (not(bool)):
            return False
    return True
