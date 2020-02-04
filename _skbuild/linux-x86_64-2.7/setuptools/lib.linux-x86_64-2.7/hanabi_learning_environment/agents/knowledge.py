"""Knowledge and card classes"""

import numpy as np

class Knowledge:
    """Knowledge is the class that manages the knowledge of the player on the game.
    In this first version, it uses a probability matrix to represent it's own hand"""

    def __init__(self, config):

        self.config = config

        # default values to avoid errors if the parameters are not given
        # Verify if they are correct some day
        self.players = config['players'] if 'players' in config else 5
        self.colors = config['colors'] if 'colors' in config else 5
        self.ranks = config['ranks'] if 'ranks' in config else 5
        self.handSize = config['hand_size'] if 'hand_size' in config else 5


        #self.myHand = [Card(self) in range(handSize)]
        # find this fucking repartition of the cards ! Fuck !
        self.unknownColors = []
        self.unknownRanks  = []

    def update(self, observation):
        pass

class Card:
    """Knowledge and estimations about a specific card"""

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.probaColor = np.array([1/knowledge.colors for i in range(knowledge.colors)])
        self.probaRank = np.array([1/knowledge.ranks for i in range(knowledge.ranks)])

        # print(self.probaColor)
        # print(self.getProbaMatrix())

    def getProbaMatrix(self):
        return np.outer( self.probaColor,self.probaRank)

    def getProbaCard(self, colorIndex, rankIndex):
        return self.probaColor[colorIndex]*self.probaRank[rankIndex]
