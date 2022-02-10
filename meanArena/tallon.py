# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

import world
import random
import utils
import config
import numpy as np
import mdptoolbox
from utils import Directions

class Tallon():
    grid_size = (config.worldBreadth, config.worldLength)
    white_cell_reward = -0.04   
    bonus_cell_reward= 1.0
    meanies_cell_reward= -1.0
    hole_cell_reward = -1.0
    other_sides = (1 - config.directionProbability)/2
    black_cells = (1,1)
    action_lrfb_prob=(other_sides, other_sides, config.directionProbability, 0.)

    def __init__(self, arena):
        
        # Make a copy of the world an attribute, so that Tallon can
        # query the state of the world
        self.gameWorld = arena

        # What moves are possible.
        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        
    def makeMove(self):
        self.bonuses = self.gameWorld.getBonusLocation()
        self.pits = self.gameWorld.getPitsLocation()
        self.meanies = self.gameWorld.getMeanieLocation()
        
        num_states = self.grid_size[0] * self.grid_size[1]
        num_actions = 4
        self.P = np.zeros((num_actions, num_states, num_states))
        self.R = np.zeros((num_states, num_actions))
        self.fill_in_probs()

    def fill_in_probs(self):
        # helpers
        to_2d = lambda x: np.unravel_index(x, self.grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, self.grid_size)
        # print(to_1d)

        def hit_wall(cell):
            # print(cell)
            # if cell in self.black_cells:
            #     return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False


        # make probs for each action
        a_up = [self.action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [self.action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [self.action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [self.action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        for i, a in enumerate(actions):
            # print('i', i)
            # print('a', a)
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}
        
        def update_P_and_R(cell, new_cell, a_index, a_prob):
            # print('cell', cell)
            bonus_location = (self.bonuses[0].x, self.bonuses[0].y)
            pit_location = (self.pits[0].x, self.pits[0].y)
            meanies_location = (self.meanies[0].x, self.meanies[0].y)

            if cell == bonus_location:
                # print('added to bonus')
                self.P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                self.R[to_1d(cell), a_index] = self.bonus_cell_reward

            elif cell == pit_location:
                # print('added to pit')
                self.P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                self.R[to_1d(cell), a_index] = self.hole_cell_reward

            elif cell == meanies_location:
                # print('added to meanies')
                self.P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                self.R[to_1d(cell), a_index] = self.meanies_cell_reward


            elif hit_wall(new_cell):  # add prob to current cell
                # print('new cell', new_cell)
                # self.P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                # self.R[to_1d(cell), a_index] = self.white_cell_reward
                self.P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                self.R[to_1d(cell), a_index] = self.white_cell_reward

            else:
                # print('added to none')
                self.P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                self.R[to_1d(cell), a_index] = self.white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(self.grid_size):
                # up
                new_cell = (cell[0]-1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['up'])

                # down
                new_cell = (cell[0]+1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['down'])

                # left
                new_cell = (cell[0], cell[1]-1)
                update_P_and_R(cell, new_cell, a_index, action['left'])

                # right
                new_cell = (cell[0], cell[1]+1)
                update_P_and_R(cell, new_cell, a_index, action['right'])
        

        # print('length of P', len(self.P))
        # print('length of R', len(self.R))
        print('P',self.P)
        print('R',self.R)

        mdptoolbox.util.check(self.P, self.R)
        vi2 = mdptoolbox.mdp.ValueIteration(self.P, self.R, 0.99)
        vi = mdptoolbox.mdp.PolicyIteration(self.P, self.R, 0.99)
        vi2.run()
        # print('Values:\n', vi2.V)
        print('Policy:\n', vi2.policy)
        print('Policy:\n', vi.policy)
        
        def movetallon(self):
         for m in self.moves:
            if m == vi2.policy:# to move to the required direction
             return m
        # This is the function you need to define

        # For now we have a placeholder, which always moves Tallon
        # directly towards any existing bonuses. It ignores Meanies
        # and pits.
        # 
        # Get the location of the Bonuses.
        # allBonuses = self.gameWorld.getBonusLocation()

        # # if there are still bonuses, move towards the next one.
        # if len(allBonuses) > 0:
        #     nextBonus = allBonuses[0]
        #     myPosition = self.gameWorld.getTallonLocation()
        #     # If not at the same x coordinate, reduce the difference
        #     if nextBonus.x > myPosition.x:
        #         return Directions.EAST
        #     if nextBonus.x < myPosition.x:
        #         return Directions.WEST
        #     # If not at the same y coordinate, reduce the difference
        #     if nextBonus.y < myPosition.y:
        #         return Directions.NORTH
        #     if nextBonus.y > myPosition.y:
        #         return Directions.SOUTH

        # if there are no more bonuses, Tallon doesn't move
