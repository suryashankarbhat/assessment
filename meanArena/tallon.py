# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22


import numpy as np
import utils
from utils import Directions
import mdptoolbox
import config
class Tallon():

    def __init__(self, arena):

        # Make a copy of the world an attribute, so that Tallon can
        # query the state of the world
        self.gameWorld = arena

        # What moves are possible.
        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]#CHECK UTILS FOR DIRECTION VALUES
        self.move = [0,1,2,3]
    def makeMove(self):
        # This is the function you need to define

        # For now we have a placeholder, which always moves Tallon
        # directly towards any existing bonuses. It ignores Meanies
        # and pits.
        # 
        # Get the location of the Bonuses.
        
        #Policy iteration
        try:
            myPosition = self.gameWorld.getTallonLocation()
            current_position = utils.pickRandomPose(myPosition.x,myPosition.y)#search
            #gives the optimum policiy each time the tallon moves by calling below function
            P,R = self.probability()
            mdptoolbox.util.check(P,R)
            vi2 = mdptoolbox.mdp.ValueIteration(P,R,0.99)
            vi2.run()
            print('Policy:\n', vi2.policy)
           #A lambda operator or lambda function is used for creating small, one-time, anonymous function objects in Python.
           #lamda converts mulltidimensional array and make it into 1D array
            tallon_position = lambda x: np.ravel_multi_index(x, (10,10))
            #the current position gives the current tallon position 
            current_tallon = int(tallon_position((myPosition.y,myPosition.x)))# tallon_at
            print("Tallon's position is  ",current_tallon)
          # the below code is used to move the tallon to the grid based on the optimum policy
            if int(vi2.policy[current_tallon]) == self.move[0]:
                print("tallon is going north")
                return Directions.SOUTH
            if int(vi2.policy[current_tallon]) == self.move[1]:
                print("tallon is going south")
                return Directions.NORTH
            if int(vi2.policy[current_tallon]) == self.move[2]:
                print("tallon is going east")
                return Directions.EAST
            if int(vi2.policy[current_tallon]) ==self.move[3]:
                print("tallon is going west")
                return Directions.WEST
                #during partially observable process the tallon cannot see all the bonuses in the grid then it goes to the below code and 
     #below code will try to move the tallon at random position and try find the position of bonus if the bonus is out of range          
        except Exception as e:
            print("Searching ......")
            if  current_position.x > myPosition.x:
                return Directions.EAST
            if  current_position.x < myPosition.x:
                return Directions.WEST
            if  current_position.y > myPosition.y:
                return Directions.NORTH
            if  current_position.y < myPosition.y:
                return Directions.SOUTH
            print(e)
        
# the below code is created using the reference mentioned below
#https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem
    
    def probability(self):
        grid_size = (config.worldBreadth, config.worldLength)
        totalBonuses = self.gameWorld.getBonusLocation()#allBonuses
        totalmeanies= self.gameWorld.getMeanieLocation()#allmeanies
        totalpits = self.gameWorld.getPitsLocation()#allpits
        grid_edges = [(0,0),(0,10),(10,0),(10,10)]#grid boundry
        other_cell_reward = -0.04#whitecellreward
        b_r= 1.0
        n_r= -1.0
        action_N_S_E_W=(.025, .025, config.directionProbability, 0.)  # actions to move in all 4 directions   
        #this give the total number of states and actions for mdp process
        num_states = grid_size[0] * grid_size[1]
        num_actions = 4
        Pr = np.zeros((num_actions, num_states, num_states))
        Re = np.zeros((num_states, num_actions))
        try:
            myPosition = self.gameWorld.getTallonLocation()
            tallon_position = lambda x: np.ravel_multi_index(x, (10,10))
            tallon_current = tallon_position((myPosition.y,myPosition.x))
            allpitsloc = lambda x: np.ravel_multi_index(x,(10,10))#pits position
            allmeaniesloc = lambda x: np.ravel_multi_index(x,(10,10)) #meanie_positions
            allbonusloc = lambda x: np.ravel_multi_index(x,(10,10))#bonus_positions
              
            bonuses_loc=[]
            pits_loc=[]
            meanies_loc=[]
            #i created an array of the coordinates and used the min function to get the location closest to Tallon so it could go to that first
            #https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            for b in range(len(totalBonuses)):
                bonuses_loc.append(  allbonusloc((totalBonuses[b].y,totalBonuses[b].x)))
                near_bonus = min(bonuses_loc,key=lambda x:abs(x-tallon_current))
                near_bonus = str(near_bonus)
                near_bonus = (int(near_bonus[0]),int(near_bonus[1])) if (len(near_bonus)>1) else (0,int(near_bonus))
                nearest_bonus = near_bonus#currentbonus

            for meanie in range(len(totalmeanies)):
                meanies_loc.append(allmeaniesloc((totalmeanies[meanie].y,totalmeanies[meanie].x)))
                near_meanie = min(meanies_loc,key=lambda x:abs(x-tallon_current))
                near_meanie = str(near_meanie)
                near_meanie = (int(near_meanie[0]),int(near_meanie[1])) if (len(near_meanie)>1) else (0,int(near_meanie))
                nearest_meanie = near_meanie#current meanie
              
            for pit in range(len(totalpits)):
                pits_loc.append( allpitsloc((totalpits[pit].y,totalpits[pit].x)))
                near_pit = min(pits_loc,key=lambda x:abs(x-pits_loc))
                near_pit = str(near_pit)
                near_pit = (int(near_pit[0]),int(near_pit[1])) if(len(near_pit)>1) else (0,int(near_pit))
                nearest_pit= near_pit
                
            #https://www.w3schools.com/python/python_lambda.asp
            #https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
            to_1d = lambda x: np.ravel_multi_index(x, grid_size)
            #to avoid the error if the tallon is at edge
            def hit_edges(cell):
                if cell in grid_edges:
                    return True
                try: 
                    to_1d(cell)
                except ValueError as e:
                    return True
                return False
             # creating probability table for each action
            #probability table for moving north
            North = [action_N_S_E_W[i] for i in (0, 1, 2, 3)]
            #probability table for moving south
            South = [action_N_S_E_W[i] for i in (1, 0, 3, 2)]
            #probability table for moving east
            West = [action_N_S_E_W[i] for i in (2, 3, 1, 0)]
             #probability table for moving west
            East = [action_N_S_E_W[i] for i in (3, 2, 0, 1)]
            actions = [North, South, East, West]
            for i, a in enumerate(actions):
                actions[i] = {'North':a[2], 'South':a[3], 'West':a[0], 'East':a[1]}
                
            # work in terms of the 2d grid representation for updating reward and probability values
        #below function will give the values for each cell depending on the cell
            def update_Probability_and_Reward(cell, new_cell, a_index, a_prob):
                    
                if cell == nearest_bonus:
                    Pr[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    Re[to_1d(cell), a_index] = b_r
                    
                elif cell == nearest_meanie:
                    Pr[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    Re[to_1d(cell), a_index] = n_r

                elif cell ==  nearest_pit:  # add prob to current cell
                    Pr[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    Re[to_1d(cell), a_index] = n_r
                
                elif  hit_edges(new_cell):  # add prob to current cell
                    Pr[a_index, to_1d(cell), to_1d(cell)] += a_prob
                    Re[to_1d(cell), a_index] =  other_cell_reward
                else:
                    Pr[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                    Re[to_1d(cell), a_index] =  other_cell_reward

            for a_index, action in enumerate(actions):
                for cell in np.ndindex(grid_size):
                    # the below code will allow the tallon to move up
                    new_cell = (cell[0]-1, cell[1])
                    update_Probability_and_Reward(cell, new_cell, a_index, action['North'])

                    # down
                    new_cell = (cell[0]+1, cell[1])#adding to row makes us go down a row
                    update_Probability_and_Reward(cell, new_cell, a_index, action['South'])

                    # left
                    new_cell = (cell[0], cell[1]-1)
                    update_Probability_and_Reward(cell, new_cell, a_index, action['West'])
                    # right
                    new_cell = (cell[0], cell[1]+1)
                    update_Probability_and_Reward(cell, new_cell, a_index, action['East'])
            
            return Pr, Re
        except Exception as e:
         print(e)
 # in this try and except is used to neglate the errors when there are no bonuses or when tallon cannot see the bonuses         

