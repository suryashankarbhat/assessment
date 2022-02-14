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
        self.movevalues = [0,1,2,3]
    def makeMove(self):        
        #Policy iteration
        try:
            myPosition = self.gameWorld.getTallonLocation()
            search = utils.pickRandomPose(myPosition.y,myPosition.x)
            meanies = self.gameWorld.getMeanieLocation()
            pits = self.gameWorld.getPitsLocation()
            Bonuses = self.gameWorld.getBonusLocation()
            #Shows updating policy as time goes on. The values in this need to be parsed to a function to create updating Optimum policy
            P,R = self.fill_in_probs()
            mdptoolbox.util.check(P,R)
            vi2 = mdptoolbox.mdp.ValueIteration(P,R,0.99)
            vi2.run()
            print('Policy:\n', vi2.policy)
           
            tallon_position = lambda x: np.ravel_multi_index(x, (10,10))
            tallon_at = int(tallon_position((myPosition.y,myPosition.x)))
            print("Tallon is at: ",tallon_at)

            if int(vi2.policy[tallon_at]) == self.movevalues[0]:
                print("At:",tallon_at," best move is going", vi2.policy[tallon_at],"north")
                return Directions.SOUTH## due to wrong addition in the world.py line 158 and 162, North actually moves Tallon down(SOUTH) on the graph and South moves Tallon up(NORTH) on the graph 
            if int(vi2.policy[tallon_at]) == self.movevalues[1]:
                print("At:",tallon_at," best move is going",vi2.policy[tallon_at],"south")
                return Directions.NORTH
            if int(vi2.policy[tallon_at]) == self.movevalues[2]:
                print("At:",tallon_at,"best move is going", vi2.policy[tallon_at],"east")
                return Directions.EAST
            if int(vi2.policy[tallon_at]) ==self.movevalues[3]:
                print("At:",tallon_at,"best move is going",vi2.policy[tallon_at],"west")
                return Directions.WEST
                #if there are no more bonuses, Tallon searches for them

        except Exception as e:
            print("Navigating blindly without policy, and searching for bonuses while avoiding meanies and pits")
            print("Tallon at :", myPosition.y, myPosition.x)
            #if there are only meanies nearby:
            if len(meanies)>0:
                for i in range(len(meanies)):
                    nextmeanie = meanies[i]
                    print("Meanie seen at ",nextmeanie.y, nextmeanie.x," , run!!!")
                    if nextmeanie.x < myPosition.x:    
                        return Directions.EAST#move away from the meanie
                    if nextmeanie.x > myPosition.x:
                        return Directions.WEST# move away from meanie
                    if nextmeanie.y < myPosition.y:    
                        return Directions.NORTH
                    if nextmeanie.y > myPosition.y:
                        return Directions.SOUTH
            #If there are only pits nearby            
            elif len(pits)>0:  
                for i in range(len(pits)):
                    nextpit = pits[i]
                    print("pit seen at ",nextpit.y,nextpit.x," , avoiding!!!")
                    if nextpit.x < myPosition.x:    
                        return Directions.EAST
                    if nextpit.x > myPosition.x:
                        return Directions.WEST
                    if nextpit.y < myPosition.y:    
                        return Directions.NORTH
                    if nextpit.y > myPosition.y:
                        return Directions.SOUTH
            #If there are only bonuses nearby            
            elif len (Bonuses)>0:
                for i in range(len(Bonuses)):
                    nextbonus = Bonuses[i]
                    print("Bonus seen at ",nextbonus.y,nextbonus.x," , acquiring!!!")
                    if nextbonus.x > myPosition.x:    
                        return Directions.EAST
                    if nextbonus.x < myPosition.x:
                        return Directions.WEST
                    if nextbonus.y > myPosition.y:    
                        return Directions.NORTH
                    if nextbonus.y < myPosition.y:
                        return Directions.SOUTH
            else:
                print("searching aimlessely", search.y,search.x)
                if search.x > myPosition.x:
                    return Directions.EAST
                if search.x < myPosition.x:
                    return Directions.WEST
                if search.y > myPosition.y:
                    return Directions.NORTH
                if search.y < myPosition.y:
                    return Directions.SOUTH    
            print(e)
        
#Baised of this example   
#https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem
    
    def fill_in_probs(self):
        grid_size = (config.worldLength, config.worldBreadth)
        white_cell_reward = -0.04
        
        allBonuses = self.gameWorld.getBonusLocation() if len(allBonuses)>=1 else (0,0)
        allmeanies = self.gameWorld.getMeanieLocation() if len(allmeanies)>=1 else (0,1)
        allpits = self.gameWorld.getPitsLocation()if len(allpits)>=1 else (0,2)
        grid_boundary = [(0,0),(0,grid_size[1]-1),(grid_size[0]-1,0),(grid_size[0]-1,grid_size[1]-1)]# These determine the four corners of the grid
        bonuses_reward=1.0
        negative_reward=-1.0
        action_North_South_East_West=(.025, .025, config.directionProbability, 0.)  #West,East,North,South     
          
        num_states = grid_size[0] * grid_size[1]
        num_actions = 4
        P = np.zeros((num_actions, num_states, num_states))
        R = np.zeros((num_states, num_actions))
        try:
            myPosition = self.gameWorld.getTallonLocation()
            tallon_position = lambda x: np.ravel_multi_index(x, (10,10))
            tallon_at = tallon_position((myPosition.y,myPosition.x))
            pit_positions = lambda x: np.ravel_multi_index(x,(10,10))
            meanie_positions = lambda x: np.ravel_multi_index(x,(10,10))
            bonus_positions = lambda x: np.ravel_multi_index(x,(10,10))
              
            bonuses_at=[]
            pits_at=[]
            meanies_at=[]
            #i created an array of the coordinates and used the min function to get the location closest to Tallon so it could go to that first
            #https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            for bonus in range(len(allBonuses)):
                bonuses_at.append( bonus_positions((allBonuses[bonus].y,allBonuses[bonus].x)))
                closestbonus = min(bonuses_at,key=lambda x:abs(x-tallon_at))
                
                #print("Bonueses unravelled is: ", bonuses_at)
                closestbonus = str(closestbonus)
                closestbonus = (int(closestbonus[0]),int(closestbonus[1])) if (len(closestbonus)>1) else (0,int(closestbonus))
                currentbonus = closestbonus
            print("Bonus closest to Tallon is: ",closestbonus)

            for meanie in range(len(allmeanies)):
                meanies_at.append(meanie_positions((allmeanies[meanie].y,allmeanies[meanie].x)))
                closestmeanie = min(meanies_at,key=lambda x:abs(x-tallon_at))
                closestmeanie = str(closestmeanie)
                closestmeanie = (int(closestmeanie[0]),int(closestmeanie[1])) if (len(closestmeanie)>1) else (0,int(closestmeanie))
                currentmeanie = closestmeanie
            print("Meanie closest to Tallon is: ",closestmeanie)
              
            for pit in range(len(allpits)):
                pits_at.append(pit_positions((allpits[pit].y,allpits[pit].x)))
                closestpit = min(pits_at,key=lambda x:abs(x-tallon_at))
                closestpit = str(closestpit)
                closestpit = (int(closestpit[0]),int(closestpit[1])) if(len(closestpit)>1) else (0,int(closestpit))
                currentpit = closestpit
            print("Pit closest to Tallon is: ",closestpit)
                
            #https://www.w3schools.com/python/python_lambda.asp
            #https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
            to_1d = lambda x: np.ravel_multi_index(x, grid_size)
            
            def hit_wall(cell):
                if cell in grid_boundary:
                    return True
                try: # ...good enough...
                    to_1d(cell)
                except ValueError as e:
                    return True
                return False
            #print("to_1d array", to_1d((5,5)))
            # converts the multidimentional grid to a 1 dimentional grid by converting each 2D cell location to its 1d equivalent 5 by 5 becomes position 55 for example
            # and then the indices of the cell are applied to the shape of the grid size (10,10)
            #########################################################################################
            # make probs for each action
            
            North = [action_North_South_East_West[i] for i in (0, 1, 2, 3)]#up
            South = [action_North_South_East_West[i] for i in (1, 0, 3, 2)]#down
            West = [action_North_South_East_West[i] for i in (2, 3, 1, 0)]#left
            East = [action_North_South_East_West[i] for i in (3, 2, 0, 1)]#right
            actions = [North, South, East, West]
            for i, a in enumerate(actions):
                actions[i] = {'North':a[2], 'South':a[3], 'West':a[0], 'East':a[1]}
                
            # work in terms of the 2d grid representation
        
            def update_P_and_R(cell, new_cell, a_index, a_prob):
                    
                if cell == currentbonus:
                    P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    R[to_1d(cell), a_index] = bonuses_reward
                    
                elif cell == currentmeanie:
                    P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    R[to_1d(cell), a_index] = negative_reward

                elif cell == currentpit:  # add prob to current cell
                    P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    R[to_1d(cell), a_index] = negative_reward
                
                elif hit_wall(new_cell):  # add prob to current cell
                    P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                    R[to_1d(cell), a_index] = white_cell_reward
                
                else:
                    P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                    R[to_1d(cell), a_index] = white_cell_reward

            for a_index, action in enumerate(actions):
                for cell in np.ndindex(grid_size):
                    # up
                    new_cell = (cell[0]-1, cell[1])#subtracting from row makes us go up to previous row
                    update_P_and_R(cell, new_cell, a_index, action['North'])

                    # down
                    new_cell = (cell[0]+1, cell[1])#adding to row makes us go down a row
                    update_P_and_R(cell, new_cell, a_index, action['South'])

                    # left
                    new_cell = (cell[0], cell[1]-1)
                    update_P_and_R(cell, new_cell, a_index, action['West'])
                    # right
                    new_cell = (cell[0], cell[1]+1)
                    update_P_and_R(cell, new_cell, a_index, action['East'])
            
            return P, R
        except Exception as e:
            print(e)
          

