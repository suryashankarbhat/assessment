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
        
        #since the data types are diffrent in the above declaration
        
        self.move = [0,1,2,3]
    def makeMove(self):
        # This is the function you need to define

        # For now we have a placeholder, which always moves Tallon
        # directly towards any existing bonuses. It ignores Meanies
        # and pits.
        # 
        # Get the location of the Bonuses.
        #the policy iteration is done using below code
        try:
            myPosition = self.gameWorld.getTallonLocation()
            current_position = utils.pickRandomPose(myPosition.x,myPosition.y)
           
            P,R = self.probability()
           
            
            mdptoolbox.util.check(P,R)
 
            vi2 = mdptoolbox.mdp.ValueIteration(P,R,0.99)
            vi2.run()
            
           
            print('Policy:\n', vi2.policy)
            #the below code is run to convert the co -ordinates of the tallon to 1d array
            tallon_position = lambda x: np.ravel_multi_index(x, (10,10))
            # this gives the current position of the tallon
            current_tallon = int(tallon_position((myPosition.y,myPosition.x)))
            print("Tallon's position: ",current_tallon)
            # the below code is used for the agent that is tallon to take action based the optimum policy
            if int(vi2.policy[current_tallon]) == self.move[0]:
                print("tallon's best move is going north")
                return Directions.SOUTH
            if int(vi2.policy[current_tallon]) == self.move[1]:
                print("tallon's best move is going south")
                return Directions.NORTH
            if int(vi2.policy[current_tallon]) == self.move[2]:
                print("tallon's best move is going east")
                return Directions.EAST
            if int(vi2.policy[current_tallon]) ==self.move[3]:
                print("tallon's best move is going west")
                return Directions.WEST
                #if there are no more bonuses, Tallon searches for them

        except Exception as e:
            print("Searching...")
            if current_position.x > myPosition.x:
                return Directions.EAST
            if current_position.x < myPosition.x:
                return Directions.WEST
            # If not at the same y coordinate, reduce the difference
            if current_position.y > myPosition.y:
                return Directions.NORTH
            if current_position.y < myPosition.y:
                return Directions.SOUTH
            print(e)
        
 #the below code is written based on the below link which explains how the probability and reward array can be obtained for a grid 
        #https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem
    
    def probability(self):
        grid_size = (config.worldBreadth, config.worldLength)
       
        
        totalBonuses = self.gameWorld.getBonusLocation()#allBonuses
        totalmeanies= self.gameWorld.getMeanieLocation()#allmeanies
        totalpits = self.gameWorld.getPitsLocation()#allpits
        grid_edges = [(0,0),(0,10),(10,0),(10,10)]#grid boundry
        bonuses_reward= 1.0
        meanie_reward= -1.0
        pit_reward= -1.0
        empty_cell_reward = -0.04
        action_N_S_E_W=(.025, .025, config.directionProbability, 0.)  #West,East,North,South     
          
        num_states = grid_size[0] * grid_size[1]
        num_actions = 4
        Pr = np.zeros((num_actions, num_states, num_states))
        Re = np.zeros((num_states, num_actions))
        try:
            myPosition = self.gameWorld.getTallonLocation()
            tallon_position = lambda x: np.ravel_multi_index(x, (10,10))
            tallon_current = tallon_position((myPosition.y,myPosition.x))
            allpitsloc = lambda x: np.ravel_multi_index(x,(10,10))
            allmeaniesloc = lambda x: np.ravel_multi_index(x,(10,10))
            allbonusloc = lambda x: np.ravel_multi_index(x,(10,10))
            bonuses_loc=[]
            pits_loc=[]
            meanies_loc=[] 
           #this code  creates an array of the coordinates and used the min function to get the location closest to Tallon so it could go to that first
           
            for b in range(len(totalBonuses)):
                bonuses_loc.append(allbonusloc((totalBonuses[b].y,totalBonuses[b].x)))
                near_bonus = min( bonuses_loc,key=lambda x:abs(x- tallon_current))
                near_bonus = str(near_bonus)
                near_bonus = (int(near_bonus[0]),int(near_bonus[1])) if (len(near_bonus)>1) else (0,int(near_bonus))
                nearestbonus = near_bonus
                
            for m in range(len(totalmeanies)):
                meanies_loc.append( allmeaniesloc((totalmeanies[m].y,totalmeanies[m].x)))
                near_meanie = min( meanies_loc,key=lambda x:abs(x- tallon_current))
                near_meanie = str(near_meanie)
                near_meanie = (int(near_meanie[0]),int(near_meanie[1])) if (len(near_meanie)>1) else (0,int(near_meanie))
                nearestmeanie = near_meanie
              
            for p in range(len(totalpits)):
                pits_loc.append(allpitsloc((totalpits[p].y,totalpits[p].x)))
                near_pit = min(pits_loc,key=lambda x:abs(x- tallon_current))
                near_pit = str(near_pit)
                near_pit = (int(near_pit[0]),int(near_pit[1])) if(len(near_pit)>1) else (0,int(near_pit))
                nearestpit = near_pit
                
            #this function used to convert 2D array into 1D array as mentioned earlier
            to_1d = lambda x: np.ravel_multi_index(x, grid_size)
            
            def hit_boundries(cell):
                if cell in grid_edges:
                    return True
                try: 
                    to_1d(cell)
                except ValueError as e:
                    return True
                return False
           
            
            North = [action_N_S_E_W[i] for i in (0, 1, 2, 3)]
            #south
            South = [action_N_S_E_W[i] for i in (1, 0, 3, 2)]
            #west
            West = [action_N_S_E_W[i] for i in (2, 3, 1, 0)]
            #east
            East = [action_N_S_E_W[i] for i in (3, 2, 0, 1)]
            actions = [North, South, East, West]
            for i, a in enumerate(actions):
                actions[i] = {'North':a[2], 'South':a[3], 'West':a[0], 'East':a[1]}
                
            
        #this
            def update_Prob_and_Reward(cell, new_cell, a_index, a_prob):
                    
                if cell == nearestbonus:
                    Pr[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    Re[to_1d(cell), a_index] = bonuses_reward
                    
                elif cell == nearestmeanie:
                    Pr[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    Re[to_1d(cell), a_index] =meanie_reward

                elif cell == nearestpit:  # add prob to current cell
                    Pr[a_index, to_1d(cell), to_1d(cell)] = 1.0
                    Re[to_1d(cell), a_index] = pit_reward
                
                elif hit_boundries(new_cell):  # add prob to current cell
                    Pr[a_index, to_1d(cell), to_1d(cell)] += a_prob
                    Re[to_1d(cell), a_index] = empty_cell_reward
                
                else:
                    Pr[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                    Re[to_1d(cell), a_index] = empty_cell_reward

            for a_index, action in enumerate(actions):
                for cell in np.ndindex(grid_size):
                  # the below code will allow the tallon to move up
                    new_cell = (cell[0]-1, cell[1])
                    update_Prob_and_Reward(cell, new_cell, a_index, action['North'])

                    # the below code will allow the tallon to move down
                    new_cell = (cell[0]+1, cell[1])#adding to row makes us go down a row
                    update_Prob_and_Reward(cell, new_cell, a_index, action['South'])

                    #the below code will allow the tallon left
                    new_cell = (cell[0], cell[1]-1)
                    update_Prob_and_Reward(cell, new_cell, a_index, action['West'])
                    #the below code will allow the tallon right
                    new_cell = (cell[0], cell[1]+1)
                    update_Prob_and_Reward(cell, new_cell, a_index, action['East'])
            
            return Pr, Re
        except Exception as e:
            print(e)