
import util
import os, sys
import datetime, time
import argparse
import math
from itertools import combinations

class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None
        self.reachable_space = set()
        self.moves = ((1, 0, 'd'), (-1, 0, 'u'), (0, 1, 'r'), (0,-1, 'l'))
        self.box_moves = []
        
    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())
    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data
    def __lt__(self, other):
        return self.data < other.data
    def __hash__(self):
        return hash(self.data)
    # return player location
    def player(self):
        return self.data[0]
    # return boxes locations
    def boxes(self):
        return self.data[1:]
    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved
    def act(self, problem, act):
        if act in self.adj: return self.adj[act]
        else:
            val = problem.valid_move(self,act)
            self.adj[act] = val
            return val
        
    def deadp(self, problem):        
        
        
        #if box is not found in the simple dead end list then we need to check for frozen states
        #where boxes cannot be moved bue to their orientation        
        #check for each box if its in a frozen situation      
        
        #chooses between precomputed and dynamic computing of frozen state
        #return False
        boxes = self.boxes()
        mode = problem.algorithm
        
        if mode == 'nf' and problem.combs:
            
            boxes = (tuple(sorted(boxes)))
            
            if problem.frozenBoxes[tuple(boxes)]:
                return True
            else:
                self.dead = False
        

        else:
            for box in boxes:
                #this is a set to keep track of the visited boxes in each state
                marked = set()
                #if the box is a target we must not check for the frozed condition
                if not problem.map[box[0]][box[1]].target and problem.checkFrozen(box, marked, boxes):
                        #problem.print_state(self)
                        return True 
                else:
                    self.dead = False
        
        return self.dead

    
    #this tells the dynamic area that the boxes occupy
    def tellArea(self, problem):
        
        boxes = self.boxes()
        stateArea = {}
        
        for box in boxes:
            
            target_of_box = problem.targetsReachable[box]
            target_of_box = tuple(target_of_box)
            if target_of_box not in stateArea:
                stateArea[target_of_box] = 1
            else:
                stateArea[target_of_box] += 1
        
        return stateArea
    
    
    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache    
    
            
    # This function locates the player on the map and returns a set of all the
    # positions that are reachable for the player in the current state.
    # (No boxes can be moved.)       
    def find_all_pos(self, problem):
        
        # find player
        player_pos = self.player()
        
        # initialize frontier with player position
        frontier = {player_pos}
        
        # search whole space for reachable positions
        while True:
            f = set()
            for pos in frontier:
                for move in self.moves:
                    # if not wall and not box and not in reachable_space already
                    if (not problem.map[pos[0]+move[0]][pos[1]+move[1]].wall) and \
                     (((pos[0]+move[0], pos[1]+move[1])) not in self.boxes()) and \
                     ((pos[0]+move[0], pos[1]+move[1]) not in self.reachable_space):
                         f.add((pos[0]+move[0], pos[1]+move[1]))
            # update frontier for next iteration
            frontier = f
            # add new frontier positions to reachable_space
            self.reachable_space = frontier | self.reachable_space
            # break as soon as frontier is empty (no new position are found)
            if len(frontier) == 0:
                break
    
    #tells if the new move of the box is valif or not
    def tellValid(self, oldBox, problem, newBox, area):        
        
        target = problem.targetsReachable[oldBox]
        target = tuple(target)
        if target in area:
            area[target] -= 1
        
        target = problem.targetsReachable[newBox]
        target = tuple(target)
        if target in area:
            area[target] += 1
        else:
            area[target] = 1
        
        if area[target] > problem.maxArea[target]:
            return False
        else:
            return True    
    
    
    # This function takes the map (current state) and the set of all reachable
    # positions and returns the available box moves as a list of tuples with
    # (box vertical position, box horizontal position, direction), direction = 'u','d','l','r'
    def find_box_moves(self, problem):
        
        #area = self.tellArea(problem)
        
        for pos in self.reachable_space:
            
            for move in self.moves:
                # 1. line of if-statement: checks if a box borders on a reachable field
                # 2. and 3. line of if-statement: checks if the box can be moved (if there
                #                                 is a wall or box behind the box)
                if (pos[0]+move[0], pos[1]+move[1]) in self.boxes() and \
                 (not problem.map[pos[0]+2*move[0]][pos[1]+2*move[1]].wall) and \
                 (((pos[0]+2*move[0], pos[1]+2*move[1])) not in self.boxes() and \
                 ((pos[0]+2*move[0], pos[1]+2*move[1])) in problem.visitable):
                     # stores box location and available move for the box
                     
                     #if self.tellValid((pos[0]+move[0], pos[1]+move[1]), problem, (pos[0]+2*move[0], pos[1]+2*move[1]), area):
                     self.box_moves.append((pos[0]+move[0], pos[1]+move[1], move[2]))
    
    
    def all_adj_compressed(self, problem):
        self.find_all_pos(problem)
        self.find_box_moves(problem)
        
        moves_dict = {'d':(1, 0), 'u':(-1, 0), 'r':(0, 1), 'l':(0,-1)}
        
        if self.all_adj_cache is None:
            succ = []
            # expand all the states for all the moves in box_moves
            for move in self.box_moves:
                boxes = list(self.boxes())
                # find the current box in the list of boxes
                idx = boxes.index((move[0], move[1]))
                # update the position of the moved box
                boxes[idx] = (move[0]+moves_dict[move[2]][0], move[1]+moves_dict[move[2]][1])
                # next state
                nextState = SokobanState((move[0], move[1]), tuple(boxes))
                succ.append((move, nextState, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache
    

class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target

def parse_move(move):
    if move == 'u': return (-1,0)
    elif move == 'd': return (1,0)
    elif move == 'l': return (0,-1)
    elif move == 'r': return (0,1)
    raise Exception('Invalid move character.')

class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SokobanProblem(util.SearchProblem):
    # valid sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False, algorithm='nf'):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0,0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)        
        self.deadEnds = set()
        self.reached = set()
        self.algorithm = algorithm
        self.getLocks()
        self.pythoGrean = {}
        self.Manhattan = {}
        self.preHeuristics()
        
    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map)-1, len(self.map[-1])-1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row,col) in s.boxes()
                player = (row,col) == s.player()
                if box and target: print(DrawObj.BOX_ON, end='')
                elif player and target: print(DrawObj.PLAYER, end='')
                elif target: print(DrawObj.TARGET, end='')
                elif box: print(DrawObj.BOX_OFF, end='')
                elif player: print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall: print(DrawObj.WALL, end='')
                else: print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx,dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1,y1) in s.boxes():
            if self.map[x2][y2].floor and (x2,y2) not in s.boxes() and (x2, y2) in self.visitable:
                return True, True, SokobanState((x1,y1),
                    [b if b != (x1,y1) else (x2,y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1,y1), s.boxes())
        
    
    #this is a recursive function that tries to push a box to the goal 
    def travel(self, square, visited, checking, breath, height):
        
        #if the box is visited in the recursive call then it is reachable from the original box 
        checking.add(square)
        #if the square is already visited we must return to the previous stack
        if square in visited:
            return
        #if the coordinates are out of bound we must return
        if (square[0] < 0 or square[0] >= breath) or (square[1] < 0 or square[1] >= height):
            return
        
        #tells if we can have visited this point or not
        visit = False
        
        #vertical/horizontal axes of the box
        axes = ['ud', 'lr']
        #these are the valid moves
        validMoves = {}
        #for all the directions the block can reach
        for move in 'uldr':
            
            coordinate = parse_move(move)
            x = square[0] + coordinate[0]
            y = square[1] + coordinate[1]
            newDirection = (x, y)
            
            #if the direction is visited we will mark it visited
            if (x, y) in visited:
                validMoves[move] = ('visited', True, (x, y))
                visit = True
            #check if the box is a wall and within bounds
            elif (x >= 0 and x < breath) and (y >= 0 and y < height) and self.map[x][y].wall:
                validMoves[move] = ('wall', True, (x, y))
            #check if it is in the deadends
            elif (x, y) in self.deadEnds:
                validMoves[move] = ('lock', True, (x, y))
            else:
                validMoves[move] = ('floor', True, (x, y))
                
        #This dictionary will store more info about movement of the box
        isLock = {}
        isNotMovable = {}
        
        #we check if the axis is movable or not
        for axis in axes:
            
            #these conditions check if the box can be moved vertically/horizontally
            if validMoves[axis[0]][0] == 'wall' or validMoves[axis[1]][0] == 'wall':
                truthValue = validMoves[axis[0]][1] or validMoves[axis[1]][1]
            elif validMoves[axis[0]][0] == 'lock' and validMoves[axis[1]][0] == 'lock':
                truthValue = validMoves[axis[0]][1] and validMoves[axis[1]][1]
            else:
                truthValue = False
            
            isNotMovable[axis[0]] = truthValue
            isNotMovable[axis[1]] = truthValue
            isLock[axis] = truthValue
        #if the block is not movable then we just add it to the dead ends
        if isLock['ud'] and isLock['lr']:
            if not self.map[square[0]][square[1]].target and not visit:
                self.deadEnds.add(square)
                return
            
        #mark it visited from this point     
        if square not in self.deadEnds:
            visited.add(square)
            
        #for moves in all other directions    
        for move in 'uldr':
            
            #we will only be able to go to squares that can be reached from the current square
            #so we will only recurse to those blocks
            info = validMoves[move]
            sqType = info[0]
            boolean = info[1]
            
            #check if the block is movable to in the direction given
            if not isNotMovable[move]:
                self.travel(info[2], visited, checking, breath, height)
        
        return  
    
    #this function will actually check if i can reach from the box to the goal
    def getLocks(self):
        
        #tells if the box is pushable to the goal 
        isPushable = {}
        self.targetsReachable = {}
        self.maxArea = {}
        #Until the status of isPushable does not change
        while(True):
            temp = isPushable.copy()
            visited = set()
            checking = set()
            #check for each point in the sokoban map
            for row in range(len(self.map)):
                for col in range(len(self.map[row])):
                    box = (row, col)
                    #if the point is not a wall test if its reachable
                    if not self.map[row][col].wall:
                        #call the travel function to travel to the goal
                        self.travel(box, visited, checking, len(self.map), len(self.map[row]))
                        truthVal = False
                        #check if any goal is in the set
                        myTargets = []
                        for target in self.targets:
                            if target in checking:
                                myTargets.append(target)
                                truthVal = True
                        if len(myTargets) > 0:
                            self.targetsReachable[box] = myTargets
                            self.maxArea[tuple(myTargets)] = len(myTargets)
                        #mark the value of the box
                        isPushable[box] = truthVal
                    checking.clear()
                    visited.clear()
            #if it does not change
            if isPushable == temp:
                break
        
        #put all the pushable boxes as reachable boxes and add to the visitable set
        self.visitable = set()
        for points in isPushable:
            
            if isPushable[points]:
                self.visitable.add(points)
        
        self.combs =  ((math.factorial(len(self.visitable)) / (math.factorial(len(self.init_boxes)) * math.factorial(len(self.visitable) - len(self.init_boxes)))) * len(self.init_boxes))  < 20000000
        
        
        if self.algorithm == 'nf' and self.combs:
            self.frozenBoxes = {}
            comb = combinations(self.visitable, len(self.init_boxes))
            for boxes in comb:
                boxes = (tuple(sorted(boxes)))
                self.frozenBoxes[boxes] = False
                for box in boxes:
                    marked = set()
                    if self.checkFrozen(box, marked, boxes):
                        self.frozenBoxes[boxes] = True
                        break
    
    #this is a recursive function to check is there are frozen boxes in every state during game play 
    def checkFrozen(self, box, marked, boxes):
        
        
        #making an array for the 2 vertical axes or opposite sides
        oppositeSides = ['ud', 'lr']       
        
        #this is a dictionary to what lies ahead in all the four directions relative to the current position
        lockInfo = {}
        #checking for each of the reachable directions
        for move in 'uldr':
            #mapping stores the object that lies ahead
            mapping = {}
            #get the coordinat of the move
            cord = parse_move(move)
            #get the next coordinates
            x = cord[0] + box[0]
            y = cord[1] + box[1]
            
            #check if the next point is a wall/ dead lock / or box
            if (self.map[x][y].wall):
                mapping['wall'] = True
            elif((x, y) not in self.visitable):
                mapping['lock'] = True              
            elif((x, y) in boxes and (x, y) != (box[0], box[1])):
                # I am keeping a set of boxes that I have visited while the recursive call is 
                # going on, so that I am not stuck in a infinite cycle for checking boxes
                #treat it as a wall if such a case arises
                if (x, y) in marked:
                    #if such a case arises then treat is as a wall
                    mapping['wall'] = True
                else:
                    mapping['box'] = (x, y)
            #store this information for that particular move
            lockInfo[move] = mapping
        
        #this dictionary is for keeping check if a box can be moved in vertical/horizontal axis or not
        check = {}
        #check for the opposite sides of the box
        for opposite in oppositeSides:
            
             #get the moves in teh sides
             a = lockInfo[opposite[0]]
             b = lockInfo[opposite[1]]
             
             #check if the vertical/hgorizontal axis is blocked based on the following conditions
             if 'wall' in a.keys() or 'wall' in b.keys():
                 check[opposite] = True
             elif 'lock' in a.keys() and 'lock' in b.keys():
                 check[opposite] = True
             elif 'box' in a.keys() or 'box' in b.keys():
                 
                 #if there is a box we need to recursively check if the box adjacent is frozed or not
                 #this makes sure that we are only considering completely frozen boxes
                 if 'box' in a.keys() and 'box' not in b.keys():
                     marked.add(a['box'])
                     check[opposite] = self.checkFrozen(a['box'], marked, boxes)
                 elif 'box' in b.keys() and 'box' not in a.keys():
                     marked.add(b['box'])
                     check[opposite] = self.checkFrozen(b['box'], marked, boxes)
                 else:
                     marked.add(a['box'])                         
                     check[opposite] = self.checkFrozen(a['box'], marked, boxes)
                     marked.add(b['box'])
                     check[opposite] = self.checkFrozen(b['box'], marked, boxes)
             else:
                 check[opposite] = False
        
        #if the current block is blocked in both the vertical and the horizontal axis
        #recursively proven, hence this is a frozen state and we must return to the previous stack
        if (check['ud'] and check['lr']):
            #make sure its not a target
            if not (self.map[box[0]][box[1]].target):
                return True
        else:
            return False    
    
    def preHeuristics(self):
        
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                
                minDist = 2**31
                minDist2 = 2**31
                for target in self.targets:
                    newDist = (abs(row - target[0])) + (abs(col - target[1]))
                    newDist2 = math.sqrt((row - target[0])**2 + (col - target[1])**2)
                    if newDist < minDist:
                        minDist = newDist
                    if newDist2 < minDist2:
                        minDist2 = newDist2
                
                self.Manhattan[(row, col)] = minDist
                self.pythoGrean[(row, col)] = minDist2
    

    ##############################################################################
    # Problem 1: Dead end detection                                              #
    # Modify the function below. We are calling the deadp function for the state #
    # so the result can be cached in that state. Feel free to modify any part of #
    # the code or do something different from us.                                #
    # Our solution to this problem affects or adds approximately 50 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False
        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):        
        if self.dead_end(s):
            return []
        #self.print_state(s) 
        return s.all_adj(self)

class SokobanProblemFaster(SokobanProblem):
    ##############################################################################
    # Problem 2: Action compression                                              #
    # Redefine the expand function in the derived class so that it overrides the #
    # previous one. You may need to modify the solve_sokoban function as well to #
    # account for the change in the action sequence returned by the search       #
    # algorithm. Feel free to make any changes anywhere in the code.             #
    # Our solution to this problem affects or adds approximately 80 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    
    def expand(self, s):
        if self.dead_end(s):
            return []
        #self.print_state(s) 
        return s.all_adj_compressed(self)
	

class Heuristic:
    def __init__(self, problem):
        self.problem = problem
        self.targets = problem.targets
        self.Manhattan = problem.Manhattan
        self.pythoGrean = problem.pythoGrean

    ##############################################################################
    # Problem 3: Simple admissible heuristic                                     #
    # Implement a simple admissible heuristic function that can be computed      #
    # quickly based on Manhattan distance. Feel free to make any changes         #
    # anywhere in the code.                                                      #
    # Our solution to this problem affects or adds approximately 10 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def heuristic(self, s):       
        
       dist = 0
       for box in s.boxes():
           dist += self.Manhattan[box]
          
       return dist
	

    ##############################################################################
    # Problem 4: Better heuristic.                                               #
    # Implement a better and possibly more complicated heuristic that need not   #
    # always be admissible, but improves the search on more complicated Sokoban  #
    # levels most of the time. Feel free to make any changes anywhere in the     # # code. Our heuristic does some significant work at problem initialization   #
    # and caches it.                                                             #
    # Our solution to this problem affects or adds approximately 40 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def heuristic2(self, s):
        
        player = s.player()
        dist = 0
        for box in s.boxes():
            dist += self.pythoGrean[box]
            
        return dist

## solve sokoban map using specified algorithm
def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection, algorithm)
    else:
        #print(dead_detection)
        problem = SokobanProblem(map, dead_detection, 'nf')

    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    #print(search.actions)
    if search.actions is not None:
        print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
         #raise NotImplementedError('Override me')
#        return search.totalCost, search.actions, search.numStatesExplored
        return search.totalCost, convert_actions(search.actions, problem), search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored


def bfs(goal, problem, player_pos, cost, moves, possibleActions, state, visited):   
  
    #print(possibleActions)
    
    #print(goal)
    #print(player_pos)
    
    if player_pos in visited:
        #print('visited')
        return
      
#    if cost in possibleActions:
#        #print('cost in possibleActions')
#        return  
      
    if player_pos == goal:
        possibleActions[cost] = moves
        #print('goal reached')
        return
    
    for move in 'rdlu':
        (valid, box_moved, _) = problem.valid_move(state, move, player_pos)
        if valid and not box_moved:
            (dx, dy) = parse_move(move)
            (x, y) = (player_pos[0] + dx, player_pos[1] + dy)
            #print('moves: {}'.format(moves))
            visited.add(player_pos)
            bfs(goal, problem, (x,y), cost+1, moves+[move], possibleActions, state, visited)
  
  
def convert_actions(box_actions, problem):
    
    moves_dict = {'d': (1, 0), 'u': (-1, 0), 'r': (0, 1), 'l': (0, -1)}
    player_actions = []
    
    # start state
    ss = SokobanState(problem.init_player, problem.init_boxes)
    
    for box_action in box_actions:
        possibleActions = {}
        moves = []
        visited = set()
        #player actions for each action
        goal = (box_action[0]-moves_dict[box_action[2]][0], box_action[1]-moves_dict[box_action[2]][1]) 
        bfs(goal, problem, ss.player(), 0, moves, possibleActions, ss, visited)
        print(possibleActions)
        
#        exit()
        
        #take path with minimum cost in possibleActions
        ss = SokobanState((box_action[0], box_action[1]), update_boxes(box_action, ss))
        pa = possibleActions[min(possibleActions.keys())]
        player_actions = player_actions + pa + [box_action[2]]
        
    return player_actions
             
  
def update_boxes(box_move, s):
    moves_dict = {'d': (1, 0), 'u': (-1, 0), 'r': (0, 1), 'l': (0, -1)}
    boxes = list(s.boxes())
    idx = boxes.index((box_move[0], box_move[1]))
    boxes[idx] = (box_move[0]+moves_dict[box_move[2]][0], \
                  box_move[1]+moves_dict[box_move[2]][1])
    return tuple(boxes)

# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)

# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else: break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')

# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels

def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
        (toc - tic).seconds + (toc - tic).microseconds/1e6, algorithm, nstates))
    
    if type(sol[0]) != tuple:
        seq = ''.join(sol)
        print(len(seq), 'moves')
        print(' '.join(seq[i:i+5] for i in range(0, len(seq), 5)))
    else:
        seq = 'test'
        
    if simulate:
        animate_sokoban_solution(map, seq)

def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300)", type=int, default=1000)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if (algorithm == 'all' and level == 'all'):
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now(): solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(maxSeconds):
        try:
            util.TimeoutFunction(solve_now, maxSeconds)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % maxSeconds)

    if level == 'all':
        levels = extract_levels(file)
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    else:
        solve_with_timeout(maxSeconds)

if __name__ == '__main__':
    main()
