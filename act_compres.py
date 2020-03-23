
import time

# Wall              #
# Player            @
# Player on target  +
# Box               $
# Box on target     *
# Target            .
# Floor             (space)


mp = [[' ', ' ', ' ', ' ', '#', '#', '#', '#', '#'],
      [' ', ' ', '#', '#', '#', '@', ' ', ' ', '#'],
      [' ', ' ', '#', '.', ' ', '$', ' ', '.', '#'],
      [' ', ' ', '#', ' ', ' ', '$', ' ', '#', '#'],
      [' ', '#', '#', '#', ' ', '#', '#', '#', ' '],
      ['#', '#', ' ', '$', ' ', ' ', '#', ' ', ' '],
      ['#', ' ', ' ', ' ', ' ', ' ', '#', ' ', ' '],
      ['#', '.', ' ', ' ', '#', '#', '#', ' ', ' '],
      ['#', '#', '#', '#', '#', ' ', ' ', ' ', ' ']]
       

# This function locates the player on the map and returns a set of all the
# positions that are reachable for the player in the current state.
# (No boxes can be moved.)       
def find_all_pos(mp):
    # find player
    for i in range(len(mp)):
        for j in range(len(mp[i])):
            # '@+' stands for player
            if mp[i][j] in '@+':
                player_pos = (i, j)
                break
            
    reachable_space = {player_pos}
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    
    # search whole space for reachable positions
    n = 1
    while True:
        b = set()
        for pos in reachable_space:
            a = set()
            for move in moves:
                # '#$*' wall, box or box on target
                if mp[pos[0]+move[0]][pos[1]+move[1]] not in '#$*':
                    a.add((pos[0]+move[0], pos[1]+move[1]))
            b = a | b
        reachable_space = b | reachable_space
        if len(reachable_space) == n:
            break
        n = len(reachable_space)
   
    return reachable_space


# This function takes the map (current state) and the set of all reachable
# positions and returns the available box moves as a list of tuples with
# (box vertical position, box horizontal position, direction), direction = 'u','d','l','r'
def expand(mp, reachable_space):
    box_moves = []
    moves = [(1, 0, 'd'), (-1, 0, 'u'), (0, 1, 'r'), (0,-1, 'l')]
    for pos in reachable_space:
        for move in moves:
            if mp[pos[0]+move[0]][pos[1]+move[1]] in '$*' and mp[pos[0]+2*move[0]][pos[1]+2*move[1]] not in '#$*':
                box_moves.append((pos[0]+move[0], pos[1]+move[1], move[2]))
    return box_moves
                


s = time.time()

rs = find_all_pos(mp)
box_moves = expand(mp, rs)

e = time.time()

print('Time used: {}s'.format(e-s))









