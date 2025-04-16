adjacency = {
    0: [1, 2, 3],            # 3 connections
    1: [0, 2, 4, 5],         # 4 connections
    2: [0, 1, 3, 6],         # 4 connections
    3: [0, 2, 6],            # 3 connections
    4: [1, 5, 7],            # 3 connections
    5: [1, 4, 6, 8],         # 4 connections
    6: [2, 3, 5, 7, 9],      # 5 connections (central hub)
    7: [4, 6, 8],            # 3 connections
    8: [5, 7, 9],            # 3 connections
    9: [6, 8, 2]             # 3 connections
}

'''
Current map structure:

       0
     / | \
    1--2--3
   / \ |  \
  4---5----6
   \   \  /|\
    7---8  |
        \  |
         9-'

'''

# Simpler map with clearer strategic positions
simple_adjacency = {
    0: [1, 3],              # top left corner
    1: [0, 2, 4],          # top center
    2: [1, 5],             # top right corner
    3: [0, 4, 6],          # middle left
    4: [1, 3, 5, 7],       # middle center (small hub)
    5: [2, 4, 8],          # middle right
    6: [3, 7],             # bottom left corner
    7: [4, 6, 8, 9],       # bottom center
    8: [5, 7],             # bottom right corner
    9: [7]                 # bottom extension
}

'''
Simpler map structure:

    0---1---2
    |   |   |
    3---4---5
    |   |   |
    6---7---8
        |
        9

Key strategic features:
- Territory 4 is a small hub (4 connections)
- Territory 7 is important for controlling bottom (4 connections)
- Corner territories (0,2,6,8) are more defensible (2 connections)
- Territory 9 is a special extension (1 connection)
'''

# To switch between maps, just change which one is imported in other files
# from core.risk_map import adjacency  # for current complex map
# from core.risk_map import simple_adjacency as adjacency  # for simpler map


# old map

'''adjacency = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1, 6],
    4: [1, 7],
    5: [2, 8],
    6: [3, 9],
    7: [4, 9],
    8: [5, 9],
    9: [6, 7, 8]
}'''


'''

     0
    / \
   1   2
  / \   \
 3   4   5
 |   |   |
 6   7   8
   \ |  /
     9

'''