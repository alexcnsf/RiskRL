adjacency = {
    0: [1, 2, 3],            # 3 connections
    1: [0, 2, 4, 5],         # 4 connections
    2: [0, 1, 3, 6],         # 4 connections
    3: [0, 2, 6],            # 3 connections
    4: [1, 5, 7],            # 3 connections
    5: [1, 4, 6, 8],         # 4 connections
    6: [2, 3, 5, 7, 9],      # ⭐ 5 connections (central hub)
    7: [4, 6, 8],            # 3 connections
    8: [5, 7, 9],            # 3 connections
    9: [6, 8, 2]             # 3 connections
}

'''
       0
     / | \
    1--2--3
   / \ |  \
  4---5----6
   \   \  /
    7---8
        |
        9

'''


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