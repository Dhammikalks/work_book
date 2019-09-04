#!/bin/python -f

#grid format
#0 <- navigatable space
#1 <- block space

grid = [[0, 0, 1, 0, 0, 0],
 	[0, 0, 1, 0, 0, 0],
	[0, 0, 0, 0, 1, 0],
	[0, 0, 1, 1, 1, 0],
	[0, 0, 0, 0, 1, 0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
print(goal)
delta = [[-1, 0], #go up
	 [ 0, -1],#go left
	 [ 1, 0], #go down
	 [ 0, 1]] #go riht

delta_name = ['^', '<', 'V', '>']

cost = 1

def search():
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid[1]))]
    closed[init[0]][init[1]] = 1
    x = init[0]
    y = init[1]
    g = 0
    open = [[g, x, y]]
    #print(open)
    Found = False #flag to set when goal is found
    resign = False #flag to set when canot expand
    while Found is False and resign is False:
        if(len(open) == 0):
            #print(open)
            resign = True
            print('fail')
        else:
            print(open)
            open.sort()
            open.reverse()
            next = open.pop()
            print(next)
            x = next[1]
            y = next[2]
            g = next[0]

            if x == goal[0] and y == goal[1]:
                Found = True;
                print(next)

            else:
                for  i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2,x2,y2])
                            closed[x2][y2] = 1


search()
