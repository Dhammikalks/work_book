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
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    expand = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    action = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 1
    expand[init[0]][init[1]] = 0
    x = init[0]
    y = init[1]
    g = 0
    e = 0
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
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]

            if x == goal[0] and y == goal[1]:
                Found = True;
                print(next)
                for i in range(len(expand)):
                    print(expand[i])
                policy = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
                x = goal[0]
                y = goal[1]
                policy[goal[0]][goal[1]] = '*'
                while x != init[0] or y != init[1]:
                    x2 = x - delta[action[x][y]][0]
                    y2 = y - delta[action[x][y]][1]
                    policy[x2][y2] = delta_name[action[x][y]]
                    x = x2;
                    y = y2;

                for i in range(len(policy)):
                    print(policy[i])


            else:
                for  i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            e = e +1
                            open.append([g2,x2,y2])
                            closed[x2][y2] = 1
                            expand[x2][y2] = e
                            action[x2][y2] = i


search()
