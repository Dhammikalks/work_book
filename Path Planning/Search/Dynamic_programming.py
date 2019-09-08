# ----------
# User Instructions:
#
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal.
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']
delta_name_inverse = ['V', '>', '^', '<']
def compute_value(grid,goal,cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    value = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
    action = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]

    x = goal[0]
    y = goal[1]
    v = 0
    open = [[0,x,y]]
    value[x][y] = 0;
    closed[x][y] = 1;
    while open:
        open.sort()
        open.reverse()
        next = open.pop()
        v = next[0]
        x = next[1]
        y = next[2]
        for i in range(len(delta)):
            x2 = x + delta[i][0]
            y2 = y + delta[i][1]
            if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                    val = v + cost
                    closed[x2][y2] = 1
                    open.append([val,x2,y2])
                    value[x2][y2] = val;
                    action[x2][y2] = delta_name_inverse[i]
    # make sure your function returns a grid of values as
    # demonstrated in the previous video.
    return value,action

value,action = compute_value(grid,goal,cost)

for i in range(len(value)):
    print(value[i])
for i in range(len(action)):
    print(action[i])
