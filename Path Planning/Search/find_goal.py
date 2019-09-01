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
goal = [len(grid)-1, len(grid)-1]

delta = [[-1, 0], #go up
	 [ 0, -1],#go left
	 [ 1, 0], #go down
	 [ 0, 1]] #go riht

delta_name = ['^', '<', 'V', '>']

cost = 1

def search():
	closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid[1])]
	closed[init[0],init[1]] = 1
	x = init[0]
	y = init[1]
	g = 0
	open = [[g, x, y]]
	found = False #flag to set when goal is found 
	resign = False #flag to set when canot expand
