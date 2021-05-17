import itertools
import os
import time

import cv2
import numpy as np

import heuristics
from pathfinder import PathFinder


def get_maze():
	# Read the given image
	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'data/Task_1_Low.png')
	img = cv2.imread(path)

	# Given colors (BGR format)
	start_color = np.array((113, 204, 45))
	end_color = np.array((60, 76, 231))
	obstacle_color = np.array((255, 255, 255))

	start = end = None
	maze = np.ones((100, 100))

	# Loop over the pixels of the image and compare with the given values
	for i in range(100):
		for j in range(100):
			if np.all(img[i][j] == obstacle_color):
				maze[i][j] = 0

			elif np.all(img[i][j] == start_color):
				start = np.array((i, j))

			elif np.all(img[i][j] == end_color):
				end = np.array((i, j))

	return maze, start, end


def visualize(final_state, i, codes):
	# Color values
	colors = {
		codes['wall']: (255, 255, 255),
		codes['path']: (0, 0, 0),
		codes['start']: (113, 204, 45),
		codes['end']: (60, 76, 231),
		codes['best_path']: (250, 60, 90),
		codes['explored']: (00, 120, 250),
		codes['frontier']: (255, 150, 50),
	}

	# Generate image by mapping final_state values to corresponding colors
	img = np.array(
		[[colors[x] for x in row] for row in final_state], dtype=np.uint8
	)
	# Scale up image
	out = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)

	# Save image
	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/output {i}.png')
	cv2.imwrite(path, out)


def save_data(data, filename):
	# Write data to file
	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/{filename}.txt')
	np.savetxt(path, data, delimiter=', ', fmt='%g')


def main():
	# Get maze and start/end data from the input image
	maze, start, end = get_maze()

	# List of valid moves for the 2 cases
	moves_4 = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
	moves_diagonal = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
	moves_8 = np.append(moves_4, moves_diagonal, axis=0)
	move_sets = (moves_4, moves_8)

	# List of heuristics
	heuristics_list = (
		heuristics.HeuristicZero,
		heuristics.HeuristicAdmissible,
		heuristics.HeuristicInadmissible,
		heuristics.HeuristicDiagonal,
		heuristics.HeuristicManhattan,
		heuristics.HeuristicEuclidean,
	)

	# Lists to hold the values of interest
	path_lengths = []
	explored_nodes = []
	runtimes = []

	# Codes for various elements in the final state,
	# used for convenient mapping of colors for the visualization step.
	codes = {
		'wall': 0,
		'path': 1,
		'start': 2,
		'end': 3,
		'best_path': 4,
		'explored': 5,
		'frontier': 6,
	}

	# Counter generates unique numbers for the figure indices
	c = itertools.count(1)

	for i, move_set in enumerate(move_sets):
		for j, heuristic in enumerate(heuristics_list):
			# Create a pathfinder object
			pathfinder = PathFinder(maze, start, end, move_set, heuristic)

			# Solve maze with timer
			t_start = time.perf_counter()
			path_length = pathfinder.solve()
			t_end = time.perf_counter()

			# Visualize the final state after exploration
			final_state, explored_count = pathfinder.get_final_state(codes)
			visualize(final_state, next(c), codes)

			# Store data
			path_lengths.append(path_length)
			explored_nodes.append(explored_count)
			runtimes.append(t_end - t_start)

	# Save data
	save_data(np.array(path_lengths).reshape(2, 6), "path_lengths")
	save_data(np.array(explored_nodes).reshape(2, 6), "explored_nodes")
	save_data(np.array(runtimes).reshape(2, 6), "runtimes")


main()
