import cv2
import itertools
import numpy as np
import os

from heuristics import HeuristicDiagonal, HeuristicManhattan
from pathfinder import PathFinder


def get_maze(img):
	h, w = img.shape
	maze = np.ones(img.shape)

	# Loop over the pixels of the image and check for obstacles
	for i in range(h):
		for j in range(w):
			if np.all(img[i][j] == 255):
				maze[i][j] = 0

	return maze


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
	out = cv2.resize(img, (2285, 900), interpolation=cv2.INTER_NEAREST)

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
	maze = cv2.imread('data/maze_lv3.png')
	color = 230

	mask = maze[:, :, 0] != color
	# mask = mask.all(axis=-1)
	maze2 = maze.copy()
	maze2[mask] = [0, 0, 0]
	maze2[~mask] = [255, 255, 255]
	maze3 = maze2[:, :, 0]

	grid = get_maze(maze3)
	start = np.array([150, 20])
	end = np.array([160, 430])

	moves_4 = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
	moves_diagonal = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
	moves_8 = np.append(moves_4, moves_diagonal, axis=0)

	variants = [(moves_4, HeuristicManhattan), (moves_8, HeuristicDiagonal)]

	codes = {
		'wall': 0,
		'path': 1,
		'start': 2,
		'end': 3,
		'best_path': 4,
		'explored': 5,
		'frontier': 6,
	}

	c = itertools.count()

	for moves, heuristic in variants:
		pathfinder = PathFinder(grid, start, end, moves, heuristic)
		path_length = pathfinder.solve()
		print(path_length)

		final_state, explored_count = pathfinder.get_final_state(codes)
		visualize(final_state, next(c), codes)


if __name__ == '__main__':
	main()
