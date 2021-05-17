from abc import ABC, abstractmethod

import numpy as np


class HeuristicAbstractBaseClass(ABC):
	# Interface that all heuristic functions need to implement
	@staticmethod
	@abstractmethod
	def evaluate(current, end):
		pass


class HeuristicZero(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return 0


class HeuristicAdmissible(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		# Half the Diagonal Distance
		return np.max(np.abs(end - current)) / 2


class HeuristicDiagonal(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return np.max(np.abs(end - current))


class HeuristicEuclidean(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return np.sqrt(np.sum(np.square(end - current)))


class HeuristicManhattan(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return np.sum(np.abs(end - current))


class HeuristicInadmissible(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		# 5 times the Manhattan Distance
		return 5 * np.sum(np.abs(end - current))
