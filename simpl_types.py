""" The code was translated from the original C++ code provided by Denis Rohde.
"""
from abc import ABC, abstractmethod
from typing import Iterable
# from copy import copy

import math
# import numpy as np
# from python_frechet_decider import (
#     cpp_decide as decide,
#     decide_segment,
#     cpp_discrete_frechet_distance,
#     cpp_hausdorff_distance,
#     PRECISION,
# )


PYTHON = 2
ASSERTION_LEVEL = 0  # 0: no assertions, 1: fast assertions, 2: distrust this code assertions, 3: distrust also libary assertions
DEBUG_PLOTING = False

class Simplifiable(ABC):
    """Interface for objects that can be simplified."""

    @abstractmethod
    def distance(self, other) -> float:
        """Computes the distance between the object and another object."""

    @property
    @abstractmethod
    def complexity(self) -> int:
        """Computes the complexity of the object."""

    @abstractmethod
    def simplify_dist(self, length) -> "Simplifiable":
        """Simplifies the object with the given length."""

    @abstractmethod
    def simplify_num(self, eps) -> "Simplifiable":
        """Simplifies the object with the given eps."""

    def range_simplification(self, base, ell) -> Iterable["Simplifiable"]:
        """Computes the range of simplifications for the object."""
        l = math.ceil(base * ell)
        old = float("inf")
        while l >= 1:
            if l < old:
                simpl = self.simplify_dist(int(l))
                yield simpl
                old = simpl.complexity
            l = l // base

    def decide(self, other, eps):
        """Decides if the object is within the given distance."""
        return self.distance(other) <= eps

    def __lt__(self, other):
        return self.complexity < other.complexity


class Number(Simplifiable):
    """Toy example of a simplifiable class"""

    def __init__(self, value):
        self.value = value

    @property
    def complexity(self) -> int:
        return len(str(self.value))

    def distance(self, other) -> float:
        return abs(self.value - other.value)

    def simplify_dist(self, length) -> "Number":
        return Number(round(self.value, length))

    def simplify_num(self, eps) -> "Number":
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self.value)
