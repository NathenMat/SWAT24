""" The code was translated from the original C++ code provided by Denis Rohde.
"""

import math
import numba
import numpy as np

from typing import Iterable
from python_frechet_decider import (
    cpp_decide as decide,
    decide_segment,
    _decide_segment,
    cpp_discrete_frechet_distance,
    cpp_hausdorff_distance,
    PRECISION,
)

import networkx as nx

import bisect

from simpl_types import Simplifiable


from copy import copy

# from Fred import (
#     Curve as CPPCurve,
#     frechet_approximate_minimum_error_simplification as cpp_approximate_minimum_error_simplification,
#     frechet_approximate_range_simplification as cpp_approximate_range_simplification,
#     frechet_approximate_minimum_link_simplification as cpp_approximate_minimum_link_simplification,
# )


from light_frechet import (
    # LightCurve,
    # LightCurves,
    calc_distance,
    less_than_with_filters,
    simplify_num as cpp_simplify_num,
    simplify_eps as cpp_simplify_eps,
    range_simplify_eps as cpp_range_simplify,
)

PYTHON = 2
ASSERTION_LEVEL = 0  # 0: no assertions, 1: fast assertions, 2: distrust this code assertions, 3: distrust also libary assertions
DEBUG_PLOTING = False


class Curve(Simplifiable):
    """Polygonal curve class with simplification methods"""

    #@profile
    def __init__(self, nparr):
        assert isinstance(nparr, np.ndarray)
        if isinstance(nparr, Curve):
            print("Curve")
        self.nparr = nparr
        # self._nparr = points.nparr if hasattr(points, "nparr") else None
        # self._cppc = points.cppc if hasattr(points, "cppc") else None
        self._spline = None

    @property
    def complexity(self) -> int:
        """Computes the complexity of the curve."""
        return self.nparr.shape[0]

    @property
    def dimensions(self) -> int:
        """Computes the dimensions of the curve."""
        return self.nparr.shape[1]

    @property
    def spline(self) -> "Curve":
        """Computes the spline of the curve."""
        if self._spline is None:
            self._spline = Curve([self.nparr[0], self.nparr[-1]])
        return self._spline

    # @property
    # def nparr(self) -> np.ndarray:
    #     """Returns the curve as a numpy array."""
    #     if self._nparr is None:
    #         self._nparr = np.array(self.points, dtype=np.float64)
    #     return self._nparr

    # @property
    # def cppc(self) -> CPPCurve:
    #     """Returns the curve as a C++ curve."""
    #     if self._cppc is None:
    #         self._cppc = CPPCurve(self.nparr)
    #     return self._cppc

    def __getitem__(self, index):
        return self.nparr[index]
        if isinstance(index, int):
            return self.points[index]
        elif isinstance(index, tuple):
            return Curve(self.points[index[0] : index[1]])
        elif isinstance(index, slice):
            return Curve(self.points[index])
        else:
            raise NotImplementedError

    @property
    def spine(self) -> "Curve":
        """Straight line between the first and last point of the curve."""
        return Curve([self[0], self[-1]])

    def distance(self, other, lb = None, ub = None) -> float:
        """Computes the distance between the curve and another curve."""
        if ASSERTION_LEVEL > 0:
            assert self.complexity > 0 and other.complexity > 0
        erg = distance(self, other, lb=lb, ub=ub)
        if ASSERTION_LEVEL > 2:
            # assert erg == distance(self, other, lb, ub)
            assert decide(self, other, erg + PRECISION), (
                decide(self, other, erg),
                erg,
            )
        return erg

    # @numba.njit
    def decide(self, other, eps) -> bool:
        """Decides whether the curve is close enough to another curve."""
        # if ASSERTION_LEVEL > 0:
        #     assert self.complexity > 0 and other.complexity > 0
        # if eps < 0:
        #     return False
        # if self.complexity == 2:
        #     decision = (
        #         decide_segment(self, other, eps)
        #         if other.complexity > self.complexity
        #         else decide_segment(other, self, eps)
        #     )
        # else:
        #     decision = decide(self, other, eps)
        decision = decide(self, other, eps)
        if ASSERTION_LEVEL > 2:
            assert decision == decide(other, self, eps)
            if decision:
                assert (distance(self, other) <= eps + PRECISION) == decision, (
                    distance(self, other),
                    eps,
                    decision,
                )
            else:
                assert (distance(self, other) <= eps - PRECISION) == decision, (
                    distance(self, other),
                    eps,
                    decision,
                )
        return decision

    #@profile
    def simplify_num(self, eps) -> "Curve":
        """Computes as curve of atmost Fréchet distance eps to the curve that minimizes the length."""
        simpl = minimum_link_simplification(self, eps)
        if ASSERTION_LEVEL > 2:
            assert self.decide(simpl, eps), (self.distance(simpl), eps)
        return simpl

    def simplify_num_fast(self, eps) -> "Curve":
        """Computes as curve of atmost Fréchet distance eps to the curve that minimizes the length."""
        simpl = approximate_minimum_link_simplification(self, eps)
        if ASSERTION_LEVEL > 2:
            assert self.decide(simpl, eps), (self.distance(simpl), eps)
        return simpl

    def simplify_dist(self, length) -> "Curve":
        """Computes as curve of atmost length that minimizes the Fréchet distance to the curve.
        Uses Imai and Iri algorithm.
        """
        if ASSERTION_LEVEL > 0:
            assert length >= 1
        simpl = minimum_error_simplification(self, length)
        if ASSERTION_LEVEL > 2:
            assert simpl.complexity <= length
        return simpl

    def simplify_dist_fast(self, length) -> "Curve":
        """Computes as curve of atmost length that minimizes the Fréchet distance to the curve.
        Uses Agarwal et al. algorithm.
        """
        if ASSERTION_LEVEL > 0:
            assert length >= 1
        simpl = approximate_minimum_error_simplification(self, length)
        if ASSERTION_LEVEL > 2:
            assert simpl.complexity <= length
        return simpl

    def range_simplification_old(
        self, base, ell, fast_simplification=False
    ) -> Iterable[Simplifiable]:
        """Integrates fast_simplification into the range_simplification method."""
        # yield from self.approximate_range_simplification(base, ell, fast_simplification=fast_simplification)
        # return
        l = math.ceil(base * ell)
        old = float("inf")
        while l >= 2:
            if l < old:
                if fast_simplification:
                    simpl, err = self.simplify_dist_fast(int(l))
                else:
                    simpl, err = self.simplify_dist(int(l))
                yield simpl, err
                old = simpl.complexity
            l = l // base

    def range_simplification(
        self, base, ell, fast_simplification=False
    ) -> Iterable[Simplifiable]:
        aimed_lengths = np.geomspace(2, ell, int(math.log(ell, base)) + 1, dtype=int)
        aimed_lengths = np.unique(aimed_lengths)
        aimed_lengths = aimed_lengths[::-1]
        ergs = cpp_range_simplify(self.nparr, aimed_lengths.tolist())
        for erg in ergs:
            yield Curve(erg.curve), erg.error

    def approximate_range_simplification(
        self, base, ell, fast_simplification=False
    ) -> Iterable[Simplifiable]:
        """Integrates fast_simplification into the range_simplification method."""
        aimed_lengths = np.geomspace(2, ell, int(math.log(ell, base)) + 1)
        aimed_lengths = np.unique(np.floor(aimed_lengths))
        yield from self._simul_simpl(aimed_lengths, fast_simplification=fast_simplification)


    def _simul_simpl(self, lengths, interval=None, fast_simplification=False) -> list[Simplifiable]:
        if len(lengths) == 0:
            return []
        if interval is None:
            # Exponential search
            low = PRECISION
            high = 2*low
            new_simplification = self.simplify_num_fast(high) if fast_simplification else self.simplify_num(high)
            lidx = bisect.bisect_right(lengths, new_simplification.complexity)
            left, right = lengths[:lidx], lengths[lidx:]
            yield from self._simul_simpl(right, [low, high], fast_simplification=fast_simplification)
            while len(left) > 0:
                low = high
                high *= 8
                new_simplification = self.simplify_num_fast(high) if fast_simplification else self.simplify_num(high)
                lidx = bisect.bisect_right(lengths, new_simplification.complexity)
                right, left = lengths[lidx:], lengths[:lidx]
                yield from self._simul_simpl(right, [low, high], fast_simplification=fast_simplification)
            return

        # Binary search
        mid_r = (interval[0] + interval[1]) / 2
        mid_simpl = self.simplify_num_fast(mid_r) if fast_simplification else self.simplify_num(mid_r)
        epsilon = max(interval[0] * PRECISION / 100, math.nextafter(0, 1))
        if interval[1] - interval[0] < epsilon:
            simpl = self.simplify_num_fast(interval[1]) if fast_simplification else self.simplify_num(interval[1])
            print(abs(simpl.complexity - lengths[0]))
            yield from (simpl for _ in range(len(lengths)))
        else:
            left = [interval[0], mid_r]
            lidx = bisect.bisect_right(lengths, mid_simpl.complexity)
            yield from self._simul_simpl(lengths[lidx:], left, fast_simplification=fast_simplification)
            right = [mid_r, interval[1]]
            yield from self._simul_simpl(lengths[:lidx], right, fast_simplification=fast_simplification)


    def _4_simplification_dist(self, length) -> float:
        """Computes a 4-approximation of the Fréchet distance of the curve to the curve that minimizes the length."""
        if length >= self.complexity:
            return 0
        z_new = [float("inf") for _ in range(self.complexity)]
        z_new[0] = 0
        for _ in range(0, length):
            z_old = copy(z_new)
            for i in range(1, self.complexity):
                z_new[i] = min(
                    [
                        (
                            self[j, i + 1].distance(self[j, i + 1].spine)
                            if self[j, i + 1].decide(self[j, i + 1].spine, z_old[j])
                            else z_old[j]
                        )
                        # max(z_old[j], self[j, i + 1].distance(self[j, i + 1].spine))
                        for j in range(0, i)
                    ]
                )
            del z_old
        return z_new[self.complexity - 1]

    def _base_simplification_dist(self, base, length) -> float:
        """Improves the 4-approximation to an base-approximation"""
        high = self._4_simplification_dist(length)
        low = high / 4
        while high - low > base - 1:
            mid = (high + low) / 2
            if self.simplify_num(mid).complexity > length:
                low = mid
            else:
                high = mid
        return high

    def __repr__(self) -> str:
        return str(self.nparr)

    def __iter__(self):
        return iter(self.nparr)

#@profile
# @numba.njit
def distance(curve1 : Curve, curve2 : Curve, lb = None, ub = None) -> float:
    return calc_distance(curve1.nparr, curve2.nparr)
    if curve1.complexity < 2 or curve2.complexity < 2:
        raise ValueError("Curves must have at least two points.")

    if curve1.dimensions != curve2.dimensions:
        raise ValueError("Curves must have the same number of dimensions.")

    # lb = _hausdorff_distance(curve1, curve2)
    lb = 0 if lb is None else lb # cpp_hausdorff_distance(curve1, curve2) # Too slow
    # ub_old = _discrete_frechet_distance(curve1, curve2)
    ub = cpp_discrete_frechet_distance(curve1, curve2) if ub is None else ub

    dist : float = _distance(curve1, curve2, ub, lb)

    return dist

# @numba.njit
def _hausdorff_distance(curve1, curve2):
    """Hausdorff distance as a lower bound"""
    curve1 = curve1.nparr
    curve2 = curve2.nparr
    distances2_sqr = np.empty(curve1.shape[0] + curve2.shape[0] + 2)

    offset = curve1 - curve2[:, np.newaxis]  # (n - 1, m, d)
    distances2_sqr[: curve1.shape[0]] = _min_squared_distance_to_line_segments(
        curve1, curve2, offset[:-1]
    )
    # min_squared_distance_to_line_segments(curve1, curve2, offset[:-1])
    distances2_sqr[curve1.shape[0] : -2] = _min_squared_distance_to_line_segments(
        curve2, curve1, -offset[:, :-1].transpose(1, 0, 2)
    )
    # min_squared_distance_to_line_segments(curve2, curve1, -offset[:, :-1].transpose(1, 0, 2))
    distances2_sqr[-2] = np.sum((curve1[0] - curve2[0]) * (curve1[0] - curve2[0]))
    distances2_sqr[-1] = np.sum((curve1[-1] - curve2[-1]) * (curve1[-1] - curve2[-1]))
    maximum = np.max(distances2_sqr)
    return np.sqrt(maximum)


def _line_segment_dist_sqr(point, start, end):
    line = end - start
    len_sqr = np.dot(line, line)

    if len_sqr == 0:
        return np.linalg.norm(point - start)

    t = np.clip(np.dot(point - start, end - start) / len_sqr, 0, 1)
    projection = start + t * line
    return np.linalg.norm(point - projection) ** 2

# @numba.njit
def min_squared_distance_to_line_segments(points, line_segments, offsets):
    """
    Compute the minimum squared distance from each point to the given line segments.

    Args:
        points (ndarray): Array of shape (m, d) representing m points in d-dimensional space.
        line_segments (ndarray): Array of shape (n, d) representing n line segment endpoints in d-dimensional space.
        offsets (ndarray): Array of shape (n - 1, m, d) representing offsets for each line segment.

    Returns:
        ndarray: Array of shape (m,) containing the minimum squared distance from each point to the line segments.
    """
    # Compute vectors representing line segments
    line = line_segments[1:] - line_segments[:-1]  # (n - 1, d)

    # Compute squared lengths of line segments
    len_sqr = np.sum(line * line, axis=-1)  # (n - 1, )

    # Find segments with zero length
    zero_len_sqr = len_sqr == 0  # (n - 1, )

    # Compute squared distances for segments with zero length
    result_zero_len = np.sum(offsets[zero_len_sqr] ** 2, axis=-1)  # (n - 1, m)

    # Compute dot products between offset vectors and line vectors for non-zero length segments
    zw = np.sum(
        offsets[~zero_len_sqr] * line[~zero_len_sqr, np.newaxis], axis=-1
    )  # (n - 1, m)

    # Compute parameter t for projection onto line segments
    t = np.clip(zw / len_sqr[~zero_len_sqr, np.newaxis], 0, 1)  # (n - 1, m)

    # Compute projections onto line segments
    projection = (
        line_segments[:-1, np.newaxis][~zero_len_sqr]
        + t[:, :, np.newaxis] * line[:, np.newaxis, :][~zero_len_sqr]
    )  # (n - 1, m, d)

    # Shape points array for broadcasting
    tiled_points = np.tile(
        points, (line_segments.complexity - 1, 1, 1)
    )  # (n - 1, m, d)

    # Compute squared distances for non-zero length segments
    result_non_zero_len = np.sum(
        (projection - tiled_points) ** 2, axis=-1
    )  # (n - 1, m)

    # Combine squared distances for zero and non-zero length segments
    result = np.empty(
        (line.complexity, points.complexity), dtype=np.float64
    )  # (n - 1, m)
    result[zero_len_sqr] = result_zero_len
    result[~zero_len_sqr] = result_non_zero_len

    # Find minimum squared distance for each point
    return np.min(result, axis=0)


# @numba.njit
def _min_squared_distance_to_line_segments(points, line_segments, offsets):
    """
    Compute the minimum squared distance from each point to the given line segments.

    Args:
        points (ndarray): Array of shape (m, d) representing m points in d-dimensional space.
        line_segments (ndarray): Array of shape (n, d) representing n line segment endpoints in d-dimensional space.
        offsets (ndarray): Array of shape (n - 1, m, d) representing offsets for each line segment.

    Returns:
        ndarray: Array of shape (m,) containing the minimum squared distance from each point to the line segments.
    """
    # Compute vectors representing line segments
    line = line_segments[1:] - line_segments[:-1]  # (n - 1, d)

    # Compute squared lengths of line segments
    len_sqr = np.sum(line * line, axis=-1)  # (n - 1, )

    # Compute dot products between offset vectors and line vectors
    zw = np.sum(offsets * line[:, np.newaxis, :], axis=-1)  # (n - 1, m)

    # Compute parameter t for projection onto line segments
    t = np.clip(zw / len_sqr[:, np.newaxis], 0, 1)  # (n - 1, m)

    # Compute projections onto line segments
    projection = (
        line_segments[:-1, np.newaxis, :] + t[:, :, np.newaxis] * line[:, np.newaxis, :]
    )  # (n - 1, m, d)

    # Shape points array for broadcasting
    tiled_points = np.tile(
        points, (line_segments.shape[0] - 1, 1, 1)
    )  # (n - 1, m, d)

    # Compute squared distances for each segment
    segment_distances = np.sum((projection - tiled_points) ** 2, axis=-1)  # (n - 1, m)

    # Find minimum squared distance for each point
    min_distances = np.min(segment_distances)  # (m,)

    return min_distances


# @numba.njit
def _discrete_frechet_distance(curve1 : np.ndarray, curve2 : np.ndarray) -> float:
    """
    Calculate the discrete Fréchet distance between two curves.
    """
    dist_mat = np.sum((curve1[:, np.newaxis] - curve2) ** 2, axis=-1)

    # Initialize the first row and column
    # dist_mat[:, 0] = np.maximum.accumulate(dist_mat[:, 0])
    # dist_mat[0, :] = np.maximum.accumulate(dist_mat[0, :])
    for i in range(1, curve1.complexity):
        dist_mat[i, 0] = max(dist_mat[i - 1, 0], dist_mat[i, 0])
    for j in range(1, curve2.complexity):
        dist_mat[0, j] = max(dist_mat[0, j - 1], dist_mat[0, j])

    # Compute memoization matrix using dynamic programming
    for i in range(1, curve1.complexity):
        for j in range(1, curve2.complexity):
            zw = np.min(np.array([dist_mat[i - 1, j], dist_mat[i - 1, j - 1], dist_mat[i, j - 1]]))
            dist_mat[i, j] = max(zw, dist_mat[i, j])
    return np.sqrt(dist_mat[-1, -1])


#@profile
# @numba.njit
def _distance(curve1 : Curve, curve2 : Curve, ub : float, lb : float) -> float:
    if curve1.complexity > curve2.complexity:
        curve1, curve2 = curve2, curve1
    if curve1.complexity == 2:
        decider = decide_segment
    else:
        decider = decide

    split = (ub + lb) / 2
    p_error = (
        lb * PRECISION / 100
        if lb * PRECISION / 100 > np.finfo(float).eps
        else np.finfo(float).eps
    )
    number_searches = 0

    if ub - lb > p_error:

        if math.isnan(lb) or math.isnan(ub):
            # result.value = np.nan
            return np.nan

        while ub - lb > p_error:
            number_searches += 1
            split = (ub + lb) / 2
            if split == lb or split == ub:
                break
            is_less_than = decider(curve1, curve2, split)
            if is_less_than:
                ub = split
            else:
                lb = split

    if not decider(curve1, curve2, ub + PRECISION):
        print("Error in _distance")
        ub_old = _distance(curve1, curve2, ub * 10, ub)
        print(ub, ub_old)
        return ub_old

    return ub

class Subcurve_Shortcut_Graph:
    def __init__(self, pcurve):
        self.pcurve = pcurve
        self.nparr = pcurve.nparr
        self._edges = np.full(
            (pcurve.complexity, pcurve.complexity), np.inf, dtype=np.float64
        )

    @property
    def edges(self, i, j):
        if i < j:
            return np.inf
        if self._edges[j][i] == np.inf:
            subcurve = self.pcurve[i : j + 1]
            spline = subcurve.spline
            self._edges[j][i] = spline.distance(subcurve)
        return self._edges[j][i]

    #@profile
    def minimum_error_simplification(self, ll):
        if ll >= self.nparr.complexity:
            return self.nparr
        l = ll - 1
        result = np.empty((0, self.nparr.dimensions))

        if ll <= 2:
            result = np.vstack((result, self.nparr[0]))
            result = np.vstack((result, self.nparr[-1]))
            return result

        distances = np.full((self.nparr.complexity, l), np.inf, dtype=np.float64)
        predecessors = np.zeros((self.nparr.complexity, l), dtype=int)

        for i in range(l):
            if i == 0:
                for j in range(1, self.nparr.complexity):
                    distances[j][0] = self.edges(0, j)
                    predecessors[j][0] = 0
            else:
                for j in range(1, self.nparr.complexity):
                    others = np.maximum(
                        distances[:j, i - 1],
                        np.array([self.edges(k, j) for k in range(j)]),
                    )  # self.edges(:j, j)
                    best = np.argmin(others)
                    distances[j][i] = others[best]
                    predecessors[j][i] = best

        ell = l - 1
        result = np.vstack((result, self.nparr[-1]))
        predecessor = predecessors[self.nparr.complexity - 1][ell]

        for i in range(l - 1):
            result = np.vstack((result, self.nparr[predecessor]))
            predecessor = predecessors[predecessor][ell]
            ell -= 1

        result = np.vstack((result, self.nparr[0]))
        result = np.flip(result, axis=0)
        return result

    #@profile
    def minimum_link_simplification(self, epsilon):
        #@profile
        def weight_func(*args):
            u, v, _ = args
            if v == u + 1:
                return True
            subcurve = self.pcurve[u : v + 1]
            spline = subcurve.spline
            decision = spline.decide(subcurve, epsilon)
            return decision if decision else np.inf

        def heuristic(*args):
            _, v = args
            return 1 - v / self.pcurve.complexity

        # edges = np.ones((self.pcurve.complexity, self.pcurve.complexity), dtype=np.bool_)
        # for i in range(self.pcurve.complexity):
        #     for j in range(i + 1, self.pcurve.complexity):
        #         edges[i][j] = weight_func(i, j, i)
        # edges = np.transpose(edges.nonzero())
        shortcut_graph_A = nx.DiGraph(
            [
                (i, j)
                for i in range(self.pcurve.complexity)
                for j in range(i + 1, self.pcurve.complexity)
            ]
        )
        # shortcut_graph_B = nx.DiGraph([(i, j) for i in range(self.pcurve.complexity) for j in range(i + 1, self.pcurve.complexity)])
        shortest_path = nx.astar_path(
            shortcut_graph_A,
            source=0,
            target=self.pcurve.complexity - 1,
            heuristic=heuristic,
            weight=weight_func,
        )
        # shortest_path_ref = nx.shortest_path(shortcut_graph_B, source=0, target=self.pcurve.complexity - 1, weight=weight_func)
        # assert shortest_path == shortest_path_ref
        simpl = Curve(self.pcurve.nparr[shortest_path])
        assert simpl.decide(self.pcurve, epsilon)
        return simpl


def minimum_link_simplification(curve, epsilon):
    return Subcurve_Shortcut_Graph(curve).minimum_link_simplification(epsilon)


def minimum_error_simplification(curve, ell):
    return Subcurve_Shortcut_Graph(curve).minimum_error_simplification(ell)

def approximate_minimum_link_simplification(curve, epsilon):
    erg = cpp_simplify_num(curve.nparr, epsilon)
    return Curve(erg.curve)
    erg = cpp_approximate_minimum_link_simplification(curve.cppc, epsilon)
    return Curve(erg.curve.values)
    return Curve(_approximate_minimum_link_simplification(curve.nparr, epsilon))

#@profile
# @numba.njit
def _approximate_minimum_link_simplification(curve : np.ndarray, epsilon : float) -> np.ndarray:
    complexity = curve.complexity
    i = 0
    simplification = np.empty((0, curve.nparr.dimensions))
    simplification = np.vstack((simplification, curve[0]))

    while i < complexity - 1:
        segment = np.zeros((2, curve.nparr.dimensions))
        segment[0] = curve[i]
        j = 0
        while True:
            j += 1
            if i + 2**j >= complexity:
                break
            segment[1] = curve[i + 2**j]
            subcurve = curve[i : i + 2**j + 1]
            # spline = Curve(segment)

            if not _decide_segment(segment, subcurve, epsilon):
                break

        low = 2 ** (j - 1)
        high = min(2**j, complexity - i - 1)

        while low < high:
            mid = low + (high - low + 1) // 2
            segment[1] = curve[i + mid]
            subcurve = curve[i : i + mid + 1]
            # spline = Curve(segment)

            if _decide_segment(segment, subcurve, epsilon):
                low = mid
            else:
                high = mid - 1

        i += low
        simplification = np.vstack((simplification, curve[i]))
        simpl = simplification
        # print(simpl.distance(curve[:i+1]), epsilon)
        # simpl.decide(curve[:i+1], epsilon + PRECISION)
        # assert simpl.decide(curve[:i+1], epsilon + PRECISION)
    simpl = simplification
    # if not simpl.decide(curve, epsilon + PRECISION):
    #     simpl.decide(curve, epsilon + PRECISION)
    #     print(simpl.distance(curve), epsilon)
    return simpl

def approximate_minimum_error_simplification(curve, ell : int):
    erg = cpp_simplify_eps(curve.nparr, ell)
    return Curve(erg.curve), erg.error
    # erg = cpp_approximate_minimum_error_simplification(curve.cppc, ell)
    # return Curve(erg.curve.values), erg.error
    # return Curve(_approximate_minimum_error_simplification(curve.nparr, ell))

#@profile
# @numba.njit
def _approximate_minimum_error_simplification(curve : np.ndarray, ell : int) -> np.ndarray:
    erg = cpp_simplify_eps(curve.nparr, ell)
    return Curve(erg.curve), erg.error
    # erg = cpp_approximate_minimum_error_simplification(curve.cppc, ell)
    # return Curve(erg.curve.values), erg.error

    assert ell > 0

    if ell >= curve.complexity:
        return curve

    if curve.complexity < 2:
        return curve

    simplification = curve[np.array([0, -1])]

    if ell == 2:
        return simplification

    if ell == 1:
        return simplification

    min_distance = 0
    max_distance : float = distance(curve, simplification) + 1
    mid_distance = 0

    new_simplification = _approximate_minimum_link_simplification(curve, max_distance)

    while new_simplification.complexity > ell:
        max_distance *= 2
        new_simplification = _approximate_minimum_link_simplification(
            curve, max_distance
        )

    epsilon = max(min_distance * PRECISION / 100, math.nextafter(0, 1))
    while max_distance - min_distance > epsilon:
        mid_distance = (min_distance + max_distance) / 2.0
        if mid_distance == max_distance or mid_distance == min_distance:
            break

        new_simplification = _approximate_minimum_link_simplification(
            curve, mid_distance
        )

        if new_simplification.complexity > ell:
            min_distance = mid_distance
        else:
            simplification = new_simplification
            max_distance = mid_distance

    return simplification
