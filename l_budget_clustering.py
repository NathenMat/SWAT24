"""
    Implementation of the main algorithms from the paper.
"""

from __future__ import annotations

from copy import copy
from bisect import bisect
from abc import ABC, abstractmethod
import math
from typing import Iterable, Set

import colorsys

import heapq as hq
import itertools as it
import matplotlib.pyplot as plt

import Fred as fred


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
    def simplify_dist(self, length) -> Simplifiable:
        """Simplifies the object with the given length."""

    @abstractmethod
    def simplify_num(self, eps) -> Simplifiable:
        """Simplifies the object with the given eps."""

    def range_simplification(self, base, ell) -> Iterable[Simplifiable]:
        """ Computes the range of simplifications for the object."""
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
    """ Toy example of a simplifiable class """

    def __init__(self, value):
        self.value = value

    @property
    def complexity(self) -> int:
        return len(str(self.value))

    def distance(self, other) -> float:
        return abs(self.value - other.value)

    def simplify_dist(self, length) -> Number:
        return Number(round(self.value, length))

    def simplify_num(self, eps) -> Number:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self.value)


class Curve(Simplifiable):
    """ Polygonal curve class with simplification methods """

    def __init__(self, points, curve=None):
        self.points = points
        if curve is not None:
            self.curve = curve
        else:
            self.curve = fred.Curve(points)

    @property
    def complexity(self) -> int:
        """Computes the complexity of the curve."""
        return len(self.points)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.points[index]
        elif isinstance(index, tuple):
            return Curve(self.points[index[0] : index[1]])
        elif isinstance(index, slice):
            return Curve(self.points[index])
        else:
            raise NotImplementedError

    @property
    def spine(self) -> Curve:
        """ Straight line between the first and last point of the curve. """
        return Curve([self[0], self[-1]])

    def distance(self, other) -> float:
        """Computes the distance between the curve and another curve."""
        assert self.complexity > 0 and other.complexity > 0
        return fred.continuous_frechet(self.curve, other.curve).value

    def decide(self, other, eps) -> bool:
        """Decides whether the curve is close enough to another curve."""
        assert self.complexity > 0 and other.complexity > 0
        return fred.decide_continuous_frechet(self.curve, other.curve, eps)

    def simplify_num(self, eps) -> Curve:
        """ Computes as curve of atmost Fréchet distance eps to the curve that minimizes the length. 
        """
        simpl = fred.frechet_approximate_minimum_link_simplification(self.curve, eps)
        return Curve(simpl.values, simpl)

    def simplify_dist(self, length) -> Curve:
        """ Computes as curve of atmost length that minimizes the Fréchet distance to the curve. 
            Uses Imai and Iri algorithm.
        """
        assert length >= 1
        simpl = fred.frechet_minimum_error_simplification(self.curve, length)
        return Curve(simpl.values, simpl)

    def simplify_dist_fast(self, length) -> Curve:
        """ Computes as curve of atmost length that minimizes the Fréchet distance to the curve. 
            Uses Agarwal et al. algorithm.
        """
        assert length >= 1
        simpl = fred.frechet_approximate_minimum_error_simplification(
            self.curve, length
        )
        return Curve(simpl.values, simpl)

    def range_simplification(self, base, ell, fast_simplification=False) -> Iterable[Simplifiable]:
        """ Integrates fast_simplification into the range_simplification method. """
        l = math.ceil(base * ell)
        old = float("inf")
        while l >= 1:
            if l < old:
                if fast_simplification:
                    simpl = self.simplify_dist_fast(int(l))
                else:
                    simpl = self.simplify_dist(int(l))
                yield simpl
                old = simpl.complexity
            l = l // base

    def _4_simplification_dist(self, length) -> float:
        """ Computes a 4-approximation of the Fréchet distance of the curve to the curve that minimizes the length."""
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
        """ Improves the 4-approximation to an base-approximation """
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
        return str(self.points)

    def __iter__(self):
        return iter(self.points)


class Cluster:
    """ Base class for a cluster with a center."""
    
    def __init__(self, center) -> None:
        self.curves = set()
        self.center = center

    def add(self, curve):
        """ Adds a curve to the cluster."""
        self.curves.add(curve)

    def expand(self, curves):
        """ Adds a set of curves to the cluster."""
        self.curves.union(curves)

    def merge(self, other):
        """ Merges the cluster with another cluster. """
        self.curves.union(other.curves)

    def __len__(self):
        return len(self.curves)

    @property
    def complexity(self) -> int:
        """ Computes the complexity of the cluster."""
        return self.center.complexity

    def is_in_cover(self, r, curve: Curve):
        """ Decides if the curve is in the cover of the cluster."""
        # if curve in self.curves:
        #     return True
        return self.center.decide(curve, r)


class Clustering:
    def __init__(self) -> None:
        self.clustering: Set[Cluster] = set()

    def add(self, cluster: Cluster):
        self.clustering.add(cluster)

    def remove(self, cluster: Cluster):
        self.clustering.remove(cluster)

    def merge(self, other):
        self.clustering.union(other.clustering)

    def is_in_cover(self, r, curve: Curve):
        return any([cluster.is_in_cover(r, curve) for cluster in self.clustering])

    def __len__(self):
        return len(self.clustering)

    @property
    def complexity(self) -> int:
        return sum([cluster.complexity for cluster in self.clustering])

    def __iter__(self):
        return iter(self.clustering)


class LBudgetClustering:

    def __init__(
        self, big_l, r=10e-8, fast_simplification=False, track_curves=True
    ) -> None:
        self.r = r
        self.upper_bound = None
        self.lower_bound = None
        self.clustering = Clustering()
        self.big_l = big_l
        self.eta = 3
        self.fast_simplification = fast_simplification
        self.track_curves = track_curves

    def _update_r(self, r):
        self.lower_bound = r
        self.r = r
        self.upper_bound = r * self.eta

    @property
    def complexity(self) -> int:
        return self.clustering.complexity

    def __len__(self):
        return len(self.clustering)

    def __iter__(self):
        return iter(self.clustering)

    def main(self, gen):
        set_of_curves = set(gen)
        while not self._decide(self.r, set_of_curves):
            self._update_r(self.eta * self.r)
        high = self.r
        low = high / self.eta
        while high - low > 10e-8:
            mid = (high + low) / 2
            if self._decide(mid, set_of_curves):
                high = mid
            else:
                low = mid
        self._update_r(high)
        self._decide(self.r, set_of_curves)
        # assert self.invariant()
        return self

    def _decide(self, r, set_of_curves: Set[Curve]):
        self.clustering = Clustering()
        heap = []
        for curve in set_of_curves:
            simpl = curve.simplify_num(r)
            heap.append((simpl.complexity, simpl, curve))
        hq.heapify(heap)
        uncovered_curves = copy(set_of_curves)
        while uncovered_curves:
            ell, center, curve = hq.heappop(heap)
            if curve not in uncovered_curves:
                continue
            if self.complexity + ell > self.big_l:
                return False
            covered_curves = {
                z for z in uncovered_curves if center.decide(z, self.eta * r)
            }
            cluster = Cluster(center)
            cluster.curves = cluster.curves.union(covered_curves)
            self.clustering.add(cluster)
            uncovered_curves -= covered_curves
        assert self.complexity <= self.big_l
        return True

    def plot_clusters(self, save=False, savename=""):
        plt.axes().set_aspect("equal")
        c = [colorsys.hsv_to_rgb(i / len(self.clustering), 1.0, .6) for i in range(len(self.clustering))]
        if self.track_curves:
            for color, cluster in zip(c, self.clustering):
                for curve in cluster.curves:
                    if curve.complexity > 1:
                        plt.plot(
                            [x for x, _ in curve],
                            [y for _, y in curve],
                            c=color,
                            alpha=0.3,
                        )
                    else:
                        plt.scatter(
                            [x for x, _ in curve],
                            [y for _, y in curve],
                            c=color,
                            alpha=0.3,
                        )

        for color, cluster in zip(c, self.clustering):
            if cluster.complexity > 1:
                plt.plot(
                    [x for x, _ in cluster.center],
                    [y for _, y in cluster.center],
                    c=color,
                    marker="x",
                )
            else:
                plt.scatter(
                    cluster.center[0],
                    cluster.center[1],
                    c=color,
                    marker="x",
                )
        title = savename.split("/")[-1]
        title = title.split(":")[0]
        plt.title(f"{title}")
        if save:
            plt.savefig(
                f"{savename}.jpg",
                dpi=300,
            )
        else:
            plt.show()
        plt.clf()
        plt.close()

    def compute_assignment(self, curves=None):
        if curves is None:
            curves = set()
            for cluster in self.clustering:
                curves = curves.union(cluster.curves)
        erg = max(
            [
                min([cluster.center.distance(curve) for cluster in self.clustering])
                for curve in curves
            ]
        )
        if self.upper_bound is not None:
            print(erg, self.upper_bound, self.upper_bound >= erg)
            assert self.upper_bound >= erg - 10e-3
        if self.lower_bound is not None:
            print(erg, self.lower_bound, self.lower_bound <= erg)
            assert self.lower_bound <= erg + 10e-3
        self.upper_bound = erg


class DynamicKLClustering(LBudgetClustering):
    def __init__(
        self, k, l, alpha=2, r=0, fast_simplification=False, track_curves=True
    ) -> None:
        super().__init__(
            None,
            r=r,
            fast_simplification=fast_simplification,
            track_curves=track_curves,
        )
        self.k = k
        self.l = l
        self.alpha = alpha
        self.eta = 2 * self.alpha**2 / (self.alpha - 1)
        self.number_of_curves = 0

    def main(self, gen):
        while self.number_of_curves < self.k:
            obj = next(gen)
            self.clustering.add(Cluster(obj))
            self.number_of_curves += 1
            # assert self.invariant()
            yield self
        for obj in gen:
            self.number_of_curves += 1
            if self.r == 0:
                curves = [cluster.center for cluster in self.clustering] + [obj]
                self._update_r(self._compute_r(curves))
            for cluster in self.clustering:
                center = cluster.center
                if obj.decide(center.center, self.r * self.eta):
                    cluster.add(obj)
                    break
            while len(self.clustering) > self.k:
                self._update_r(self.alpha * self.r)
                self._merge_clusters()
            # assert self.invariant()
            yield self

    def _compute_r(self, curves):
        return min(
            [curve.distance(other) for curve, other in it.combinations(curves, 2)]
        )

    def _merge_clusters(self):
        if len(self.clustering) <= 1:
            return
        clusters = copy(self.clustering)
        while clusters:
            cluster1 = clusters.pop()
            ball = [
                cluster
                for cluster in self.clustering
                if cluster1.center.decide(cluster.center, 2 * self.alpha * self.r)
            ]
            for cluster2 in ball:
                if cluster1 is not cluster2:
                    cluster1.merge(cluster2)
                    self.clustering.remove(cluster2)
                    clusters.remove(cluster2)


class RangeCluster:
    def __init__(
        self, obj: Simplifiable, base, big_l, track_curves=False, fast=False
    ) -> None:
        self.cluster_range = []
        self.track_curves = track_curves
        self.fast = fast
        if track_curves:
            self.curves = [obj]
        old_dist = 0
        for simpl in obj.range_simplification(base, big_l, fast=self.fast):
            self.cluster_range.append((simpl, old_dist, obj.distance(simpl)))
            old_dist = self.cluster_range[-1][2]
        assert len(self.cluster_range) > 0

    @property
    def center(self):
        return self.cluster_range[0][0]

    @property
    def w(self):
        return self.cluster_range[0][1]

    @property
    def r(self):
        return self.cluster_range[0][2]

    def shorten_center_list(self, radius):
        index = bisect([w for _, w, r in self.cluster_range], radius) - 1
        self.cluster_range = self.cluster_range[index:]
        assert len(self.cluster_range)

    def insert(self, obj):
        for center, _, r in self.cluster_range:
            if r_new := obj.distance(center) <= r:
                r = r_new
        if self.track_curves:
            self.curves.append(obj)

    def merge(self, other):
        center2, _, r2 = other.center, other.w, other.r
        for center1, _, r1 in self:
            tmp_r = center1.distance(center2) if center1.decide(center2, r1) else r1
            r1 = tmp_r + r2
            # r1 = max(r1, center1.distance(center2)) + r2

        if self.track_curves:
            self.curves.extend(other.curves)

    def __iter__(self):
        return iter(self.cluster_range)

    @property
    def complexity(self) -> int:
        # _max = max([center.complexity for center, _, _ in self])
        # _sum = sum([center.complexity for center, _, _ in self])
        # print(_max, _sum)
        return max([center.complexity for center, _, _ in self])

    def __repr__(self) -> str:
        return str(self.center)

    def invariant(self, upper_bound):
        if self.track_curves:
            try:
                assert all(
                    [self.center.decide(curve, upper_bound) for curve in self.curves]
                )
            except AssertionError:
                for curve in self.curves:
                    dist = self.center.distance(curve)
                    if dist > upper_bound:
                        print(curve)
                        print(self.center)
                        print(dist)
                        print(upper_bound)
        return True


class RangeClustering:
    def __init__(self) -> None:
        self.clustering: Set[Cluster] = set()

    def add(self, cluster: Cluster):
        self.clustering.add(cluster)

    def remove(self, cluster: Cluster):
        self.clustering.remove(cluster)

    def merge(self, other):
        self.clustering.union(other.clustering)

    def is_in_cover(self, r, curve: Curve):
        return any([cluster.is_in_cover(r, curve) for cluster in self.clustering])

    def __len__(self):
        return len(self.clustering)

    @property
    def complexity(self) -> int:
        return sum([cluster.complexity for cluster in self.clustering])

    def __iter__(self):
        return iter(self.clustering)


class DynamicLBudgetClustering(DynamicKLClustering):
    def __init__(
        self,
        big_l,
        eps,
        r_0,
        alpha=2.0,
        track_curves=False,
        initial_simplify=False,
        fast_simplification=False,
    ) -> None:
        super().__init__(
            None,
            None,
            max(alpha, 2),
            r=r_0,
            fast_simplification=fast_simplification,
            track_curves=track_curves,
        )

        self.base = 1 + eps
        self.big_l = big_l
        self.clustering = RangeClustering()
        self.initial_simplify = initial_simplify

    def main(self, gen):
        for obj in gen:
            self._insert(obj)
            self._make_clustering_valid()
            # assert self.invariant()
            yield self

    def _insert(self, obj: Simplifiable):
        self.number_of_curves += 1
        for cluster in self.clustering:
            if obj.decide(cluster.center, self.r * self.eta):
                cluster.insert(obj)
                return
        if self.initial_simplify:
            obj = obj.simplify_num(self.r / self.eta)
        cluster = RangeCluster(
            obj, self.base, self.big_l, self.track_curves, fast=self.fast_simplification
        )
        cluster.shorten_center_list(self.r)
        self.clustering.add(cluster)

    def _make_clustering_valid(self):
        while self.complexity > self.base * self.big_l:
            self._update_r(self.alpha * self.r)
            for cluster in self.clustering:
                cluster.shorten_center_list(self.r)
            self._merge_clusters()

    def _merge_clusters(self):
        if len(self.clustering) <= 1:
            return
        clusters = copy(self.clustering)
        while clusters:
            cluster1 = clusters.pop()
            ball = [
                cluster
                for cluster in self.clustering
                if cluster1.center.decide(cluster.center, 2 * self.alpha * self.r)
            ]
            for cluster2 in ball:
                if cluster1 is not cluster2:
                    cluster1.merge(cluster2)
                    self.clustering.remove(cluster2)
                    clusters.remove(cluster2)

    def invariant(self):
        assert all([cluster.invariant(self.upper_bound) for cluster in self.clustering])
        assert self.complexity <= self.base * self.big_l
        try:
            assert all(
                [
                    cluster_a.center.decide(cluster_b.center, 2 * self.alpha * self.r)
                    for cluster_a, cluster_b in it.combinations(self.clustering, 2)
                ]
            )
        except AssertionError:
            for cluster_a, cluster_b in it.combinations(self.clustering, 2):
                distance = cluster_a.center.distance(cluster_b.center)
                if distance <= 2 * self.alpha * self.r:
                    print(distance)
                    print(cluster_a.center)
                    print(cluster_b.center)
                    print(2 * self.alpha * self.r)
        return True


class ScalingDynamicLBudgetClustering:
    def __init__(
        self,
        big_l,
        eps,
        r_0,
        track_curves=False,
        initial_simplify=False,
        fast_simplification=False,
    ) -> None:
        self.parallel_session = []
        self.current_best = None
        alpha = max(1 / eps, 2)
        m = math.ceil(1 / eps * math.log(1 / eps))
        for i in range(1, m + 1):
            r_i = alpha ** ((i / m) - 1) * r_0
            self.parallel_session.append(
                DynamicLBudgetClustering(
                    big_l,
                    eps,
                    r_i,
                    alpha,
                    track_curves=track_curves,
                    initial_simplify=initial_simplify,
                    fast_simplification=fast_simplification,
                )
            )

    def main(self, gen):
        ergs = []
        for session in self.parallel_session:
            ergs.append(session.main(gen))
        for _ in gen:
            res = [next(erg) for erg in ergs]
            self.current_best = min(res, key=lambda x: x.r)
            # assert self.invariant()
            yield self.current_best

    def plot_clusters(self, c="r", save=False, savename=""):
        self.current_best.plot_clusters(c, save, savename=savename)

    def invariant(self):
        assert all([session.invariant() for session in self.parallel_session])
        assert self.current_best.invariant()
        return True
