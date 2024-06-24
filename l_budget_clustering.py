"""
    Implementation of the main algorithms from the paper.
"""

from __future__ import annotations

from copy import copy, deepcopy
from bisect import bisect
import math
from typing import Iterable, Set
import sympy as sp

import colorsys

import heapq as hq
import itertools as it
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPolygon
import numpy as np

from line_profiler import profile

from python_frechet import (
    distance as py_dist,
    decide as py_decide,
    approximate_minimum_link_simplification as py_min_num_smpl_aprx,
    minimum_error_simplification as py_min_dist_smpl,
    approximate_minimum_error_simplification as py_min_dist_smpl_aprx,
    Curve,
    Simplifiable,
    PRECISION,
)

LIGHT = False


# if LIGHT:
#     from light_frechet import (
#         LightCurve as cpp_Curve,
#         LightCurves as cpp_Curves,
#         calc_distance as cpp_dist,
#         less_than_with_filters as cpp_decide,
#         simplify_num as cpp_min_num_smpl_aprx,
#         simplify_eps as cpp_min_dist_smpl,
#         simplify_eps as cpp_min_dist_smpl_aprx,
#         # discrete_klcenter as cpp_klcenter,
#     )

# else:
#     from Fred import (
#         Curve as cpp_Curve,
#         Curves as cpp_Curves,
#         continuous_frechet as cpp_dist,
#         decide_continuous_frechet as cpp_decide,
#         frechet_approximate_minimum_link_simplification as cpp_min_num_smpl_aprx,
#         frechet_approximate_minimum_error_simplification as cpp_min_dist_smpl,
#         frechet_approximate_minimum_error_simplification as cpp_min_dist_smpl_aprx,
#         discrete_klcenter as cpp_klcenter,
#     )

from Fred import Curve as CPPCurve

INIT_R = PRECISION
PYTHON = 2
ASSERTION_LEVEL = 0  # 0: no assertions, 1: fast assertions, 2: distrust this code assertions, 3: distrust also libary assertions
DEBUG_PLOTING = False

# if PYTHON <= 0:
#     np_Curve = cpp_Curve
#     np_Curves = cpp_Curves
#     decide = cpp_decide
#     dist = cpp_dist
#     min_num_smpl_aprx = cpp_min_num_smpl_aprx
#     min_dist_smpl = cpp_min_dist_smpl
#     min_dist_smpl_aprx = cpp_min_dist_smpl_aprx
# elif PYTHON <= 1:
#     np_Curve = py_Curve
#     np_Curves = py_Curves
#     decide = cpp_decide
#     dist = py_dist
#     min_num_smpl_aprx = py_min_num_smpl_aprx
#     min_dist_smpl = py_min_dist_smpl
#     min_dist_smpl_aprx = py_min_dist_smpl_aprx
# else:
decide = py_decide
dist = py_dist
min_num_smpl_aprx = py_min_num_smpl_aprx
min_dist_smpl = py_min_dist_smpl
min_dist_smpl_aprx = py_min_dist_smpl_aprx


class Cluster:
    """Base class for a cluster with a center."""

    def __init__(self, center: Curve, radius=None) -> None:
        self.curves: Set[Simplifiable] = set()
        self.center = center
        self.radius = radius
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def add(self, curve):
        """Adds a curve to the cluster."""
        self.curves.add(curve)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def expand(self, curves):
        """Adds a set of curves to the cluster."""
        self.curves = self.curves.union(curves)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def merge(self, other):
        """Merges the cluster with another cluster."""
        self.curves = self.curves.union(other.curves)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def __len__(self):
        return len(self.curves)

    @property
    def complexity(self) -> int:
        """Computes the complexity of the cluster."""
        return self.center.complexity

    def is_in_cover(self, r, curve: Curve):
        """Decides if the curve is in the cover of the cluster."""
        # if curve in self.curves:
        #     return True
        return self.center.decide(curve, r)

    def shorten_center_list(self, radius):
        """ Dummy function for compatibility with RangeCluster"""
        pass

    def invariant(self):
        assert all(
            [self.is_in_cover(self.radius + PRECISION, curve) for curve in self.curves]
        ), (self.radius, [self.center.distance(curve) for curve in self.curves])
        return True


class Clustering:
    def __init__(self, radius=None) -> None:
        self.set_of_clusters: Set[Cluster] = set()
        self.radius = radius
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def add(self, cluster: Cluster):
        self.set_of_clusters.add(cluster)
        self.radius = max(self.radius, cluster.radius)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def remove(self, cluster: Cluster):
        self.set_of_clusters.remove(cluster)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def merge(self, other):
        self.set_of_clusters = self.set_of_clusters.union(other.clustering)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    def is_in_cover(self, r, curve: Curve):
        return any([cluster.is_in_cover(r, curve) for cluster in self.set_of_clusters])

    def __len__(self):
        return len(self.set_of_clusters)

    @property
    def complexity(self) -> int:
        return sum([cluster.complexity for cluster in self.set_of_clusters])

    def plot_clusters(self, save=False, savename="", hippodrome=True):
        c = [
            colorsys.hsv_to_rgb(i / len(self.set_of_clusters), 1.0, 0.6)
            for i in range(len(self.set_of_clusters))
        ]
        self.plot_curves(c)

        self.plot_centers(hippodrome, c)
        plt.axis("equal")
        title = savename.split("/")[-1]
        title = title.split(":")[0]
        plt.title(f"{title}")
        if save:
            plt.savefig(
                f"{savename}.svg",
                dpi=300,
            )
        else:
            plt.show()
        plt.clf()
        plt.close()

    def plot_centers(self, hippodrome, c):
        for color, cluster in zip(c, self.set_of_clusters):
            center = cluster.center
            if hippodrome:
                # Create a Shapely Polygon object from the given points
                polygon = LineString(center)

                # Calculate the Minkowski sum by buffering the polygon with the given radius
                minkowski_sum = polygon.buffer(cluster.radius)


                # Handle different types of geometries
                if isinstance(minkowski_sum, Polygon):
                    # Extract the exterior coordinates of the resulting shape if it's a Polygon
                    exterior_coords = np.array(minkowski_sum.exterior.coords.xy)
                    # Plot the Minkowski sum
                    plt.plot(exterior_coords[0], exterior_coords[1], color=color, alpha=0.4)
                elif isinstance(minkowski_sum, MultiPolygon):
                    # Handle MultiPolygon by accessing the exterior of the first polygon in the collection
                    for poly in minkowski_sum:
                        exterior_coords = np.array(poly.exterior.coords.xy)
                        plt.plot(exterior_coords[0], exterior_coords[1], color=color, alpha=0.4)
                else:
                    raise NotImplementedError("Unhandled geometry type: {}".format(type(minkowski_sum)))

                # Extract the exterior coordinates of the resulting shape

            if cluster.complexity > 1:
                plt.plot(
                    [x for x, _ in center],
                    [y for _, y in center],
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

    def plot_curves(self, c):
        for color, cluster in zip(c, self.set_of_clusters):
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

    def compute_assignment(self, curves=None, full=True):
        if full:
            return self._full_reassignment(curves)
        else:
            return self._recalculate_radius()

    def _full_reassignment(self, curves):
        # Collect all curves if none are given
        if curves is None:
            curves = set()
            for cluster in self.set_of_clusters:
                curves = curves.union(cluster.curves)
        # Compute the distance of each curve to each cluster
        dist_dict = {
            curve: {
                cluster: cluster.center.distance(curve)
                for cluster in self.set_of_clusters
            }
            for curve in curves
        }
        dist_min = [
                min([dist_dict[curve][cluster] for cluster in self.set_of_clusters])
                for curve in curves
            ]
        # Compute best clustering radius
        erg = max(
            dist_min
        )
        # Reassign curves to clusters
        for cluster in self.set_of_clusters:
            cluster.curves = set()

        for curve in curves:
            cluster = min(
                self.set_of_clusters, key=lambda cluster: dist_dict[curve][cluster]
            )
            cluster.curves.add(curve)

        # Recalculate the radius of each cluster
        for cluster in self.set_of_clusters:
            # KLClustering with approximated center can lead to empty clusters after reassignment
            cluster.radius = max(
                [dist_dict[curve][cluster] for curve in cluster.curves], default=0
            )
            if ASSERTION_LEVEL > 2:
                assert cluster.invariant()

        # Recalculate the radius of the clustering
        self.radius = erg
        if ASSERTION_LEVEL > 2:
            assert self.invariant()

        return self

    def _recalculate_radius(self):
        # Compute the radius of the clustering
        erg = max(
            [
                max([cluster.center.distance(curve) for curve in cluster.curves])
                for cluster in self.set_of_clusters
            ]
        )
        # Set the radius of each cluster
        for cluster in self.set_of_clusters:
            cluster.radius = erg
            if ASSERTION_LEVEL > 2:
                assert cluster.invariant()
        # Set the radius of the clustering
        self.radius = erg
        if ASSERTION_LEVEL > 2:
            assert self.invariant()

        return self

    def __iter__(self):
        return iter(self.set_of_clusters)

    def invariant(self):
        assert all([cluster.invariant() for cluster in self.set_of_clusters])
        assert self.radius >= max(
            [cluster.radius for cluster in self.set_of_clusters], default=0
        ), (
            self.radius,
            [cluster.radius for cluster in self.set_of_clusters],
        )
        return True


class KLClustering:
    def __init__(
        self, ell, curves, fast_simplification=True, track_curves=True
    ) -> None:
        self.r = np.inf
        self.big_l = ell
        self.fast_simplification = fast_simplification
        self.track_curves = track_curves
        self.curves = curves
        self.centers = {}
        self.curr_maxcurve = curves[
            0
        ]  # Could be randomised for expected better results
        self.curve_dist_dict = {curve: np.inf for curve in curves}
        self.nearest_center = {}
        self.stuck_with_bad_center = False  # See _update_curve_distances

    # @profile
    def _simplification(self, curve: Curve) -> Curve:
        return (
            curve.simplify_dist_fast(self.big_l)[0]
            if self.fast_simplification
            else curve.simplify_dist(self.big_l)
        )

    # @profile
    def main(self):
        for k in range(len(self.curves) - 1):
            self.centers.update(
                {self.curr_maxcurve: self._simplification(self.curr_maxcurve)}
            )
            assert len(self.centers.values()) == k + 1
            if self.track_curves:
                self._update_curve_distances()
                yield self._get_current_clustering()
            else:
                yield self._get_current_clustering()
                self._update_curve_distances()
        # Every curve gets its own cluster
        self.centers.update(
            {self.curr_maxcurve: self._simplification(self.curr_maxcurve)}
        )
        assert len(self.centers.values()) == len(self.curves)
        if not self.stuck_with_bad_center:
            if (
                max([self.curve_dist_dict[center] for center in self.centers])
                < self.curve_dist_dict[self.curr_maxcurve]
            ):
                self.r = self.curve_dist_dict[self.curr_maxcurve]

        clustering = Clustering(radius=self.r)
        for initial_curve, center in self.centers.items():
            cluster = Cluster(center, radius=self.r)
            if self.track_curves:
                cluster.curves = {
                    initial_curve,
                }
            clustering.add(cluster)
        yield clustering

    # @profile
    def _update_curve_distances(self):
        new_center = self.centers[self.curr_maxcurve]
        # Approximat simplification can lead to centers stealing other centers
        # To proteced against empty clusters we forbid centers to change clusters
        # This is anyway true for optimal simplifications
        new_cluster = [
            curve
            for curve in self.curves
            if curve not in self.centers
            and new_center.decide(curve, self.curve_dist_dict[curve])
        ]
        new_cluster.append(self.curr_maxcurve)
        assert len(new_cluster) > 0
        self.curve_dist_dict.update(
            {curve: new_center.distance(curve) for curve in new_cluster}
        )
        self.nearest_center.update({curve: new_center for curve in new_cluster})
        # But this also means that centers can be a curr_maxcurve
        self.curr_maxcurve = max(
            [curve for curve in self.curves if curve not in self.centers],
            key=self.curve_dist_dict.get,
        )
        # But this also means that the radius may not be the distance to the curr_maxcurve
        if self.stuck_with_bad_center:
            return
        if (
            max([self.curve_dist_dict[center] for center in self.centers])
            > self.curve_dist_dict[self.curr_maxcurve]
        ):
            self.stuck_with_bad_center = True
        else:
            self.r = self.curve_dist_dict[self.curr_maxcurve]

    # @profile
    def _get_current_clustering(self):
        clustering = Clustering(radius=self.r)
        for center in self.centers.values():
            cluster = Cluster(center, radius=self.r)
            if self.track_curves:
                cluster.curves = {
                    curve
                    for curve in self.curves
                    if self.nearest_center[curve] == center
                }
            clustering.add(cluster)
        return clustering


class LBudgetClustering:

    def __init__(
        self,
        big_l,
        r=INIT_R,
        fast_simplification=False,
        track_curves=True,
        new_decider=False,
    ) -> None:
        self.r = r
        self.upper_bound = 0
        self.lower_bound = 0
        self.clustering = Clustering(radius=r)
        self.big_l = big_l
        self.eta = 3
        self.fast_simplification = fast_simplification
        self.track_curves = track_curves
        self.new_decider = new_decider

    def _update_lower_r(self, r):
        self.lower_bound = max(self.lower_bound, r)
        if ASSERTION_LEVEL > 0:
            assert (
                self.lower_bound <= r + PRECISION <= self.upper_bound + 2 * PRECISION
            ), (self.lower_bound, r, self.upper_bound)

    def _update_r(self, r):
        self.r = r
        if ASSERTION_LEVEL > 0:
            assert (
                self.lower_bound <= r + PRECISION <= self.upper_bound + 2 * PRECISION
            ), (self.lower_bound, r, self.upper_bound)

    def _update_upper_r(self, r):
        self.upper_bound = min(r, self.upper_bound)
        for cluster in self.clustering:
            cluster.radius = self.upper_bound
        self.clustering.radius = self.upper_bound
        if ASSERTION_LEVEL > 0:
            assert (
                self.lower_bound <= r + PRECISION <= self.upper_bound + 2 * PRECISION
            ), (self.lower_bound, r, self.upper_bound)

    @property
    def complexity(self) -> int:
        return self.clustering.complexity

    def __len__(self):
        return len(self.clustering)

    def __iter__(self):
        return iter(self.clustering)

    # @profile
    def main(self, gen):
        """Computes the L-budget clustering of a set of curves.
        Output is a 3-Approximation of the L-budget clustering.
        """
        set_of_curves = set(gen)
        self.upper_bound = float("inf")
        # Exponential search
        while not self._decide(self.r, set_of_curves):
            self._update_lower_r(self.r)
            self._update_r(self.eta * self.r)
        # Binary search
        high = self.r
        low = high / self.eta
        while high - low > PRECISION:
            mid = (high + low) / 2
            if self._decide(mid, set_of_curves):
                high = mid
                self._update_upper_r(mid * self.eta)
            else:
                low = mid
                self._update_lower_r(low)
        # self._update_r(high)
        if ASSERTION_LEVEL > 1:
            assert self.invariant()
        return self

    # @profile
    def _decider_hochbaum_old(self, r, set_of_curves: Set[Curve]):
        clustering = Clustering(radius=self.eta * r)
        # Covering the curves
        uncovered_curves = copy(set_of_curves)
        while uncovered_curves:
            curve = uncovered_curves.pop()
            uncovered_curves.add(curve)
            # Computing the simplifications
            center = (
                curve.simplify_num_fast(r)
                if self.fast_simplification
                else curve.simplify_num(r)
            )
            ell = center.complexity
            if clustering.complexity + ell > self.big_l:
                # The budget is exceeded
                return False
            # Ball of radius eta * r around the center
            covered_curves = {
                z for z in uncovered_curves if center.decide(z, self.eta * r)
            }

            assert curve in covered_curves
            if ASSERTION_LEVEL > 0:
                assert len(covered_curves) > 0

            # Building the cluster
            cluster = Cluster(center, radius=3 * r)
            if self.track_curves:
                cluster.curves = cluster.curves.union(covered_curves)
            clustering.add(cluster)
            # Removing the covered curves
            uncovered_curves -= covered_curves
        if ASSERTION_LEVEL > 0:
            assert clustering.complexity <= self.big_l
        self.clustering = clustering
        return True


    # @profile
    def _decider_hochbaum(self, r, set_of_curves: Set[Curve]):
        clustering = Clustering(radius=self.eta * r)
        # Covering the curves
        uncovered_curves = copy(set_of_curves)
        while uncovered_curves:
            curve = uncovered_curves.pop()
            # Checking if the curve is already covered
            for cluster in clustering:
                if cluster.center.decide(curve, self.eta * r):
                    cluster.curves.add(curve)
                    break
            else:
                # Computing the simplifications
                center = (
                    curve.simplify_num_fast(r)
                    if self.fast_simplification
                    else curve.simplify_num(r)
                )
                ell = center.complexity
                if clustering.complexity + ell > self.big_l:
                    # The budget is exceeded
                    return False

                # Building the cluster
                cluster = Cluster(center, radius=3 * r)
                if self.track_curves:
                    cluster.curves = {curve}
                clustering.add(cluster)
        if ASSERTION_LEVEL > 0:
            assert clustering.complexity <= self.big_l
        self.clustering = clustering
        return True

    # @profile
    def _decide(self, r, set_of_curves: Set[Curve]):
        if self.new_decider:
            return self._decider_hochbaum(r, set_of_curves)
        return self._decider(r, set_of_curves)

    # @profile
    def _decider(self, r, set_of_curves: Set[Curve]):
        clustering = Clustering(radius=self.eta * r)
        # Computing the simplifications
        heap = []
        for curve in set_of_curves:
            simpl = (
                curve.simplify_num_fast(r)
                if self.fast_simplification
                else curve.simplify_num(r)
            )
            # if not simpl.decide(curve, r):
            #     print(r, simpl.distance(curve), curve.complexity, simpl.complexity)
            #     simpl = curve.simplify_num(r)
            heap.append((simpl.complexity, simpl, curve))
        hq.heapify(heap)
        # Covering the curves
        uncovered_curves = copy(set_of_curves)
        while uncovered_curves:
            ell, center, curve = hq.heappop(heap)
            if curve not in uncovered_curves:
                # Lazy deletion
                continue
            if clustering.complexity + ell > self.big_l:
                # The budget is exceeded
                return False
            # Ball of radius eta * r around the center
            covered_curves = {
                z for z in uncovered_curves if center.decide(z, self.eta * r)
            }
            covered_curves.add(curve)
            # assert curve in covered_curves
            if ASSERTION_LEVEL > 0:
                assert len(covered_curves) > 0
            # Building the cluster
            cluster = Cluster(center, radius=self.eta * r)
            if self.track_curves:
                cluster.curves = cluster.curves.union(covered_curves)
            clustering.add(cluster)
            # Removing the covered curves
            uncovered_curves -= covered_curves
        if ASSERTION_LEVEL > 0:
            assert clustering.complexity <= self.big_l
        self.clustering = clustering
        return True

    def plot_clusters(self, save=False, savename="", hippodrome=True):
        """Plots the clusters of the L-budget clustering."""
        self.clustering.plot_clusters(
            save=save, savename=savename, hippodrome=hippodrome
        )

    def compute_assignment(self, curves=None):
        """Computes the assignment of the curves to the clusters."""
        if curves is None:
            curves = set()
            for cluster in self.clustering:
                curves = curves.union(cluster.curves)
        erg = self.clustering.compute_assignment(curves)
        # self._update_upper_r(self.clustering.radius)
        # self._update_r(self.clustering.radius)
        if ASSERTION_LEVEL > 1:
            assert erg.invariant()

        return erg

    def invariant(self):
        assert self.clustering.invariant()
        if self.upper_bound is not None:
            assert self.upper_bound >= self.r - PRECISION, (
                self.upper_bound,
                self.r,
            )
        if self.lower_bound is not None:
            assert self.lower_bound <= self.r + PRECISION, (
                self.lower_bound,
                self.r,
            )

        return True


class DynamicKLClustering(LBudgetClustering):
    def __init__(
        self, k, l, alpha=2, r=INIT_R, fast_simplification=False, track_curves=True
    ) -> None:
        super().__init__(
            l,
            r=r,
            fast_simplification=fast_simplification,
            track_curves=track_curves,
        )
        self.k = k
        self.l = l
        self.alpha = alpha
        self.gamma = 2 * self.alpha
        self.eta = 2 * self.alpha**2 / (self.alpha - 1)
        self.zeta = 2 * self.alpha / (self.alpha - 1) # Widens the simplification radius without affecting the approximation factor
        self.number_of_curves = 0
        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    # @profile
    def main(self, gen: Iterable[Simplifiable]):
        for curve in gen:
            # Inserting the curve
            self.number_of_curves += 1
            self._insert(curve)

            # Merging the clusters
            while len(self.clustering) > self.k:
                if self.r == 0:
                    self._rzero()
                self._update_lower_r(self.r)
                self._update_r(self.alpha * self.r)
                self._merge_clusters()
                if DEBUG_PLOTING:
                    self.plot_clusters(
                        save=False,
                        savename=f"num_curve={len(self.clustering)}-budget={self.big_l}",
                    )
            self._update_upper_r(self.r * self.eta)
            if ASSERTION_LEVEL > 2:
                assert self.invariant()
            yield self

    def _rzero(self):
        curves = [cluster.center for cluster in self.clustering]
        radius = self._compute_r(curves)
        self._update_lower_r(radius)
        self._update_r(radius)

    # @profile
    def _insert(self, curve : Curve):
        # Inserting in existing clusters
        for cluster in self.clustering:
            if cluster.center.decide(curve, self.r * self.eta):
                cluster.add(curve)

                if DEBUG_PLOTING:
                    self.plot_clusters(
                        save=False,
                        savename=f"num_curve={len(self.clustering)}-budget={self.big_l}",
                    )
                return
        # Creating a new cluster
        self.upper_bound = float("inf")
        if (
            curve.complexity > self.l
        ):  # Should NOT be the case, otherwise the result is not a 3-approximation
            if self.fast_simplification:
                simpl = curve.simplify_dist_fast(self.l)[0]
            else:
                simpl = curve.simplify_dist(self.l)
            dist = simpl.distance(curve)

            if DEBUG_PLOTING:
                self.plot_clusters(
                    save=False,
                    savename=f"num_curve={len(self.clustering)}-budget={self.big_l}",
                )
            # Fast Forward
            while self.r < dist:
                self._update_lower_r(self.r)
                self._update_r(self.alpha * self.r)
            cluster = Cluster(simpl, self.r)

        else:
            cluster = Cluster(curve, self.r * self.eta)
        cluster.add(curve)
        self.clustering.add(cluster)

        if DEBUG_PLOTING:
            self.plot_clusters(
                save=False,
                savename=f"num_curve={len(self.clustering)}-budget={self.big_l}",
            )

    def _compute_r(self, curves):
        return min(
            [curve.distance(other) for curve, other in it.combinations(curves, 2)]
        )

    # @profile
    def _merge_clusters(self):
        self.clustering.radius = self.r * self.eta
        clusters = copy(self.clustering.set_of_clusters)
        new_clusters = set()

        for pair in it.combinations(clusters, 2):
            assert pair[0].center.decide(pair[1].center, self.gamma * self.r) == pair[1].center.decide(pair[0].center, self.gamma * self.r)

        while clusters:
            cluster1 = clusters.pop()
            cluster1.radius = self.r * self.eta
            cluster1.shorten_center_list(self.r * self.zeta) #Check later for correctness #Not needed for KLClustering

            for cluster2 in clusters:
                dec12 = cluster1.center.decide(cluster2.center, self.gamma * self.r)
                dec21 = cluster2.center.decide(cluster1.center, self.gamma * self.r)
                if dec12 != dec21:
                    print(dec12, dec21)

            ball = {
                cluster
                for cluster in clusters
                if cluster1 is not cluster
                and cluster1.center.decide(cluster.center, self.gamma * self.r)
            }

            for cluster2 in ball:
                cluster1.merge(cluster2)
                clusters.remove(cluster2)


            new_clusters.add(cluster1)

        self.clustering.set_of_clusters = new_clusters

    def invariant(self):
        assert self.lower_bound <= self.upper_bound <= self.lower_bound * self.eta
        assert all([cluster.radius < float("inf") for cluster in self.clustering])
        assert all([len(cluster) for cluster in self.clustering])
        assert super().invariant()
        assert len(self.clustering) <= self.k, (len(self.clustering), self.k)
        for cluster1 in self.clustering:
            for cluster2 in self.clustering:
                if cluster1 == cluster2:
                    continue
                assert not cluster1.center.decide(
                    cluster2.center, self.gamma * self.lower_bound
                ), cluster1.center.distance(cluster2.center)
        return True


class ScalingDynamicKLClustering:
    def __init__(
        self,
        k,
        l,
        approx_eps = 1,
        r=INIT_R,
        track_curves=False,
        fast_simplification=False,
    ) -> None:
        self.r = r
        self.k = k
        self.l = l
        self.approx_eps = approx_eps
        self.fast_simplification = fast_simplification
        self.track_curves = track_curves
        self.parallel_session = []
        self.current_best = None

        self.alpha = self.compute_alpha(approx_eps)
        self.m = math.ceil(self.alpha - 1) # local minimum at a = m + 1
        self.m = math.ceil(self.m * math.log(self.m + 1, 2))
        self.make_instances()



    def compute_alpha(self, eps):
        # Catches the case where the equation is undefined, but has a solution
        if self.approx_eps >= 6:
            return 1
        a = sp.symbols('a', positive=True, interger=True)
        # local minimum at a = m + 1
        func = 2 * (1 + 1/(a-1)) * a**(1/(a-1))
        equation = sp.Eq(func - 2, eps)
        m_val = sp.nsolve(equation, a, 1.0001, verify=False)
        a = int(sp.ceiling(m_val))
        return a

    def make_instances(self):
        for i in range(1, self.m + 1):
            r_i = self.alpha ** ((i / self.m) - 1) * self.r
            self.parallel_session.append(
                DynamicKLClustering(
                    self.k,
                    self.l,
                    alpha=self.alpha,
                    r=r_i,
                    fast_simplification=self.fast_simplification,
                    track_curves=self.track_curves,
                )
            )

    # @profile
    def main(self, gen):
        ergs = []
        for session in self.parallel_session:
            ergs.append(session.main(gen))
        for _ in gen:
            res = [next(erg) for erg in ergs]
            self.current_best = min(res, key=lambda x: x.r)
            if ASSERTION_LEVEL > 2:
                assert self.invariant()
            yield res

    def plot_clusters(self, c="r", save=False, savename="", hippodrome=True):
        self.current_best.plot_clusters(
            c=c, save=save, savename=savename, hippodrome=hippodrome
        )

    def invariant(self):
        assert all([session.invariant() for session in self.parallel_session])
        assert self.current_best.invariant()
        return True


class RangeCluster:
    def __init__(
        self,
        obj: Simplifiable,
        base,
        big_l,
        radius=None,
        track_curves=False,
        fast_simplification=False,
    ) -> None:
        self.cluster_range = []
        self.track_curves = track_curves
        self.fast_simplification = fast_simplification
        self.radius = radius

        if track_curves:
            self.curves = {obj}
        range_simpl = list(
            obj.range_simplification(
                base, big_l, fast_simplification=self.fast_simplification
            )
        )
        # range_simpl = obj.fast_approximate_range_simplification(base, big_l)

        for idx, obj in enumerate(range_simpl):
            simpl = obj[0]
            dist_upper = range_simpl[idx + 1][1] if idx + 1 < len(range_simpl) else float("inf")
            dist_lower = obj[1]
            self.cluster_range.append([simpl, dist_upper, dist_lower])
        if ASSERTION_LEVEL > 0:
            assert len(self.cluster_range) > 0
            assert sum([center.complexity for center, _, _ in self.cluster_range]) <= (
                big_l * base**2 - 1
            ) / (base - 1), (
                (big_l * base**2 - 1) / (base - 1),
                sum([center.complexity for center, _, _ in self.cluster_range]),
            )  # TODO: make correct factor

        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    @property
    def center(self):
        return self.cluster_range[0][0]

    @property
    def w(self):
        return self.cluster_range[0][1]

    @property
    def r(self):
        return self.cluster_range[0][2]

    @property
    def triplet(self):
        return self.center, self.w, self.r

    def shorten_center_list(self, radius):
        index = bisect([w for _, w, r in self.cluster_range], radius)
        if ASSERTION_LEVEL > 2:
            assert index == 0 or self.cluster_range[index - 1][1] <= radius
            assert (
                index == len(self.cluster_range)
                or self.cluster_range[index][1] > radius
            )
        self.cluster_range = self.cluster_range[index:]
        if ASSERTION_LEVEL > 0:
            assert len(self.cluster_range)

        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    # @profile
    def add(self, curve):
        for cluster in self:
            center, _, r = cluster
            if not center.decide(curve, r):
                cluster[2] = center.distance(curve)

        if self.track_curves:
            self.curves.add(curve)

        if ASSERTION_LEVEL > 1:
            assert self.invariant()

    # @profile
    def merge(self, other):

        if ASSERTION_LEVEL > 1:
            assert self.invariant()
            assert other.invariant()

        center2, _, r2 = other.triplet
        for cluster in self:
            center1, _, r1 = cluster
            cluster[2] = (
                r1
                if center1.decide(center2, r1 - r2)
                else center1.distance(center2) + r2
            )
            # cluster[2] = max(r1, center1.distance(center2)+ r2)

        if ASSERTION_LEVEL > 1:
            assert self.invariant()

        if self.track_curves:
            self.curves = self.curves.union(other.curves)

        if ASSERTION_LEVEL > 1:
            assert self.invariant()

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

    def invariant(self):
        if self.track_curves:
            assert all(
                [
                    self.center.decide(curve, self.radius + PRECISION)
                    for curve in self.curves
                ]
            )
        return True

    def __len__(self):
        return len(self.curves)


class RangeClustering(Clustering):
    def __init__(self, radius=None) -> None:
        super().__init__(radius=radius)
        self.clustering: Set[RangeCluster] = set()


class DynamicLBudgetClustering(DynamicKLClustering):
    def __init__(
        self,
        big_l,
        simpl_augm = 1,
        r=INIT_R,
        alpha=2.0,
        track_curves=False,
        initial_simplify=False,
        fast_simplification=False,
    ) -> None:
        self.base = 1 + simpl_augm
        super().__init__(
            big_l,
            big_l,
            max(alpha, 2),
            r=r,
            fast_simplification=fast_simplification,
            track_curves=track_curves,
        )

        self.base = 1 + simpl_augm
        self.big_l = big_l
        self.clustering = RangeClustering(radius=r)
        self.initial_simplify = initial_simplify

    # @profile
    def main(self, gen):
        for obj in gen:
            self._insert(obj)
            self._make_clustering_valid()
            if ASSERTION_LEVEL > 2:
                assert self.invariant()
            yield self

    # @profile
    def _insert(self, obj: Simplifiable):
        self.number_of_curves += 1
        for cluster in self.clustering:
            if obj.decide(cluster.center, self.r * self.eta):
                cluster.add(obj)
                return
        if self.initial_simplify:
            obj = (
                obj.simplify_num_fast(self.r / self.eta)
                if self.fast_simplification
                else obj.simplify_num(self.r / self.eta)
            )
        cluster = RangeCluster(
            obj,
            self.base,
            self.big_l,
            radius=self.r,
            track_curves=self.track_curves,
            fast_simplification=self.fast_simplification,
        )
        cluster.shorten_center_list(self.r * self.zeta)

        self.clustering.add(cluster)

    def _make_clustering_valid(self):
        while self.complexity > self.base * self.big_l:
            self._update_lower_r(self.r)
            self._update_r(self.alpha * self.r)
            self._merge_clusters()

    def invariant(self):
        assert super().invariant()
        assert all([cluster.invariant() for cluster in self.clustering])
        assert self.complexity <= self.base * self.big_l
        assert all(
            [
                not cluster_a.center.decide(
                    cluster_b.center, self.gamma * self.lower_bound
                )
                for cluster_a, cluster_b in it.combinations(self.clustering, 2)
            ]
        )
        return True


class ScalingDynamicLBudgetClustering(ScalingDynamicKLClustering):
    def __init__(
        self,
        big_l,
        simpl_augm = 1,
        approx_eps = 1,
        r=INIT_R,
        track_curves=False,
        initial_simplify=False,
        fast_simplification=False,
    ) -> None:
        self.initial_simplify = initial_simplify
        self.big_l = big_l
        self.r = r
        self.k = big_l
        self.l = big_l
        self.approx_eps = approx_eps
        self.fast_simplification = fast_simplification
        self.track_curves = track_curves
        self.parallel_session = []
        self.current_best = None
        self.simpl_augm = simpl_augm
        self.alpha = self.compute_alpha(approx_eps)
        self.m = math.ceil(self.alpha - 1) # local minimum at a = m + 1
        # self.m = math.ceil(self.m * math.log(self.m + 1, 2))
        self.make_instances()

    def make_instances(self):
        for i in range(1, self.m + 1):
            r_i = self.alpha ** ((i / self.m) - 1) * self.r
            self.parallel_session.append(
                DynamicLBudgetClustering(
                    self.big_l,
                    self.simpl_augm,
                    r_i,
                    self.alpha,
                    track_curves=self.track_curves,
                    initial_simplify=self.initial_simplify,
                    fast_simplification=self.fast_simplification,
                )
            )
