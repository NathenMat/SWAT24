import cProfile
import pstats
import itertools as it
import pandas as pd
import numpy as np
import os
import signal
import random

from pyprof2calltree import convert

from l_budget_clustering import (
    Curve,
    Cluster,
    Clustering,
    KLClustering,
    DynamicKLClustering,
    ScalingDynamicKLClustering,
    LBudgetClustering,
    DynamicLBudgetClustering,
    ScalingDynamicLBudgetClustering,
)


CLUSTER_ERROR = 1
SIMPLIFICATION_ERROR = 0.01
PYTHON = True

file = "/home/lukas/NextcloudArbeit/Dissertation/Daten/Data_for_Mann_et_al_RSBL/Pigeon Flight Data.csv"


def kl_center(curves, big_l, fast_simplification=True):
    # distance_func = 0  # continuous Frechet
    clusterings = []
    old_ell = None
    instance = None
    for k in range(1, min(big_l, len(curves) + 1)):
        ell = big_l // k
        if ell <= 1:
            break
        if ell == old_ell:
            erg = next(instance)
        else:
            instance = KLClustering(ell, curves, fast_simplification).main()
            obj = zip(range(k), instance)
            erg = list(obj)[-1][1]
        clusterings.append(erg)
    return clusterings


def static_lbudget(curves, big_l, fast_simplification=True):
    return LBudgetClustering(big_l, fast_simplification=fast_simplification).main(
        curves
    )


def static_lbudget_new(curves, big_l, fast_simplification=True):
    return LBudgetClustering(big_l, fast_simplification=fast_simplification, new_decider=True).main(
        curves
    )


def dynamic_kl_center(curves, big_l, fast_simplification=True):
    clusterings = []
    for k in range(1, min(big_l, len(curves) + 1)):
        l = big_l // k
        if l <= 1:
            break
        clusterings.append(
            list(
                DynamicKLClustering(k, l, fast_simplification=fast_simplification).main(
                    curves
                )
            )[-1].clustering
        )
    return clusterings


def scaling_dynamic_kl_center(curves, big_l, fast_simplification=True):
    clusterings = []
    for k in range(1, min(big_l, len(curves) + 1)):
        l = big_l // k
        if l <= 1:
            break
        clusterings.append(
            list(
                ScalingDynamicKLClustering(
                    k,
                    l,
                    CLUSTER_ERROR,
                    track_curves=True,
                    fast_simplification=fast_simplification,
                ).main(curves)
            )[-1]
        )
    flat_clusterings = []
    for clustering in clusterings:
        flat_clusterings.extend(clustering)
    return flat_clusterings


def dynamic_lbudget(curves, big_l, fast_simplification=True):
    erg = list(
        DynamicLBudgetClustering(
            big_l, SIMPLIFICATION_ERROR, track_curves=True, fast_simplification=fast_simplification
        ).main(curves)
    )[-1]
    return erg


def scaling_dynamic_lbudget(curves, big_l, fast_simplification=True):
    erg = list(
        ScalingDynamicLBudgetClustering(
            big_l, SIMPLIFICATION_ERROR, CLUSTER_ERROR, track_curves=True, fast_simplification=fast_simplification
        ).main(curves)
    )[-1]
    return erg


def convert_clustering(cpp_clustering, curves):
    clustering_result = cpp_clustering
    clustering_result = clustering_result.compute_assignment(curves)
    radius = float("inf")
    clustering = Clustering(radius=radius)
    for i, center in enumerate(clustering_result):
        if clustering_result.assignment.count(i) == 0:
            continue
        cluster = Cluster(Curve(center), radius=radius)
        assigned_curves = []
        for j in range(clustering_result.assignment.count(i)):
            curve = curves[clustering_result.assignment.get(i, j)]
            assigned_curves.append(Curve(curve))
        cluster.expand(assigned_curves)
        clustering.add(cluster)
    return clustering


# Define a custom exception for timeout
class TimeoutError(Exception):
    pass


# Define the signal handler function
def signal_handler(signum, frame):
    raise TimeoutError("Algorithm execution timed out")


# Define a function to run the algorithm with a timeout
def run_algorithm_with_timeout(
    algo, curves, big_l, fast_simplification=True, timeout=600
):
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(timeout)  # Set a timeout of {timeout} seconds
        result = algo(curves, big_l, fast_simplification=fast_simplification)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        print(f"Timeout while running {algo.__name__}")
        return None


def test_instance(algorithm, curves, big_l, results, praefix=""):
    global blocked_values
    profiler = cProfile.Profile()
    if algorithm == "kl_center":
        algo = kl_center
    elif algorithm == "static_lbudget":
        algo = static_lbudget
    elif algorithm == "static_lbudget_new":
        algo = static_lbudget_new
    elif algorithm == "dynamic_lbudget":
        algo = dynamic_lbudget
    elif algorithm == "dynamic_kl_center":
        algo = dynamic_kl_center
    elif algorithm == "scaling_dynamic_lbudget":
        algo = scaling_dynamic_lbudget
    elif algorithm == "scaling_dynamic_kl_center":
        algo = scaling_dynamic_kl_center
    else:
        raise ValueError("Unknown algorithm")

    num_clusters = None
    radius = None
    complexity = None

    filenamestart = f"{praefix}{algorithm} num_curve={len(curves)} budget={big_l}: "
    filenamemid = f"num_cluster={num_clusters}, radius={radius}, cplx={complexity}"
    filename = filenamestart + filenamemid

    profiler.enable()
    clustering = run_algorithm_with_timeout(algo, curves, big_l)
    profiler.disable()

    if clustering is None:
        bigger_ls = [ell for ell in budget_list if ell >= big_l]
        more_curves = [
            num_curves for num_curves in num_curves_list if num_curves >= len(curves)
        ]
        blocked_values = blocked_values.union(
            set(it.product(bigger_ls, more_curves, [algorithm]))
        )
        print(f"Timeout occurred while running {algorithm}")
        clustering = Clustering()
        radius = pd.NA
    elif isinstance(clustering, list):
        clustering = compute_best_clustering(curves, clustering)
    else:
        clustering = clustering.compute_assignment()

    radius = clustering.radius

    results = export_stats(
        algorithm, curves, big_l, filenamestart, profiler, clustering, radius, results
    )
    return results


def export_stats(
    algorithm, curves, big_l, namestart, profiler, clustering, radius, data_frame
):
    num_clusters = len(clustering)
    complexity = clustering.complexity

    filenamemid = f"num_cluster={num_clusters}, radius={radius}, cplx={complexity}"
    filename = namestart + filenamemid

    clustering.plot_clusters(save=True, savename=filename, hippodrome=True)
    stats = pstats.Stats(profiler)
    data_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "Algorithm": algorithm,
                    "Curves": len(curves),
                    "Budget": big_l,
                    "Radius": radius,
                    "Time": stats.total_tt,
                    "Num Clusters": num_clusters,
                    "Complexity": complexity,
                },
                index=[0],
            ),
            data_frame,
        ],
        ignore_index=True,
    )
    stats.dump_stats(f"{filename}.prof")
    convert(profiler.getstats(), f"{filename}.kgrid")
    return data_frame


def compute_best_clustering(curves, clustering):
    clusterings = clustering
    clusterings = [clustering.compute_assignment(curves) for clustering in clusterings]
    clustering = min(clusterings, key=lambda x: x.radius)
    return clustering


csv = pd.read_csv(file, usecols=["Longitude", "Latitude", "Filename"])

for location in [
    "ch_",
    # "bh_",
    # "H",
    # "FT_",
]:
    filenamestart = f"./Experiments47{location}/"  # {location}/"
    if not os.path.exists(filenamestart):
        os.makedirs(filenamestart)
    if os.path.exists(f"{filenamestart}Pidgeon.csv"):
        data_frame = pd.read_csv(f"{filenamestart}Pidgeon.csv")
    else:
        data_frame = pd.DataFrame(columns=["Budget", "Curves", "Algorithm"])

    csv_location = csv[csv["Filename"].str.startswith(location)]
    csv_grouped = csv_location.groupby("Filename")
    curves_lbudget = []
    for group, _ in zip(csv_grouped, range(1000)):
        thing = csv_grouped.get_group(str(group[0]))
        curve = np.array([thing["Longitude"], thing["Latitude"]]).T
        curve = Curve(curve).simplify_dist_fast(400)[0]
        curves_lbudget.append(curve)

    random.seed(0)
    budget_list = np.unique(np.geomspace(2, 1_000, num=20, dtype=int))  # 54
    num_curves_list = np.unique(np.geomspace(2, 897, num=23, dtype=int))  # 23
    selected_curves = curves_lbudget
    selected_curves : list[Curve] = random.sample(curves_lbudget, 23)
    selected_curves.append(curves_lbudget[119])
    selected_curves.append(curves_lbudget[82])
    # selected_curves = [curve.simplify_dist_fast(10000)[0] for curve in selected_curves]
    # budget_list = [200, ]
    num_curves_list = [slice(0, 120, 2),] #slice(0, 120, 3), slice(0, 120, 4), slice(0, 120, 5), slice(0, 120, 6), slice(0, 120, 7)]
    product = it.product(budget_list, num_curves_list)
    sorted_product = sorted(product, key=lambda x: x[0] * 1)
    blocked_values = set()
    computed_values = [
        (budget, number_of_curves, algorithm)
        for (budget, number_of_curves, algorithm) in data_frame[
            ["Budget", "Curves", "Algorithm"]
        ].values
    ]
    blocked_values = blocked_values.union(set(computed_values))
    for budget, number_of_curves in sorted_product:
        for algorithm in [
            "static_lbudget_new",
            "static_lbudget",
            "dynamic_lbudget",
            "scaling_dynamic_lbudget",
            "kl_center",
            "dynamic_kl_center",
            "scaling_dynamic_kl_center",
        ]:

            # if (budget, number_of_curves, algorithm) in blocked_values:
            #     continue
            curves = curves_lbudget[number_of_curves] #[:number_of_curves]
            data_frame = test_instance(
                algorithm,
                selected_curves,
                budget,
                data_frame,
                praefix=f"{filenamestart}Pidgeon ",
            )
            data_frame.to_csv(f"{filenamestart}Pidgeon.csv")
