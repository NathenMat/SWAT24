import cProfile
import pstats

import Fred as fred
import pandas as pd
import numpy as np
import os

from l_budget_clustering import (
    Curve,
    LBudgetClustering,
    DynamicLBudgetClustering,
)

fred.config.verbosity = 0

file = "/home/lukas/Nextcloud/Dissertation/Daten/Data_for_Mann_et_al_RSBL/Pigeon Flight Data.csv"


def kl_center(curves, big_l):
    distance_func = 0  # continuous Frechet
    clusterings = []
    for k in range(1, big_l):
        l = big_l // k
        if l <= 1:
            break
        clusterings.append(
            fred.discrete_klcenter(
                k, l, curves, fast_simplification=True, distance_func=distance_func
            )
        )

    return min(clusterings, key=lambda x: x.value)


def static_lbudget(curves, big_l):
    return LBudgetClustering(big_l, 1e-8, fast_simplification=True).main(curves)


def dynamic_lbudget(curves, big_l):
    return list(
        DynamicLBudgetClustering(
            big_l, 0.5, 1e-8, track_curves=True, fast_simplification=True
        ).main(curves)
    )[-1]


def test_instance(algorithm, curves, big_l, data_frame, praefix=""):
    profiler = cProfile.Profile()
    if algorithm == "kl_center":
        algo = kl_center
    elif algorithm == "static_lbudget":
        algo = static_lbudget
    elif algorithm == "dynamic_lbudget":
        algo = dynamic_lbudget
    else:
        raise ValueError("Unknown algorithm")
    profiler.enable()
    clustering = algo(curves, big_l)
    profiler.disable()
    if algorithm == "kl_center":
        clustering.compute_assignment(curves)
    else:
        clustering.compute_assignment()
    if algorithm != "kl_center":
        num_clusters = len(clustering)
        radius = clustering.upper_bound
        complexity = clustering.complexity
    else:
        num_clusters = len(clustering)
        radius = clustering.value
        complexity = sum([len(clustering[i]) for i in range(len(clustering))])

    filenamestart = f"{praefix}{algorithm} num_curve={len(curves)} budget={big_l}: "
    filenamemid = f"num_cluster={num_clusters}, radius={radius}, cplx={complexity}"
    filename = filenamestart + filenamemid

    if algorithm == "kl_center":
        fred.plot_clustering(
            clustering,
            curves,
            vertex_markings=False,
            savename=filename,
            saveextension="jpg",
            legend=False,
        )
    else:
        clustering.plot_clusters(save=True, savename=filename)
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
    return data_frame


data_frame = pd.DataFrame()

csv = pd.read_csv(file, usecols=["Longitude", "Latitude", "Filename"])

# for location in ["bh_", "ch_", "H", "FT_", ]:
filenamestart = f"./Experiments/"  # f"./Experiments{location}/"
if not os.path.exists(filenamestart):
    os.makedirs(filenamestart)
# csv_location = csv[csv["Filename"].str.startswith(location)]
csv_grouped = csv.groupby("Filename")
curves_lbudget = []
for group, _ in zip(csv_grouped, range(1000)):
    curve = np.array(
        [
            (entry[1]["Longitude"], entry[1]["Latitude"])
            for entry in csv_grouped.get_group(str(group[0])).iterrows()
        ]
    )
    curves_lbudget.append(Curve(curve))
for budget in [
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
]:
    for number_of_curves in [
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        897,
    ]:
        for algorithm in ["static_lbudget", "dynamic_lbudget", "kl_center"]:
            if algorithm == "kl_center":
                curves = fred.Curves()
                for curve in curves_lbudget[:number_of_curves]:
                    curves.add(curve.curve)
            else:
                curves = curves_lbudget[:number_of_curves]
            data_frame = test_instance(
                algorithm,
                curves,
                budget,
                data_frame,
                praefix=f"{filenamestart}Pidgeon ",
            )
            data_frame.to_csv(f"{filenamestart}Pidgeon.csv")
