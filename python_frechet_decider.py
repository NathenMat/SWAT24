""" Part of the code was translated from the original C++ code provided by Denis Rohde.
"""

import math
import numba
import numpy as np
from bisect import bisect
import logging
from line_profiler import profile

from Fred import (
    discrete_frechet,
    hausdorff,
    decide_continuous_frechet,
)


from light_frechet import (
    # LightCurve,
    # LightCurves,
    calc_distance,
    less_than_with_filters,
    simplify_num,
    simplify_eps,
)

PRECISION = 1e-8

logger = logging.getLogger(__name__)


#@profile
def cpp_decide(curve1, curve2, eps: float) -> bool:
    # return decide_continuous_frechet(curve1.cppc, curve2.cppc, eps)
    # assert less_than_with_filters(curve1.nparr, curve2.nparr, eps) == less_than_with_filters(curve2.nparr, curve1.nparr, eps)
    return less_than_with_filters(curve1.nparr, curve2.nparr, eps)


#@profile
def cpp_discrete_frechet_distance(curve1, curve2):
    return discrete_frechet(curve1.cppc, curve2.cppc).value


#@profile
def cpp_hausdorff_distance(curve1, curve2):
    return hausdorff(curve1.cppc, curve2.cppc)



def decide_segment(segment, curve, eps: float) -> bool:
    return less_than_with_filters(segment.nparr, curve.nparr, eps)
    # return _decide_segment(segment.nparr, curve.nparr, eps)

#@profile
@numba.njit
def _decide_segment(segment : np.ndarray, curve : np.ndarray, eps: float) -> bool:
    assert curve.shape[0] >= segment.shape[0]
    decision = line_stabber(segment, curve, eps)
    return decision


#@profile
@numba.njit
def line_stabber(segment, curve, eps: float) -> bool:
    s_i = 0
    line = segment[-1] - segment[0]
    sqr_eps = eps**2
    for i in range(curve.shape[0]):
        i_v = _ball_intersection_interval(sqr_eps, line, curve[i] - segment[0])
        s_i = max(s_i, i_v[0])
        if s_i > i_v[1]:
            return False
    return i_v[1] >= 1


#@profile
# @numba.njit
def other_line_stabber(segment, curve, eps: float) -> bool:
    i_v = free_space_intervals_vert(eps**2, curve, segment)
    s_i = np.maximum.accumulate(i_v[:, 0, 0])
    e_i = i_v[:, 0, 1]
    valid_i = s_i <= e_i
    return np.all(valid_i)


# @numba.njit
def decide(curve1, curve2, eps: float) -> bool:
    n1 = curve1.shape[0]
    n2 = curve2.shape[0]
    try:
        sqr_eps = eps**2
    except OverflowError:
        print("ERROR: eps too large")
        sqr_eps = math.inf

    if n1 < 2 or n2 < 2:
        print("NOTIFICATION: Deciding small curves")
        if n1 == 1:
            return np.all(np.sum((curve1[0] - curve2) ** 2, axis=1) <= sqr_eps)
        if n2 == 1:
            return np.all(np.sum((curve2[0] - curve1) ** 2, axis=1) <= sqr_eps)
        print("WARNING: comparison possible only for curves of at least one point")
        return False

    if curve1.shape[1] != curve2.shape[1]:
        print(
            "WARNING: comparison possible only for curves of equal number of dimensions"
        )
        return False

    if (
        np.sum((curve1[0] - curve2[0]) ** 2) > sqr_eps
        or np.sum((curve1[-1] - curve2[-1]) ** 2) > sqr_eps
    ):
        return False

    is_less_than = (
        _less_than_or_equal(sqr_eps, curve2, curve1)
        if n1 > n2
        else _less_than_or_equal(sqr_eps, curve1, curve2)
    )
    return is_less_than


# @numba.njit
def _less_than_or_equal(dist_sqr, curve_short, curve_long):
    assert curve_short.shape[0] <= curve_long.shape[0]
    infty = np.Infinity
    n1, n2 = curve_short.shape[0], curve_long.shape[0]

    reachable_horizontal = np.full((n1 - 1, n2), infty, dtype=np.float64)
    reachable_vertical = np.full((n1, n2 - 1), infty, dtype=np.float64)

    # lbl_hori, lbl_vert = fsi_line_by_line(dist_sqr, curve_short, curve_long,)

    freespace_interval_horizontal, freespace_interval_vertical = (
        fsi_one_by_one(dist_sqr, curve_short, curve_long)
        if n1 < 10
        else free_space_intervals(dist_sqr, curve_short, curve_long)
    )

    for i in range(n1 - 1):
        if (
            np.dot(curve_long[0] - curve_short[i], curve_long[0] - curve_short[i])
            > dist_sqr
        ):
            break
        reachable_horizontal[i, 0] = 0

    for j in range(n2 - 1):
        if (
            np.dot(curve_short[0] - curve_long[j], curve_short[0] - curve_long[j])
            > dist_sqr
        ):
            break
        reachable_vertical[0, j] = 0

    for i in range(n1 - 1):
        indices_left_reachable = np.where(reachable_vertical[i, :] != infty)[0]
        splited_free_space = np.split(
            freespace_interval_horizontal[:, i, :], indices_left_reachable + 1
        )
        erg = []
        for arr in splited_free_space:
            left_values, right_values = arr[:, 0], arr[:, 1]
            left_values_envelope = np.maximum.accumulate(left_values)
            right_values_envelope = np.minimum.accumulate(right_values)
            open_interval = left_values_envelope - right_values_envelope
            idx = bisect(open_interval, 0)
            arr4 = np.concatenate(
                [left_values_envelope[:idx], np.full(len(left_values) - idx, infty)]
            )
            erg.append(arr4)
        reachable_horizontal[i] = np.concatenate(erg)

        open_right = freespace_interval_vertical[i + 1, :, 0] >= 0
        reachable_bottom = reachable_horizontal[i, :-1] != infty
        fully_reachable_right = np.logical_and(open_right, reachable_bottom)
        reachable_vertical[i + 1, fully_reachable_right] = freespace_interval_vertical[
            i + 1, fully_reachable_right, 0
        ]
        not_reachable_bottom = ~reachable_bottom
        overlap_right = (
            reachable_vertical[i, :] <= freespace_interval_vertical[i + 1, :, 1]
        )
        left_reachable = np.logical_and(not_reachable_bottom, overlap_right)
        partialy_reachable_right = np.logical_and(open_right, left_reachable)
        reachable_vertical[i + 1, partialy_reachable_right] = np.maximum(
            freespace_interval_vertical[i + 1, partialy_reachable_right, 0],
            reachable_vertical[i, partialy_reachable_right],
        )

    return reachable_vertical[-1, -1] < infty


# @numba.njit
def fsi_line_by_line(dist_sqr, curve_short, curve_long):
    n1 = curve_short.shape[0]
    n2 = curve_long.shape[0]
    freespace_interval_horizontal = np.empty((n2, n1 - 1, 2), dtype=np.float64)
    freespace_interval_vertical = np.empty((n1, n2 - 1, 2), dtype=np.float64)

    for j in range(n2):
        segment = curve_short[1:] - curve_short[:-1]
        pp_segment = curve_long[j] - curve_short[:-1]
        erg = _ball_intersection_interval_vectorized(dist_sqr, segment, pp_segment)
        freespace_interval_horizontal[j] = erg

    for i in range(n1):
        segment = curve_long[1:] - curve_long[:-1]
        pp_segment = curve_short[i] - curve_long[:-1]
        erg = _ball_intersection_interval_vectorized(dist_sqr, segment, pp_segment)
        freespace_interval_vertical[i] = erg

    return freespace_interval_horizontal, freespace_interval_vertical


# @numba.njit
def fsi_one_by_one(dist_sqr, curve_short, curve_long):
    n1, n2 = curve_short.shape[0], curve_long.shape[0]
    freespace_interval_horizontal = np.full((n2, n1 - 1, 2), -1, dtype=np.float64)
    freespace_interval_vertical = np.full((n1, n2 - 1, 2), -1, dtype=np.float64)

    for j in range(n2 - 1):
        line = curve_long[j + 1] - curve_long[j]
        for i in range(n1):
            erg = _ball_intersection_interval(
                dist_sqr, line, curve_short[i] - curve_long[j]
            )
            freespace_interval_vertical[i, j] = erg

    for i in range(n1 - 1):
        line = curve_short[i + 1] - curve_short[i]
        for j in range(n2):
            erg = _ball_intersection_interval(
                dist_sqr, line, curve_long[j] - curve_short[i]
            )
            freespace_interval_horizontal[j, i] = erg

    return freespace_interval_horizontal, freespace_interval_vertical


#@profile
@numba.njit
def _ball_intersection_interval(distance_sqr, segment, vector) -> tuple[float]:
    quadratischer_term = np.dot(segment, segment)
    konstanter_term = np.dot(vector, vector) - distance_sqr

    if quadratischer_term == 0:
        if konstanter_term <= distance_sqr:
            return 0, 1
        else:
            return 0, -1
            # return -1.9, -2

    linearer_term = np.dot(vector, segment)
    discriminant = linearer_term**2 - quadratischer_term * konstanter_term

    if discriminant < 0:
        return 0, -1
        # return -2.9, -3

    discriminant_sqrt = math.sqrt(discriminant)

    unterer_wert = (linearer_term - discriminant_sqrt) / (quadratischer_term)
    oberer_wert = (linearer_term + discriminant_sqrt) / (quadratischer_term)

    if unterer_wert > 1 or oberer_wert < 0:
        return 0, -1
        # return -3.9, -4

    return max(0, unterer_wert), min(1, oberer_wert)


# @numba.njit
def _ball_intersection_interval_vectorized(distance_sqr, segment, pp_segment):

    full_result = np.full(
        (*segment.shape[:2], 2), np.array([0, -1]), dtype=np.float64
    )  # Initialize result array

    quadratic_term = np.sum(segment * segment, axis=-1)
    const_term = np.sum(pp_segment * pp_segment, axis=-1) - distance_sqr

    zero_quadratic = quadratic_term == 0  # np.isclose(quadratic_term, 0)

    if np.any(zero_quadratic):
        result = full_result[zero_quadratic]
        # Check if quadratischer_term is close to zero
        zw = const_term[zero_quadratic] <= distance_sqr

        if np.any(zw):
            # result[zw, 0] = 0
            result[zw, 1] = 1

        # if np.any(~zw):
        #     result[~zw, 0] = -1.9
        #     result[~zw, 1] = -2

        full_result[zero_quadratic] = result

    neg_zero_quadratic = ~zero_quadratic

    if np.any(neg_zero_quadratic):
        result = full_result[neg_zero_quadratic]
        segment = segment[neg_zero_quadratic]
        pp_segment = pp_segment[neg_zero_quadratic]
        quadratic_term = quadratic_term[neg_zero_quadratic]
        const_term = const_term[neg_zero_quadratic]

        linear_term = np.sum(segment * pp_segment, axis=-1)
        discriminant = np.square(linear_term) - quadratic_term * const_term

        non_negative_discriminant = discriminant >= 0

        # neg_discriminant = ~non_negative_discriminant

        # if np.any(neg_discriminant):
        #     result[neg_discriminant, 0] = -2.9
        #     result[neg_discriminant, 1] = -3

        if np.any(non_negative_discriminant):
            zw_result = result[non_negative_discriminant]
            diff_value = np.sqrt(discriminant[non_negative_discriminant])
            linear_term = linear_term[non_negative_discriminant]
            quadratic_term = quadratic_term[non_negative_discriminant]

            lower_value = (linear_term - diff_value) / quadratic_term
            upper_value = (linear_term + diff_value) / quadratic_term
            valid = np.logical_and(lower_value <= 1, upper_value >= 0)

            # not_valid = ~valid

            # if np.any(not_valid):
            #     zw_result[not_valid, 0] = -3.9
            #     zw_result[not_valid, 1] = -4

            if np.any(valid):
                clipped_lower_value = np.maximum(0, lower_value)[valid]
                clipped_upper_value = np.minimum(1, upper_value)[valid]

                zw_result[valid, 0] = clipped_lower_value
                zw_result[valid, 1] = clipped_upper_value

            result[non_negative_discriminant] = zw_result

        full_result[neg_zero_quadratic] = result

    return full_result


#@profile
# @numba.njit
def free_space_intervals(distance_sqr, curve_short, curve_long):
    """faster on large curves
    makes heavy use of numpy vectorization and processing of multiple intervals at once
    main idea is to calculate everything at once and then filter out the results
    """

    n, m = curve_short.shape[0], curve_long.shape[0]

    segment1 = curve_short[1:] - curve_short[:-1]  # (n-1, d)
    segment2 = curve_long[1:] - curve_long[:-1]  # (m-1, d)

    quadratic_term1 = np.sum(segment1 * segment1, axis=-1)  # (n-1,)
    quadratic_term2 = np.sum(segment2 * segment2, axis=-1)  # (m-1,)

    zero_quadratic1 = quadratic_term1 == 0  # (n-1, )
    zero_quadratic2 = quadratic_term2 == 0  # (m-1, )

    fill_array = np.array([0, -1], dtype=np.float64)

    result_verti = np.full((n, m - 1, 2), fill_array, dtype=np.float64)  # (n, m - 1, 2)
    result_hori = np.full((n - 1, m, 2), fill_array, dtype=np.float64)  # (n - 1, m, 2)

    pp_segment = curve_long[:] - curve_short[:, np.newaxis]  # (n, m, d)
    const_term = np.sum(pp_segment * pp_segment, axis=-1) - distance_sqr  # (n, m)

    if np.any(zero_quadratic1):
        mask_full_1 = np.logical_and(
            const_term[:-1, :] <= distance_sqr, zero_quadratic1[:, np.newaxis]
        )  # (n-1, m)
        result_hori[mask_full_1, 1] = 1
        # mask_empty_1 = const_term[:-1, :] > distance_sqr  # (n-1, m)
        # mask_empty_1 = np.logical_and(
        #     mask_empty_1, zero_quadratic1[:, np.newaxis]
        # )  # (n-1, m)
        # result_hori[mask_empty_1, 0] = -1.9
        # result_hori[mask_empty_1, 1] = -2

    if np.any(zero_quadratic2):
        mask_full_2 = np.logical_and(
            const_term[:, :-1] <= distance_sqr, zero_quadratic2
        )  # (n, m-1)
        result_verti[mask_full_2, 1] = 1
        # mask_empty_2 = const_term[:, :-1] > distance_sqr  # (n, m-1)
        # mask_empty_2 = np.logical_and(mask_empty_2, zero_quadratic2)  # (n, m-1)
        # result_verti[mask_empty_2, 0] = -1.9
        # result_verti[mask_empty_2, 1] = -2

    if np.any(~zero_quadratic1):
        segment1_mat = np.tile(segment1, (m, 1, 1)).transpose(1, 0, 2)  # (n-1, m, d)
        linear_term_1 = np.sum(
            segment1_mat * pp_segment[:-1, :, :], axis=-1
        )  # (n-1, m)
        discriminant_1 = (
            np.square(linear_term_1)
            - quadratic_term1[:, np.newaxis] * const_term[:-1, :]
        )  # (n-1, m)
        mask_neg_1 = np.logical_and(
            discriminant_1 < 0, ~zero_quadratic1[:, np.newaxis]
        )  # (n-1, m)

        # result_hori[mask_neg_1, 0] = -2.9
        # result_hori[mask_neg_1, 1] = -3
        mask_non_neg_1 = ~mask_neg_1  # (n-1, m)
        diff_value1 = np.sqrt(discriminant_1)  # (n-1, m)
        lower_value_1 = (linear_term_1 - diff_value1) / quadratic_term1[
            :, np.newaxis
        ]  # (n-1, m)
        upper_value_1 = (linear_term_1 + diff_value1) / quadratic_term1[
            :, np.newaxis
        ]  # (n-1, m)
        mask_valid_1 = (
            np.logical_and(lower_value_1 <= 1, upper_value_1 >= 0) & mask_non_neg_1
        )  # (n-1, m)

        # mask_not_valid_1 = ~mask_valid_1  # (n-1, m)
        # temp_var_1 = mask_not_valid_1 & mask_non_neg_1
        # result_hori[mask_not_valid_1 & mask_non_neg_1, 0] = -3.9
        # result_hori[mask_not_valid_1 & mask_non_neg_1, 1] = -4
        result_hori[mask_valid_1, 0] = np.maximum(0, lower_value_1[mask_valid_1])
        result_hori[mask_valid_1, 1] = np.minimum(1, upper_value_1[mask_valid_1])

    if np.any(~zero_quadratic2):
        segment2_mat = np.tile(segment2, (n, 1, 1))  # (n, m-1, d)
        linear_term_2 = -np.sum(
            segment2_mat[:, ~zero_quadratic2]
            * pp_segment[:, :-1, :][:, ~zero_quadratic2, :],
            axis=-1,
        )  # (n, m-1)
        discriminant_2 = (
            np.square(linear_term_2)
            - quadratic_term2[:, np.newaxis].T * const_term[:, :-1]
        )  # (n, m-1)
        mask_neg_2 = np.logical_and(discriminant_2 < 0, ~zero_quadratic2)  # (n, m-1)

        # result_verti[mask_neg_2, 0] = -2.9
        # result_verti[mask_neg_2, 1] = -3
        mask_non_neg_2 = ~mask_neg_2  # (n, m-1)
        diff_value2 = np.sqrt(discriminant_2)  # (n, m-1)
        lower_value_2 = (linear_term_2 - diff_value2) / quadratic_term2  # (n, m-1)
        upper_value_2 = (linear_term_2 + diff_value2) / quadratic_term2  # (n, m-1)
        mask_valid_2 = (
            np.logical_and(lower_value_2 <= 1, upper_value_2 >= 0) & mask_non_neg_2
        )  # (n, m-1)

        # mask_not_valid_2 = ~mask_valid_2  # (n, m-1)
        # temp_var_2 = mask_not_valid_2 & mask_non_neg_2
        # result_verti[temp_var_2, 0] = -3.9
        # result_verti[temp_var_2, 1] = -4
        result_verti[mask_valid_2, 0] = np.maximum(0, lower_value_2[mask_valid_2])
        result_verti[mask_valid_2, 1] = np.minimum(1, upper_value_2[mask_valid_2])

    return result_hori.transpose(1, 0, 2), result_verti


#@profile
# @numba.njit
def free_space_intervals_vert(distance_sqr, curve_short, curve_long):
    """faster on large curves
    makes heavy use of numpy vectorization and processing of multiple intervals at once
    main idea is to calculate everything at once and then filter out the results
    """
    n, m = curve_short.shape[0], curve_long.shape[0]

    segment2 = curve_long[1:] - curve_long[:-1]  # (m-1, d)

    quadratic_term2 = np.sum(segment2 * segment2, axis=-1)  # (m-1,)

    zero_quadratic2 = quadratic_term2 == 0  # (m-1,)

    result_verti = np.full(
        (n, m - 1, 2), fill_value=[0, -1], dtype=np.float64
    )  # (n, m - 1, 2)

    pp_segment = curve_long[np.newaxis, :-1] - curve_short[:, np.newaxis]  # (n, m-1, d)
    const_term = np.sum(pp_segment * pp_segment, axis=-1) - distance_sqr  # (n, m)

    if np.any(zero_quadratic2):
        mask_full_2 = np.logical_and(
            const_term <= distance_sqr, zero_quadratic2
        )  # (n, m-1)
        result_verti[mask_full_2, 1] = 1

    segment2_mat = np.tile(segment2, (n, 1, 1))  # (n, m-1, d)
    linear_term_2 = -np.sum(segment2_mat * pp_segment, axis=-1)  # (n, m-1)
    discriminant_2 = (
        np.square(linear_term_2[:, ~zero_quadratic2])
        - quadratic_term2[~zero_quadratic2] * const_term[:, ~zero_quadratic2]
    )  # (n, m-1)

    quadratic_term2 = np.tile(quadratic_term2, (n, 1))  # (n, m-1)

    mask_non_neg_2 = discriminant_2 >= 0  # (n, m-1)
    diff_value2 = np.sqrt(discriminant_2[mask_non_neg_2])  # (n, m-1)

    masked_linear_term_2 = linear_term_2[:, ~zero_quadratic2][
        mask_non_neg_2
    ]  # (n, m-1)
    masked_quadratic_term2 = quadratic_term2[:, ~zero_quadratic2][mask_non_neg_2]

    lower_value_2 = (
        masked_linear_term_2 - diff_value2
    ) / masked_quadratic_term2  # (n, m-1)
    upper_value_2 = (
        masked_linear_term_2 + diff_value2
    ) / masked_quadratic_term2  # (n, m-1)
    mask_valid_2 = np.logical_and(lower_value_2 <= 1, upper_value_2 >= 0)  # (n, m-1)

    clip_lower_value_2 = np.maximum(0, lower_value_2[mask_valid_2])
    clip_upper_value_2 = np.minimum(1, upper_value_2[mask_valid_2])

    result_verti[:, ~zero_quadratic2][mask_non_neg_2][
        mask_valid_2, 0
    ] = clip_lower_value_2
    result_verti[:, ~zero_quadratic2][mask_non_neg_2][
        mask_valid_2, 1
    ] = clip_upper_value_2

    return result_verti
