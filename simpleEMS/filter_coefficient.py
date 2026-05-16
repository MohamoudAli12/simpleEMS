import math

# Bessel filter g-values (orders 2-19)
# Index: [order - 2][element]
BESSEL_COEFFICIENT = [
    # Order 2
    [
        0.57550275,
        2.1478055,
    ],
    # Order 3
    [
        0.33742149,
        0.97051182,
        2.2034114,
    ],
    # Order 4
    [
        0.2334158,
        0.67252481,
        1.0815161,
        2.2403786,
    ],
    # Order 5
    [
        0.17431938,
        0.50724063,
        0.80401117,
        1.1110332,
        2.2582171,
    ],
    # Order 6
    [
        0.13649238,
        0.40018984,
        0.6391554,
        0.85378587,
        1.112643,
        2.2645236,
    ],
    # Order 7
    [
        0.11056245,
        0.32588813,
        0.52489273,
        0.70200915,
        0.86902684,
        1.1051644,
        2.2659006,
    ],
    # Order 8
    [
        0.091905558,
        0.27191069,
        0.44092213,
        0.59357268,
        0.73025665,
        0.86950037,
        1.0955593,
        2.2656071,
    ],
    # Order 9
    [
        0.077965506,
        0.23129119,
        0.37698651,
        0.5107787,
        0.63059516,
        0.74073299,
        0.86387345,
        1.0862838,
        2.2648789,
    ],
    # Order 10
    [
        0.067229245,
        0.19984023,
        0.32699699,
        0.44543381,
        0.55281473,
        0.64933545,
        0.74201735,
        0.85607238,
        1.0780948,
        2.2641262,
    ],
    # Order 11
    [
        0.058753264,
        0.1749102,
        0.28706331,
        0.39272513,
        0.48983467,
        0.57742576,
        0.6573719,
        0.73864992,
        0.84789472,
        1.0711184,
        2.2634675,
    ],
    # Order 12
    [
        0.051923,
        0.15475798,
        0.25458412,
        0.34949045,
        0.43777572,
        0.51824604,
        0.59097358,
        0.65913453,
        0.73309605,
        0.84013853,
        1.0652583,
        2.2629233,
    ],
    # Order 13
    [
        0.046323098,
        0.13819538,
        0.22775963,
        0.31352943,
        0.39412779,
        0.46842754,
        0.53589822,
        0.59744624,
        0.65727966,
        0.72670805,
        0.83311934,
        1.0603538,
        2.2624829,
    ],
    # Order 14
    [
        0.041663913,
        0.12438811,
        0.20530976,
        0.28325704,
        0.35711858,
        0.42591257,
        0.48895292,
        0.54624778,
        0.59939971,
        0.6534283,
        0.72022065,
        0.82691896,
        1.0562406,
        2.262128,
    ],
    # Order 15
    [
        0.037738195,
        0.11273464,
        0.186303,
        0.25750279,
        0.32543754,
        0.38927393,
        0.44833655,
        0.5022966,
        0.55162099,
        0.59850238,
        0.64857809,
        0.71401822,
        0.8215094,
        1.0527729,
        2.2618408,
    ],
    # Order 16
    [
        0.03439122,
        0.102799,
        0.1700389,
        0.23539257,
        0.29808411,
        0.35745153,
        0.41285998,
        0.4638916,
        0.51051757,
        0.55360981,
        0.59585789,
        0.64334149,
        0.70828423,
        0.81681581,
        1.0498286,
        2.2616066,
    ],
    # Order 17
    [
        0.031523496,
        0.094214373,
        0.1560301,
        0.21622781,
        0.27429733,
        0.32962199,
        0.38165374,
        0.42998671,
        0.47437364,
        0.51507678,
        0.55331904,
        0.59220181,
        0.63808843,
        0.70308744,
        0.81274832,
        1.0473087,
        2.2614134,
    ],
    # Order 18
    [
        0.028450351,
        0.088289337,
        0.14199897,
        0.20112875,
        0.25237658,
        0.30571616,
        0.35380272,
        0.39992872,
        0.44218287,
        0.48110224,
        0.5170226,
        0.55152405,
        0.58802463,
        0.63303427,
        0.69843256,
        0.80921764,
        1.0451338,
        2.2612522,
    ],
    # Order 19
    [
        0.027944243,
        0.077389235,
        0.13716781,
        0.18087554,
        0.23831545,
        0.28142035,
        0.33055916,
        0.37249664,
        0.41348365,
        0.45066049,
        0.48503552,
        0.51711998,
        0.54877223,
        0.58365096,
        0.62829555,
        0.69429034,
        0.8061422,
        1.043241,
        2.2611162,
    ],
]


def bessel_value(index: int, order: int) -> float:
    if order < 2 or order > 19:
        raise ValueError(f"Bessel filter order {order} is not supported ")
    if index < 0 or index >= order:
        return 1.0
    return BESSEL_COEFFICIENT[order - 2][index]


def butterworth_value(index: int, order: int) -> float:
    if index < 0 or index >= order:
        return 1.0
    return 2.0 * math.sin((2.0 * index + 1.0) / (2.0 * order) * math.pi)


def chebyshev_value(index: int, order: int, ripple_db: float) -> float:
    if index < 0 or index >= order:
        return 1.0
    if order % 2 == 0:
        raise ValueError(
            "Even order Chebyshev cannot be realized with passive filters "
            "(source and load terminations differ)"
        )
    eps = math.sqrt(10.0 ** (ripple_db / 10.0) - 1.0)
    gamma = math.sinh(math.asinh(1.0 / eps) / order)
    a_prev = math.sin(0.5 / order * math.pi)
    gk = a_prev / gamma
    for i in range(1, index + 1):
        ak = a_prev
        a_curr = math.sin((2.0 * i + 1.0) / (2.0 * order) * math.pi)
        b = math.sin(i * math.pi / order)
        gk *= gamma * gamma + b * b
        gk = ak * a_curr / gk
        a_prev = a_curr
    return 2.0 * gk


def get_filter_coefficient(
    index: int,
    filter_response: str,
    filter_order: int,
    ripple_db: float | None,
) -> float:
    if ripple_db is None:
        ripple_db = 3

    t = filter_response.lower()
    if t == "bessel":
        return bessel_value(index, filter_order)
    elif t == "butterworth":
        return butterworth_value(index, filter_order)
    elif t == "chebyshev":
        return chebyshev_value(index, filter_order, ripple_db)
    else:
        raise ValueError(f"Unsupported filter type: {filter_response}")
