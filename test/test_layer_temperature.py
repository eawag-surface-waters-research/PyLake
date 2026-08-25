import numpy as np

import pylake


Temp = np.array([
    25.2, 25.1, 24.1, 22.0,
    19.8, 15.3, 12.0, 11.1
])

depth = np.arange(8)

bthA = np.array([
    10000, 8900, 5000, 3500,
    2000, 1000, 300, 10
])

bthD = np.arange(8)


def test_layer_temperature():
    result = pylake.layer_temperature(
        2,
        6,
        Temp,
        depth,
        bthA,
        bthD
    )

    assert np.isfinite(result)
    assert 12 < result < 24.1


def test_whole_lake_temperature():
    result = pylake.whole_lake_temperature(
        Temp,
        depth,
        bthA,
        bthD
    )

    assert np.isfinite(result)
    assert Temp.min() <= result <= Temp.max()


def test_epi_hypo_temperature():
    epi = pylake.epi_temperature(
        Temp,
        depth,
        bthA,
        bthD,
        meta_top=2
    )

    hypo = pylake.hypo_temperature(
        Temp,
        depth,
        bthA,
        bthD,
        meta_bottom=4
    )

    assert epi > hypo
