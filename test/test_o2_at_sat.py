import numpy as np
import pytest

import pylake


def test_o2_at_sat_temperature():
    temp = np.array([
        5.0,
        10.0,
        20.0,
        25.0
    ])

    result = pylake.o2_at_sat(
        temp
    )

    assert np.all(
        np.isfinite(result)
    )

    assert np.all(
        np.diff(result) < 0
    )


def test_o2_at_sat_salinity():
    fresh = pylake.o2_at_sat(
        20.0,
        salinity=0
    )

    saline = pylake.o2_at_sat(
        20.0,
        salinity=20
    )

    assert saline < fresh


def test_o2_at_sat_altitude():
    sea = pylake.o2_at_sat(
        20.0,
        altitude=0
    )

    high = pylake.o2_at_sat(
        20.0,
        altitude=1500
    )

    assert high < sea


def test_o2_at_sat_models():
    for model in [
        "garcia",
        "garcia-benson",
        "weiss",
        "benson"
    ]:
        result = pylake.o2_at_sat(
            20.0,
            model=model
        )

        assert np.isfinite(
            result
        )

        assert result > 0


def test_o2_at_sat_vector_salinity():
    temp = np.array([
        10.0,
        15.0,
        20.0
    ])

    salinity = np.array([
        0.0,
        5.0,
        10.0
    ])

    result = pylake.o2_at_sat(
        temp,
        salinity=salinity
    )

    assert result.shape == temp.shape


def test_o2_at_sat_invalid_model():
    with pytest.raises(
        ValueError
    ):
        pylake.o2_at_sat(
            20,
            model="banana"
        )
