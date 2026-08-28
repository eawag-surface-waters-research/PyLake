import numpy as np

import pylake


def test_k_crusius_constant():
    wind = np.array([
        0.0,
        2.0,
        3.6,
        3.7,
        5.0
    ])

    expected = np.where(
        wind < 3.7,
        1,
        5.14 * wind - 17.9
    ) * 24 / 100

    np.testing.assert_allclose(
        pylake.k_crusius(
            wind,
            method="constant"
        ),
        expected
    )


def test_k_crusius_bilinear():
    wind = np.array([
        0.0,
        2.0,
        3.6,
        3.7,
        5.0
    ])

    expected = np.where(
        wind < 3.7,
        0.72 * wind,
        4.33 * wind - 13.3
    ) * 24 / 100

    np.testing.assert_allclose(
        pylake.k_crusius(
            wind,
            method="bilinear"
        ),
        expected
    )


def test_k_crusius_power():
    wind = np.array([
        0.0,
        2.0,
        5.0,
        10.0
    ])

    expected = (
        0.228 * wind**2.2
        + 0.168
    ) * 24 / 100

    np.testing.assert_allclose(
        pylake.k_crusius(
            wind,
            method="power"
        ),
        expected
    )


def test_k_crusius_invalid_method():
    try:
        pylake.k_crusius(
            5,
            method="banana"
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
