import numpy as np
import pandas as pd
import pytest

import pylake


def make_data():
    n = 24

    datetime = pd.date_range(
        "2026-07-01",
        periods=n,
        freq="1h"
    )

    irr = np.maximum(
        0,
        np.sin(
            np.linspace(
                -np.pi / 2,
                3 * np.pi / 2,
                n
            )
        )
    ) * 1000

    wtr = np.full(
        n,
        20.0
    )

    do_sat = np.full(
        n,
        9.0
    )

    k_gas = np.zeros(
        n
    )

    z_mix = np.full(
        n,
        5.0
    )

    iota = 2e-5
    rho = -0.001

    delta_do = (
        iota * irr[:-1]
        + rho * np.log(
            wtr[:-1]
        )
    )

    do_obs = np.empty(
        n
    )

    do_obs[0] = 8.0
    do_obs[1:] = (
        do_obs[0]
        + np.cumsum(
            delta_do
        )
    )

    return (
        wtr,
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        datetime,
        iota,
        rho
    )


def test_metab_ols_coefficients():
    (
        wtr,
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        datetime,
        iota,
        rho
    ) = make_data()

    result = pylake.metab_ols(
        wtr,
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        datetime
    )

    assert len(result) == 1

    np.testing.assert_allclose(
        result["iota"].iloc[0],
        iota,
        rtol=1e-6
    )

    np.testing.assert_allclose(
        result["rho"].iloc[0],
        rho,
        rtol=1e-6
    )


def test_metab_ols_nep():
    data = make_data()

    result = pylake.metab_ols(
        *data[:7]
    )

    np.testing.assert_allclose(
        result["NEP"],
        result["GPP"]
        + result["R"]
    )


def test_metab_ols_invalid_depth():
    data = list(
        make_data()[:7]
    )

    data[4][0] = 0

    with pytest.raises(
        ValueError
    ):
        pylake.metab_ols(
            *data
        )


def test_metab_ols_invalid_temperature():
    data = list(
        make_data()[:7]
    )

    data[0][0] = 0

    with pytest.raises(
        ValueError
    ):
        pylake.metab_ols(
            *data
        )


def test_metab_ols_separates_years():
    data = make_data()

    datetime = pd.DatetimeIndex(
        list(data[6])
        + list(
            data[6]
            + pd.DateOffset(
                years=1
            )
        )
    )

    result = pylake.metab_ols(
        np.tile(data[0], 2),
        np.tile(data[1], 2),
        np.tile(data[2], 2),
        np.tile(data[3], 2),
        np.tile(data[4], 2),
        np.tile(data[5], 2),
        datetime
    )

    assert len(result) == 2
