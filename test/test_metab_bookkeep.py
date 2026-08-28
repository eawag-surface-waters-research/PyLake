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

    irr = np.zeros(n)
    irr[6:18] = 1

    do_obs = np.empty(n)
    do_obs[0] = 8

    changes = np.where(
        irr[:-1] > 0,
        0.02,
        -0.01
    )

    do_obs[1:] = (
        do_obs[0]
        + np.cumsum(changes)
    )

    do_sat = np.full(n, 9.0)
    k_gas = np.zeros(n)
    z_mix = np.full(n, 5.0)

    return (
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        datetime
    )


def test_metab_bookkeep():
    data = make_data()

    result = pylake.metab_bookkeep(
        *data
    )

    assert len(result) == 1
    assert result["GPP"].iloc[0] > 0
    assert result["R"].iloc[0] < 0


def test_metab_bookkeep_nep():
    data = make_data()

    result = pylake.metab_bookkeep(
        *data
    )

    expected = (
        np.mean(
            np.diff(data[0])
        )
        * 24
    )

    np.testing.assert_allclose(
        result["NEP"].iloc[0],
        expected
    )


def test_metab_bookkeep_gas_flux():
    data = list(
        make_data()
    )

    no_gas = pylake.metab_bookkeep(
        *data
    )

    data[2] = np.full(
        24,
        0.5
    )

    with_gas = pylake.metab_bookkeep(
        *data
    )

    assert not np.isclose(
        no_gas["NEP"].iloc[0],
        with_gas["NEP"].iloc[0]
    )


def test_metab_bookkeep_invalid_depth():
    data = list(
        make_data()
    )

    data[3][0] = 0

    with pytest.raises(
        ValueError
    ):
        pylake.metab_bookkeep(
            *data
        )


def test_metab_bookkeep_separates_years():
    data = make_data()

    datetime = pd.DatetimeIndex(
        list(data[5])
        + list(
            data[5]
            + pd.DateOffset(years=1)
        )
    )

    result = pylake.metab_bookkeep(
        np.tile(data[0], 2),
        np.tile(data[1], 2),
        np.tile(data[2], 2),
        np.tile(data[3], 2),
        np.tile(data[4], 2),
        datetime
    )

    assert len(result) == 2
