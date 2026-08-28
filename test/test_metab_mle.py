import numpy as np
import pytest

import pylake


def make_data():
    n = 24

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

    k_gas = np.full(
        n,
        0.5
    )

    z_mix = np.full(
        n,
        5.0
    )

    do_obs = np.linspace(
        8.0,
        8.4,
        n
    )

    return (
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        wtr
    )


def test_metab_mle_pe_runs():
    data = make_data()

    result = pylake.metab_mle(
        *data,
        freq=24,
        error_type="PE"
    )

    assert "params" in result
    assert "metab" in result

    assert np.isfinite(
        result["metab"]["GPP"]
    )

    assert np.isfinite(
        result["metab"]["R"]
    )

    assert np.isfinite(
        result["metab"]["NEP"]
    )


def test_metab_mle_nep():
    data = make_data()

    result = pylake.metab_mle(
        *data,
        freq=24,
        error_type="PE"
    )

    np.testing.assert_allclose(
        result["metab"]["NEP"],
        result["metab"]["GPP"]
        + result["metab"]["R"]
    )


def test_metab_mle_zero_gas():
    data = list(
        make_data()
    )

    data[2] = np.zeros(
        24
    )

    result = pylake.metab_mle(
        *data,
        freq=24,
        error_type="PE"
    )

    assert np.isfinite(
        result["params"]["nll"]
    )


def test_metab_mle_invalid_depth():
    data = list(
        make_data()
    )

    data[3][0] = 0

    with pytest.raises(
        ValueError
    ):
        pylake.metab_mle(
            *data,
            freq=24,
            error_type="PE"
        )


def test_metab_mle_invalid_temperature():
    data = list(
        make_data()
    )

    data[5][0] = 0

    with pytest.raises(
        ValueError
    ):
        pylake.metab_mle(
            *data,
            freq=24,
            error_type="PE"
        )


def test_metab_mle_oe_not_yet_ported():
    data = make_data()

    with pytest.raises(
        NotImplementedError
    ):
        pylake.metab_mle(
            *data,
            freq=24,
            error_type="OE"
        )
