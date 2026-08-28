import pandas as pd 
import numpy as np 
from .functions import *
from .functions_meta import *

def metab_bookkeep(
    do_obs,
    do_sat,
    k_gas,
    z_mix,
    irr,
    Datetime
):
    """Estimate daily lake metabolism by bookkeeping oxygen changes.

    The method removes air-water gas exchange from consecutive dissolved
    oxygen differences. Night-time changes estimate respiration, while the
    difference between mean daytime and night-time changes estimates gross
    primary production.

    Method
    ----------
    Missing observations are removed jointly from all input series and data are
    grouped by calendar day. Atmospheric exchange is estimated from oxygen
    undersaturation, gas-transfer velocity, and mixed-layer depth. The corrected
    night-time oxygen change estimates respiration. Gross primary production is
    inferred from the difference between daytime and night-time metabolic
    changes, and net ecosystem production is the daily corrected oxygen change.

    Parameters
    ----------
    do_obs, do_sat : array_like
        Observed and saturation dissolved oxygen concentrations.
    k_gas : array_like
        Gas-transfer velocity, expressed per day.
    z_mix : array_like
        Mixed-layer depth in metres. Values must be strictly positive.
    irr : array_like
        Irradiance. Values greater than zero identify daytime observations.
    Datetime : array_like
        Observation timestamps accepted by :func:`pandas.to_datetime`.

    Returns
    -------
    pandas.DataFrame
        One row per usable day with ``time``, ``GPP``, ``R``, and ``NEP``.
        Days with fewer than two valid observations are omitted.

    Raises
    ------
    ValueError
        If a retained mixed-layer depth is zero or negative.

    Examples
    --------
    >>> time = pd.date_range("2020-07-01", periods=4, freq="6h")
    >>> metab_bookkeep([8, 8.1, 8.2, 8.0], [9] * 4, [0.5] * 4,
    ...                [2] * 4, [0, 100, 50, 0], time).columns.tolist()
    ['time', 'GPP', 'R', 'NEP']
    """
    do_obs = np.asarray(
        do_obs,
        dtype=float
    )

    do_sat = np.asarray(
        do_sat,
        dtype=float
    )

    k_gas = np.asarray(
        k_gas,
        dtype=float
    )

    z_mix = np.asarray(
        z_mix,
        dtype=float
    )

    irr = np.asarray(
        irr,
        dtype=float
    )

    Datetime = pd.to_datetime(
        Datetime
    )

    valid = (
        np.isfinite(do_obs)
        & np.isfinite(do_sat)
        & np.isfinite(k_gas)
        & np.isfinite(z_mix)
        & np.isfinite(irr)
    )

    do_obs = do_obs[valid]
    do_sat = do_sat[valid]
    k_gas = k_gas[valid]
    z_mix = z_mix[valid]
    irr = irr[valid]
    Datetime = Datetime[valid]

    if np.any(
        z_mix <= 0
    ):
        raise ValueError(
            "z_mix must be greater than zero"
        )

    frame = pd.DataFrame({
        "datetime": Datetime,
        "do_obs": do_obs,
        "do_sat": do_sat,
        "k_gas": k_gas,
        "z_mix": z_mix,
        "irr": irr
    })

    frame["date"] = (
        frame["datetime"]
        .dt.date
    )

    results = []

    for date, group in frame.groupby(
        "date",
        sort=True
    ):
        if len(group) < 2:
            continue

        time = group[
            "datetime"
        ]

        delta_time = (
            time.diff()
            .dt.total_seconds()
            .dropna()
        )

        if len(delta_time) == 0:
            continue

        timestep = np.median(
            delta_time
        )

        freq = (
            86400
            / timestep
        )

        obs = group[
            "do_obs"
        ].to_numpy()

        sat = group[
            "do_sat"
        ].to_numpy()

        gas = group[
            "k_gas"
        ].to_numpy()

        mix = group[
            "z_mix"
        ].to_numpy()

        light = group[
            "irr"
        ].to_numpy()

        delta_do = np.diff(
            obs
        )

        gas_flux = (
            sat
            - obs
        ) * (
            gas
            / freq
        ) / mix

        delta_do_metab = (
            delta_do
            - gas_flux[:-1]
        )

        day = (
            light[:-1]
            > 0
        )

        night = (
            light[:-1]
            <= 0
        )

        if not np.any(day):
            gpp = np.nan
        elif not np.any(night):
            gpp = np.nan
        else:
            nep_day = np.mean(
                delta_do_metab[day]
            )

            nep_night = np.mean(
                delta_do_metab[night]
            )

            gpp = (
                nep_day
                - nep_night
            ) * np.sum(day)

        if np.any(night):
            respiration = (
                np.mean(
                    delta_do_metab[night]
                )
                * freq
            )
        else:
            respiration = np.nan

        nep = (
            np.mean(
                delta_do_metab
            )
            * freq
        )

        results.append({
            "time": str(date),
            "GPP": gpp,
            "R": respiration,
            "NEP": nep
        })

    return pd.DataFrame(
        results
    )


def metab_ols(
    wtr,
    do_obs,
    do_sat,
    k_gas,
    z_mix,
    irr,
    Datetime
):
    """Estimate daily metabolism with an ordinary least-squares model.

    Oxygen changes corrected for atmospheric exchange are regressed on
    irradiance and the logarithm of water temperature. The fitted coefficients
    are converted to daily gross primary production and respiration.

    Method
    ----------
    After jointly removing missing observations, the data are separated by day.
    Oxygen change caused by atmospheric exchange is removed. The remaining
    change is fitted without an intercept to irradiance and log water
    temperature using ordinary least squares. Coefficients are scaled by the
    inferred number of observations per day.

    Parameters
    ----------
    wtr : array_like
        Water temperature in degrees Celsius. Retained values must be positive.
    do_obs, do_sat : array_like
        Observed and saturation dissolved oxygen concentrations.
    k_gas : array_like
        Gas-transfer velocity, expressed per day.
    z_mix : array_like
        Mixed-layer depth in metres. Values must be strictly positive.
    irr : array_like
        Irradiance used as the production predictor.
    Datetime : array_like
        Observation timestamps accepted by :func:`pandas.to_datetime`.

    Returns
    -------
    pandas.DataFrame
        Daily ``GPP``, ``R``, ``NEP`` and fitted ``iota`` and ``rho``
        coefficients. Incomplete days or underdetermined fits are omitted.

    Raises
    ------
    ValueError
        If retained temperatures or mixed-layer depths are not positive.
    """
    wtr = np.asarray(
        wtr,
        dtype=float
    )

    do_obs = np.asarray(
        do_obs,
        dtype=float
    )

    do_sat = np.asarray(
        do_sat,
        dtype=float
    )

    k_gas = np.asarray(
        k_gas,
        dtype=float
    )

    z_mix = np.asarray(
        z_mix,
        dtype=float
    )

    irr = np.asarray(
        irr,
        dtype=float
    )

    Datetime = pd.to_datetime(
        Datetime
    )

    valid = (
        np.isfinite(wtr)
        & np.isfinite(do_obs)
        & np.isfinite(do_sat)
        & np.isfinite(k_gas)
        & np.isfinite(z_mix)
        & np.isfinite(irr)
    )

    wtr = wtr[valid]
    do_obs = do_obs[valid]
    do_sat = do_sat[valid]
    k_gas = k_gas[valid]
    z_mix = z_mix[valid]
    irr = irr[valid]
    Datetime = Datetime[valid]

    if np.any(
        z_mix <= 0
    ):
        raise ValueError(
            "z_mix must be greater than zero"
        )

    if np.any(
        wtr <= 0
    ):
        raise ValueError(
            "all wtr must be positive"
        )

    frame = pd.DataFrame({
        "datetime": Datetime,
        "wtr": wtr,
        "do_obs": do_obs,
        "do_sat": do_sat,
        "k_gas": k_gas,
        "z_mix": z_mix,
        "irr": irr
    })

    frame["date"] = (
        frame["datetime"]
        .dt.date
    )

    results = []

    for date, group in frame.groupby(
        "date",
        sort=True
    ):
        if len(group) < 2:
            continue

        time = group[
            "datetime"
        ]

        delta = (
            time.diff()
            .dt.total_seconds()
            .dropna()
        )

        if len(delta) == 0:
            continue

        dt = np.median(
            delta
        )

        freq = (
            86400
            / dt
        )

        obs = group[
            "do_obs"
        ].to_numpy()

        sat = group[
            "do_sat"
        ].to_numpy()

        gas = group[
            "k_gas"
        ].to_numpy()

        mix = group[
            "z_mix"
        ].to_numpy()

        light = group[
            "irr"
        ].to_numpy()

        temp = group[
            "wtr"
        ].to_numpy()

        do_diff = np.diff(
            obs
        )

        inst_flux = (
            gas
            / freq
        ) * (
            sat
            - obs
        )

        flux = inst_flux[:-1]

        noflux_do_diff = (
            do_diff
            - flux
            / mix[:-1]
        )

        lntemp = np.log(
            temp
        )

        X = np.column_stack([
            light[:-1],
            lntemp[:-1]
        ])

        valid_fit = (
            np.isfinite(
                noflux_do_diff
            )
            & np.all(
                np.isfinite(X),
                axis=1
            )
        )

        X = X[
            valid_fit
        ]

        y = noflux_do_diff[
            valid_fit
        ]

        if len(y) < 2:
            continue

        coeffs = np.linalg.lstsq(
            X,
            y,
            rcond=None
        )[0]

        iota = coeffs[0]
        rho = coeffs[1]

        gpp = (
            np.mean(
                iota * X[:, 0]
            )
            * freq
        )

        resp = (
            np.mean(
                rho * X[:, 1]
            )
            * freq
        )

        results.append({
            "date": date,
            "GPP": gpp,
            "R": resp,
            "NEP": gpp + resp,
            "iota": iota,
            "rho": rho
        })

    return pd.DataFrame(
        results
    )


def _mle_predict_pe(
    do_obs,
    do_sat,
    k_gas,
    z_mix,
    irr,
    wtr,
    c1,
    c2
):
    """Predict an oxygen series for the process-error metabolism model.

    Gas exchange is integrated analytically over each equally spaced step.

    Parameters
    ----------
    do_obs, do_sat : numpy.ndarray
        Observed and saturation dissolved oxygen concentrations.
    k_gas : numpy.ndarray
        Gas-transfer velocity per model time step.
    z_mix : numpy.ndarray
        Mixed-layer depth in metres.
    irr : numpy.ndarray
        Irradiance observations.
    wtr : numpy.ndarray
        Positive water temperatures in degrees Celsius.
    c1, c2 : float
        Production and respiration coefficients.

    Returns
    -------
    numpy.ndarray
        Predicted oxygen concentration with the same shape as ``do_obs``.
    """
    kz = k_gas / z_mix
    beta = np.exp(-kz)

    alpha = np.zeros_like(
        do_obs,
        dtype=float
    )

    alpha[0] = do_obs[0]

    for i in range(
        1,
        len(do_obs)
    ):
        production = (
            c1 * irr[i - 1]
            + c2 * np.log(
                wtr[i - 1]
            )
        )

        if np.isclose(
            kz[i - 1],
            0
        ):
            alpha[i] = (
                alpha[i - 1]
                + production
            )

        else:
            source = (
                production
                + kz[i - 1]
                * do_sat[i - 1]
            )

            alpha[i] = (
                source
                / kz[i - 1]
                * (
                    1
                    - beta[i - 1]
                )
                + beta[i - 1]
                * alpha[i - 1]
            )

    return alpha


def _mle_nll_pe(
    params,
    do_obs,
    do_sat,
    k_gas,
    z_mix,
    irr,
    wtr
):
    """Return the Gaussian negative log-likelihood of the PE model.

    Parameters
    ----------
    params : array_like
        Production coefficient, respiration coefficient, and log
        process-error variance, in that order.
    do_obs, do_sat, k_gas, z_mix, irr, wtr : numpy.ndarray
        Model inputs passed to :func:`_mle_predict_pe`.

    Returns
    -------
    float
        Gaussian negative log-likelihood. Log variance is bounded to keep the
        objective finite for exact or nearly exact fits.
    """
    c1 = params[0]
    c2 = params[1]
    log_variance = np.clip(
        params[2],
        np.log(1e-12),
        np.log(1e6)
    )

    variance = np.exp(
        log_variance
    )

    alpha = _mle_predict_pe(
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        wtr,
        c1,
        c2
    )

    residual = (
        do_obs
        - alpha
    )

    return 0.5 * np.sum(
        np.log(2 * np.pi)
        + log_variance
        + residual**2
        / variance
    )


def metab_mle(
    do_obs,
    do_sat,
    k_gas,
    z_mix,
    irr,
    wtr,
    freq,
    error_type="PE"
):
    """Estimate whole-lake metabolism by maximum likelihood.

    The implemented process-error (``PE``) model combines irradiance-driven
    production, log-temperature respiration, and air-water gas exchange.

    Method
    ----------
    The oxygen state is propagated through time with an analytical gas-exchange
    term and linear production and respiration components. Production is
    proportional to irradiance, while respiration is proportional to the
    logarithm of water temperature. The two coefficients and process-error
    variance are fitted by minimizing a Gaussian negative log-likelihood. The
    variance is bounded away from zero to keep exact fits numerically finite.

    Parameters
    ----------
    do_obs, do_sat : array_like
        Observed and saturation dissolved oxygen concentrations.
    k_gas : array_like
        Gas-transfer velocity in units compatible with ``freq``.
    z_mix : array_like
        Positive mixed-layer depths in metres.
    irr : array_like
        Irradiance observations.
    wtr : array_like
        Positive water temperatures in degrees Celsius.
    freq : float
        Number of observations per day.
    error_type : {"PE"}, default: "PE"
        Error formulation. Observation error is not yet implemented.

    Returns
    -------
    dict
        ``params`` contains fitted coefficients, variance, and negative
        log-likelihood; ``metab`` contains daily ``GPP``, ``R``, and ``NEP``.

    Raises
    ------
    ValueError
        If input lengths differ, values are missing, or required values are
        not positive.
    NotImplementedError
        If an error model other than ``PE`` is requested.

    Examples
    ----------
    >>> n = 6
    >>> result = metab_mle([8, 8.1, 8.2, 8.3, 8.2, 8.1], [9] * n,
    ...                    [0.5] * n, [5] * n, [0, 100, 200, 100, 0, 0],
    ...                    [20] * n, freq=6)
    >>> sorted(result)
    ['metab', 'params']
    """
    from scipy.optimize import minimize

    do_obs = np.asarray(
        do_obs,
        dtype=float
    )

    do_sat = np.asarray(
        do_sat,
        dtype=float
    )

    k_gas = np.asarray(
        k_gas,
        dtype=float
    )

    z_mix = np.asarray(
        z_mix,
        dtype=float
    )

    irr = np.asarray(
        irr,
        dtype=float
    )

    wtr = np.asarray(
        wtr,
        dtype=float
    )

    arrays = (
        do_obs,
        do_sat,
        k_gas,
        z_mix,
        irr,
        wtr
    )

    lengths = [
        len(array)
        for array in arrays
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            "all inputs must have the same length"
        )

    if np.any(
        ~np.isfinite(
            np.column_stack(
                arrays
            )
        )
    ):
        raise ValueError(
            "metab_mle does not support missing values"
        )

    if np.any(
        z_mix <= 0
    ):
        raise ValueError(
            "z_mix must be greater than zero"
        )

    if np.any(
        wtr <= 0
    ):
        raise ValueError(
            "all wtr must be positive"
        )

    if error_type.upper() != "PE":
        raise NotImplementedError(
            "OE model is not ported yet"
        )

    diff = np.diff(
        do_obs
    )

    q0 = np.mean(
        (
            diff
            - np.mean(diff)
        )**2
    )

    if not np.isfinite(q0) or q0 <= 0:
        q0 = 1e-8

    guesses = np.array([
        1e-4,
        1e-4,
        np.log(q0)
    ])

    fit = minimize(
        _mle_nll_pe,
        guesses,
        args=(
            do_obs,
            do_sat,
            k_gas / freq,
            z_mix,
            irr,
            wtr
        ),
        bounds=(
            (None, None),
            (None, None),
            (np.log(1e-12), np.log(1e6))
        )
    )

    c1 = fit.x[0]
    c2 = fit.x[1]
    variance = np.exp(
        fit.x[2]
    )

    gpp = (
        np.mean(
            c1 * irr
        )
        * freq
    )

    respiration = (
        np.mean(
            c2
            * np.log(
                wtr
            )
        )
        * freq
    )

    return {
        "params": {
            "gppCoeff": c1,
            "rCoeff": c2,
            "Q": variance,
            "nll": fit.fun
        },
        "metab": {
            "GPP": gpp,
            "R": respiration,
            "NEP": (
                gpp
                + respiration
            )
        }
    }


def o2_at_sat(
    temp,
    baro=None,
    altitude=0,
    salinity=0,
    model="garcia-benson"
):
    """Calculate dissolved oxygen concentration at atmospheric saturation.

    Method
    ----------
    Oxygen solubility is calculated from the selected empirical temperature and
    salinity relationship. When barometric pressure is not supplied, a standard
    atmosphere model estimates it from altitude. The result is then corrected
    for water-vapour pressure and converted to milligrams per litre.

    Parameters
    ----------
    temp : array_like
        Water temperature in degrees Celsius.
    baro : array_like or None, default: None
        Barometric pressure in millibars. When omitted, pressure is estimated
        from ``altitude`` using a standard-atmosphere approximation.
    altitude : float, default: 0
        Elevation above sea level in metres.
    salinity : array_like, default: 0
        Salinity in practical salinity units.
    model : {"garcia-benson", "garcia", "weiss", "benson"}
        Oxygen-solubility equation.

    Returns
    -------
    numpy.ndarray
        Saturation dissolved oxygen concentration in milligrams per litre.

    Raises
    ------
    ValueError
        If ``model`` is not recognized.

    Notes
    -----
    The result includes water-vapour and atmospheric-pressure corrections.

    Examples
    ----------
    >>> round(float(o2_at_sat(20)), 2)
    9.09
    """
    temp = np.asarray(
        temp,
        dtype=float
    )

    salinity = np.asarray(
        salinity,
        dtype=float
    )

    if salinity.ndim == 0:
        salinity = np.full_like(
            temp,
            salinity
        )

    mgL_mlL = 1.42905
    mmHg_mb = 0.750061683

    if baro is None:
        mmHg_inHg = 25.3970886
        standard_pressure_sea_level = 29.92126
        standard_temperature_sea_level = 288.15
        gravitational_acceleration = 9.80665
        air_molar_mass = 0.0289644
        universal_gas_constant = 8.31447

        baro = (
            (1 / mmHg_mb)
            * mmHg_inHg
            * standard_pressure_sea_level
            * np.exp(
                (
                    -gravitational_acceleration
                    * air_molar_mass
                    * altitude
                )
                / (
                    universal_gas_constant
                    * standard_temperature_sea_level
                )
            )
        )

    baro = np.asarray(
        baro,
        dtype=float
    )

    u = 10 ** (
        8.10765
        - 1750.286
        / (235 + temp)
    )

    press_corr = (
        baro * mmHg_mb - u
    ) / (
        760 - u
    )

    model = model.lower()

    if model == "garcia":
        Ts = np.log(
            (298.15 - temp)
            / (273.15 + temp)
        )

        lnC = (
            2.00856
            + 3.22400 * Ts
            + 3.99063 * Ts**2
            + 4.80299 * Ts**3
            + 9.78188e-1 * Ts**4
            + 1.71069 * Ts**5
            - salinity * (
                6.24097e-3
                + 6.93498e-3 * Ts
                + 6.90358e-3 * Ts**2
                + 4.29155e-3 * Ts**3
            )
            - 3.1168e-7 * salinity**2
        )

        o2_sat = np.exp(
            lnC
        )

    elif model == "garcia-benson":
        Ts = np.log(
            (298.15 - temp)
            / (273.15 + temp)
        )

        lnC = (
            2.00907
            + 3.22014 * Ts
            + 4.05010 * Ts**2
            + 4.94457 * Ts**3
            - 2.56847e-1 * Ts**4
            + 3.88767 * Ts**5
            - salinity * (
                6.24523e-3
                + 7.37614e-3 * Ts
                + 1.03410e-2 * Ts**2
                + 8.17083e-3 * Ts**3
            )
            - 4.88682e-7 * salinity**2
        )

        o2_sat = np.exp(
            lnC
        )

    elif model == "weiss":
        tempk = temp + 273.15

        lnC = (
            -173.4292
            + 249.6339 * (
                100 / tempk
            )
            + 143.3483 * np.log(
                tempk / 100
            )
            - 21.8492 * (
                tempk / 100
            )
            + salinity * (
                -0.033096
                + 0.014259 * (
                    tempk / 100
                )
                - 0.0017000 * (
                    tempk / 100
                )**2
            )
        )

        o2_sat = np.exp(
            lnC
        )

    elif model == "benson":
        if np.any(
            salinity != 0
        ):
            import warnings

            warnings.warn(
                "Benson model does not currently include salinity"
            )

        o2_sat = (
            -0.00006 * temp**3
            + 0.00725 * temp**2
            - 0.39571 * temp
            + 14.59030
        )

        o2_sat = (
            o2_sat
            / mgL_mlL
        )

    else:
        raise ValueError(
            f"unrecognized model: {model}"
        )

    return (
        o2_sat
        * mgL_mlL
        * press_corr
    )


def k_cole(wnd):
    """Estimate ``k600`` from wind speed using the Cole relationship.

    Method
    ----------
    The empirical Cole and Caraco wind relationship is evaluated in centimetres
    per hour and converted to metres per day.

    Parameters
    ----------
    wnd : array_like
        Wind speed in metres per second.

    Returns
    -------
    array_like
        Standardized gas-transfer velocity in metres per day.
    """
    k600 = 2.07 + (0.215*(wnd**(1.7)))
    k600 = k600*24/100 #units in m d-1
    return k600

def k_crusius(wnd, method="power"):
    """Estimate the standardized gas-transfer velocity from wind speed.

    Method
    ----------
    Three empirical relationships are available. ``constant`` assumes weak,
    constant exchange below 3.7 m s-1; ``bilinear`` uses separate linear
    relationships below and above that threshold; and ``power`` applies one
    continuous power law. All relationships are converted from centimetres per
    hour to metres per day.

    Parameters
    ----------
    wnd : array_like
        Wind speed at 10 m height in metres per second.
    method : {"power", "constant", "bilinear"}, default: "power"
        Crusius and Wanninkhof parameterization.

    Returns
    -------
    numpy.ndarray
        ``k600`` in metres per day.

    Raises
    ------
    ValueError
        If ``method`` is unsupported.

    Examples
    ----------
    >>> round(float(k_crusius(5)), 3)
    1.928
    """
    wnd = np.asarray(
        wnd,
        dtype=float
    )

    method = method.lower()

    if method == "constant":
        k600 = np.where(
            wnd < 3.7,
            1,
            5.14 * wnd - 17.9
        )

    elif method == "bilinear":
        k600 = np.where(
            wnd < 3.7,
            0.72 * wnd,
            4.33 * wnd - 13.3
        )

    elif method == "power":
        k600 = (
            0.228 * wnd**2.2
            + 0.168
        )

    else:
        raise ValueError(
            "method must be one of: power, constant, bilinear"
        )

    return k600 * 24 / 100


def k_read(wnd_z, Qsen, Qlat, Cd, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet, lwnet_mode=1, s=0.2):
    """Estimate ``k600`` from wind, heat flux, and near-surface turbulence.

    Method
    ----------
    The surface energy budget is used to calculate buoyancy production, while
    wind stress supplies shear-driven turbulence. Water density, thermal
    expansion, and kinematic viscosity are evaluated from surface temperature.
    The resulting dissipation rate is converted to a Schmidt-number-600 gas
    transfer velocity following the surface-renewal formulation used by Read et
    al. and LakeMetabolizer.

    Parameters
    ----------
    wnd_z : float
        Wind measurement height in metres.
    Qsen, Qlat : array_like
        Sensible and latent heat fluxes in watts per square metre.
    Cd : array_like
        Momentum drag coefficient.
    Kd : array_like
        Light attenuation coefficient in inverse metres.
    lat : float
        Latitude in decimal degrees. Retained for API compatibility.
    A0 : float
        Lake surface area in square metres. Retained for API compatibility.
    air_press : array_like
        Atmospheric pressure in millibars.
    dateTime : array_like
        Observation timestamps.
    Ts, airT : array_like
        Surface-water and air temperatures in degrees Celsius.
    hML : array_like
        Active mixed-layer depth in metres.
    wnd : array_like
        Wind speed in metres per second.
    RH : array_like
        Relative humidity in percent.
    sw, lwnet : array_like
        Short-wave and net long-wave radiation in watts per square metre.
    lwnet_mode : int, default: 1
        Whether ``lwnet`` already represents net long-wave radiation.
    s : float, default: 0.2
        Salinity retained for compatibility.

    Returns
    -------
    numpy.ndarray
        Standardized gas-transfer velocity in metres per day.
    """
    #'@param wnd Numeric value of wind speed, (Units:m/s)
    #'@param method Only for \link{k.crusius.base}. String of valid method . Either "constant", "bilinear", or "power"
    #'@param wnd_z Height of wind measurement, (Units: m)
    #'@param Kd Light attenuation coefficient (Units: m**-1)
    #'@param lat Latitude, degrees north
    #'@param A0 Lake area, m**2
    #'@param air_press Atmospheric pressure, (Units: millibar)
    #'@param dateTime datetime (Y-\%m-\%d \%H:\%M), (Format: \code{\link{POSIXct}})
    #'@param Ts Numeric vector of surface water temperature, (Units(deg C)
    #'@param hML Numeric vector of actively mixed layer depths. Must be the same length as the Ts parameter
    #'@param airT Numeric value of air temperature, Units(deg C)
    #'@param RH Numeric value of relative humidity, \%
    #'@param sw Numeric value of short wave radiation, W m**-2
    #'@param lwnet Numeric value net long wave radiation, W m**-2, 
    # define constants used in function
    wnd, Qsen, Qlat, Cd, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet = list(map(np.asanyarray, (wnd, \
        Qsen, Qlat, Cd, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet)))
    #if no net, convert it to net
    if not lwnet_mode:
        Tk = Ts+Kelvin # water temperature in Kelvin
        LWo = S_B*emiss*Tk**4 # long wave out
        lwnet = lwnet-LWo

    Kelvin = 273.15 # temp mod for deg K   
    emiss = 0.972 # emissivity;
    S_B = 5.67E-8 # Stefan-Boltzman constant (?K is used)
    vonK = 0.41 # von Karman  constant
    dT = 0.5   # change in temp for mixed layer depth
    C1 = 114.278 # from Soloviev et al. 2007
    nu = 0.29 # proportionality constant from Zappa et al. 2007, lower bounds
    KeCrit = 0.18     # constant for wave age = 20 (Soloviev et al. 2007)
    albedo_SW = 0.07
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)
    mnWnd = 0.2 # minimum wind speed

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    wnd[rpcI] = mnWnd

    # calculate sensible and latent heat fluxes
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        wnd = wnd/(1-e1/vonK*np.log(10/wnd_z))
        
    rhoAir = 1.2 #  air density
    tau = C_D*wnd**2*rhoAir
    uSt = np.sqrt(tau/rho_w)

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) # Kim 1976

    # calculate the thermal expansion coefficient 
    tExp = thermalExpFromTemp(Ts)

    # calculate buoyancy flux and w*
    B1 = H_star*tExp*g
    B2 = rho_w*C_w
    Bflx = B1/B2
    ltI = Bflx>0
    if type(B1)==np.float64:
        B1 = np.array(B1)
    B1[ltI] = 0
    divi = 1/3
    w1 = -B1*hML
    wSt = w1**divi

    # calculate kinematic viscosiy
    kinV = getKinematicVis(Ts)

    KeDe = (kinV*g)
    KeNm = uSt**3
    Ke = KeNm/KeDe
    tau = tau    # estimate of total tau (includes wave stress)
    euPw = (1+Ke/KeCrit)  # tau_shear = tau/(1+Ke/Kecr)
    tau_t = tau/euPw      # tau_shear, Soloviev
    uTanS = tau_t/rho_w   
    uTanS = uTanS**0.5

    # calculate viscous sublayer
    Sv = C1*kinV/uTanS
    eu_N = uTanS**3      # e_u(0) = (tau_t/rho)**1.5/(vonK*Sv)
    eu_D = vonK*Sv       # denominator
    eu_0 = eu_N/eu_D    # in m2/s3
    ew_0 = -1.0*B1       # buoyancy flux, but only when outward
    e_0 = ew_0+eu_0     # e(0) from Soloviev (w/o wave effects)
    K1 = e_0*kinV       # in units of m4/s4, want cm4/hr4
    K2 = ew_0*kinV      # convective component (m4/s4)
    K1 = K1*100**4*3600**4 # now in cm4/hr4  (Total)
    K2 = K2*100**4*3600**4 # now in cm4/hr4  (Convective)
    K600 = nu*600**(-0.5)*K1**(1/4)   # in cm/hr (Total)

    #k600 = np.numeric(K600)
    k600 = K600*24/100 #now in units in m d-1
    return(k600)

def k_heiskanen(wnd_z, Cd, Qlat, Qsen, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet):
    """Estimate ``k600`` using the Heiskanen turbulence parameterization.

    The model combines wind shear and penetrative convection derived from the
    surface heat budget. Inputs use SI units; the returned velocity is in
    metres per day.

    Method
    ----------
    Wind speed is adjusted to 10 m and converted to water-side friction
    velocity. Negative buoyancy flux produces a convective velocity scale.
    Wind and convection contributions are combined following Heiskanen et al.,
    standardized to a Schmidt number of 600, and converted to metres per day.

    Parameters
    ----------
    wnd_z : float
        Wind measurement height in metres.
    Cd : array_like
        Momentum drag coefficient.
    Qlat, Qsen : array_like
        Latent and sensible heat fluxes in watts per square metre.
    Kd : array_like
        Light attenuation coefficient in inverse metres.
    air_press : array_like
        Atmospheric pressure in millibars.
    dateTime : array_like
        Observation timestamps.
    Ts, airT : array_like
        Surface-water and air temperatures in degrees Celsius.
    hML : array_like
        Active mixed-layer depth in metres.
    wnd : array_like
        Wind speed in metres per second.
    RH : array_like
        Relative humidity in percent.
    sw, lwnet : array_like
        Short-wave and net long-wave radiation in watts per square metre.

    Returns
    -------
    float or numpy.ndarray
        Standardized gas-transfer velocity ``k600``.
    """

    #Constants
    S_B = 5.67E-8 # Stefan-Boltzman constant (K is used)
    emiss = 0.972 # emissivity
    Kelvin = 273.15 #conversion from C to Kelvin
    albedo_SW = 0.07
    vonK = 0.41 #von Karman constant
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    mnWnd = 0.2 # minimum wind speed
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    if type(wnd)==int:
        wnd=np.array(wnd)
    wnd[rpcI] = mnWnd


    # calculate sensible and latent heat fluxes
    #mm = calc.zeng(dateTime,Ts,airT,wnd,RH,air_press,wnd_z)
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        u10 = wnd/(1-e1/vonK*np.log(10/wnd_z))
    else:
        u10 = wnd


    rhoAir = 1.2 #  air density
    vonK = 0.41 # von Karman  constant
    tau = C_D*u10**2*rhoAir
    uSt = np.sqrt(tau/rho_w)

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) # Kim 1976

    # calculate the thermal expansion coefficient 
    tExp = thermalExpFromTemp(Ts)

    B1 = H_star*tExp*g #Imberger 1985: Effective heat flux * thermal expansion of water * gravity
    B2 = rho_w*C_w # mean density of the water column * specific heat of water at constant pressure
    Bflx = B1/B2

    if Bflx<0:
        wstar = (-Bflx*hML)**(1/3)#penetrative convective velocity Heiskanen 2014 (Imberger 1985)
    else:
        wstar = 0
    Hk   = np.sqrt((0.00015*u10)**2 + (0.07*wstar)**2) 
    Hk   = Hk*100*3600 # Heiskanen's K in cm/hr
    Hk600 = Hk*600**(-0.5)
    k600 = Hk600*24/100 #units in m d-1
    return(k600)


def k_macIntyre(wnd_z, Cd, Qlat, Qsen, Kd, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet,
                                params=np.array([1.2,0.4872,1.4784])):
    """Estimate ``k600`` using the MacIntyre surface-renewal model.

    Method
    ----------
    Turbulent kinetic energy is estimated from surface cooling and wind shear.
    Negative energy estimates are clipped to zero. The supplied empirical
    coefficients scale the convective, shear, and surface-renewal components
    before conversion to metres per day.

    Parameters
    ----------
    wnd_z : float
        Wind measurement height in metres.
    Cd, Qlat, Qsen, Kd : array_like
        Drag coefficient, latent and sensible heat fluxes, and light
        attenuation coefficient.
    air_press, dateTime : array_like
        Atmospheric pressure and observation timestamps.
    Ts, hML, airT, wnd, RH : array_like
        Surface-water temperature, mixed-layer depth, air temperature, wind
        speed, and relative humidity.
    sw, lwnet : array_like
        Short-wave and net long-wave radiation in watts per square metre.
    params : array_like, default: [1.2, 0.4872, 1.4784]
        Convection, shear, and transfer scaling coefficients.

    Returns
    -------
    float or numpy.ndarray
        Standardized gas-transfer velocity in metres per day.
    """

    #Constants
    S_B = 5.67E-8 # Stefan-Boltzman constant (K is used)
    emiss = 0.972 # emissivity;
    Kelvin = 273.15 #conversion from C to Kelvin
    albedo_SW = 0.07
    vonK = 0.41 #von Karman constant
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    mnWnd = 0.2 # minimum wind speed
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    if type(wnd)==int:
        wnd=np.array(wnd)
    wnd[rpcI] = mnWnd


    # calculate sensible and latent heat fluxes
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        u10 = wnd/(1-e1/vonK*np.log(10/wnd_z))
    else:
        u10 = wnd
    

    rhoAir = 1.2 #  air density
    vonK = 0.41 # von Karman  constant
    tau = C_D*u10**2*rhoAir
    uSt = np.sqrt(tau/rho_w)

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) # Kim 1976


    # calculate the thermal expansion coefficient
    tExp = thermalExpFromTemp(Ts)

    B1 = H_star*tExp*g
    B2 = rho_w*C_w
    Bflx = B1/B2

    # calculate kinematic viscosiy
    kinV = getKinematicVis(Ts)
    KeNm = uSt**3

    #SmE   = 0.84*(-0.58*Bflx+1.76*KeNm/(vonK*hML))
    SmE = params[0]*-Bflx+params[1]*KeNm/(vonK*hML) #change to two coefficients
    if type(SmE)==np.float64:
        SmE=np.array(SmE)
    SmE[SmE<0] = 0    # set negative to 0
    Sk   = SmE*kinV
    Sk   = Sk*100**4*3600**4 # Sally's K now in cm4/h4
    Sk600 = params[2]*600**(-0.5)*Sk**(1/4) # in cm/hr (Total)

    k600 = Sk600 # why is this not already numeric?
    k600 = k600*24/100 #units in m d-1
    return k600

def k_read_soloviev(wnd_z, Cd, Qlat, Qsen, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet):
    """Estimate ``k600`` with Soloviev turbulence and breaking-wave effects.

    The calculation combines convection, wind shear, wave turbulence, and a
    bubble-mediated exchange component. Lake area is used to approximate fetch.

    Method
    ----------
    The Read surface-renewal calculation is extended with the Soloviev wave-age
    formulation. Fetch is approximated from lake surface area, then wind shear,
    convection, wave breaking, and bubble-mediated exchange are combined into
    the final Schmidt-number-600 transfer velocity.

    Parameters
    ----------
    wnd_z : float
        Wind measurement height in metres.
    Cd, Qlat, Qsen, Kd : array_like
        Drag coefficient, latent and sensible heat fluxes, and light
        attenuation coefficient.
    lat : float
        Latitude in decimal degrees.
    A0 : float
        Lake surface area in square metres.
    air_press, dateTime : array_like
        Atmospheric pressure and observation timestamps.
    Ts, hML, airT, wnd, RH : array_like
        Surface-water temperature, mixed-layer depth, air temperature, wind
        speed, and relative humidity.
    sw, lwnet : array_like
        Short-wave and net long-wave radiation in watts per square metre.

    Returns
    -------
    numpy.ndarray
        Total standardized gas-transfer velocity in metres per day.
    """
    
    wnd_z, Cd, Qlat, Qsen, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet = list(map(np.asanyarray, (wnd_z,\
         Cd, Qlat, Qsen, Kd, lat, A0, air_press, dateTime, Ts, hML, airT, wnd, RH, sw, lwnet)))
    # define constants used in function
    Kelvin = 273.15 # temp mod for deg K
    emiss = 0.972 # emissivity;
    S_B = 5.67E-8 # Stefan-Boltzman constant (?K is used)
    vonK = 0.41 # von Karman  constant
    dT = 0.5   # change in temp for mixed layer depth
    C1 = 114.278 # from Soloviev et al. 2007
    nu = 0.29 # proportionality constant from Zappa et al. 2007, lower bounds
    KeCrit = 0.18     # constant for wave age = 20 (Soloviev et al. 2007)
    albedo_SW = 0.07
    swRat = 0.46 # percentage of SW radiation that penetrates the water column
    g = 9.81 # gravity
    C_w = 4186 # J kg-1 ?C-1 (Lenters et al. 2005)
    mnWnd = 0.2 # minimum wind speed

    # impose limit on wind speed
    rpcI = wnd < mnWnd
    wnd[rpcI] = mnWnd

    # calculate sensible and latent heat fluxes
    C_D = Cd # drag coefficient for momentum
    E = Qlat # latent heat flux
    H = Qsen # sensible heat flux

    # calculate total heat flux
    dUdt = sw*0.93 - E - H + lwnet
    Qo = sw*(1-albedo_SW)*swRat #PAR

    # calculate the effective heat flux
    q1 = 2-2*np.exp(hML*-Kd)
    q2 = hML*Kd
    q3 = np.exp(hML*-Kd)
    H_star = dUdt-Qo*(q1/q2-q3) #Effective surface heat flux Kim 1976

    # calculate water density
    rho_w = dens0(t=Ts, s=0.2)

    # calculate u*
    if (wnd_z != 10):
        e1 = np.sqrt(C_D)
        wnd = wnd/(1-e1/vonK*np.log(10/wnd_z))
    
    rhoAir = 1.2 #  air density
    tau = C_D*wnd**2*rhoAir
    uSt = np.sqrt(tau/rho_w)
    uSta = np.sqrt(tau/rhoAir)  #friction velocity in air

    # calculate the thermal expansion coefficient
    tExp = thermalExpFromTemp(Ts)

    # calculate buoyancy flux and w*
    B1 = H_star*tExp*g #Hstar * coefficient of thermal expansion * gravity
    B2 = rho_w*C_w
    Bflx = B1/B2
    
    if type(Bflx)==np.float64:
        Bflx = np.array(Bflx)
    Bflx[Bflx>0] = 0

    wSt = (-Bflx*hML)**1/3

    # calculate kinematic viscosiy
    kinV = getKinematicVis(Ts)
    kinVa = getKinematicVis(airT)

    KeDe = (kinV*g)
    KeNm = uSt**3
    Ke = KeNm/KeDe
    tau = tau    # estimate of total tau (includes wave stress)
    euPw = (1+Ke/KeCrit)  # tau_shear = tau/(1+Ke/Kecr) Ke is the Keulegan number
    # Could calculate KeCrit (critical Keulegan number) from wave age
    #KeCrit = (kinVa/kinV)*((rhoAir/rho_w)**1.5)*(Rbcr/Aw) # Eq1.16-Soloviev et al(2007)

    tau_t = tau/euPw      # tau_t = tangential wind stress, tau = total wind stress
    uTanS = tau_t/rho_w
    uTanS = uTanS**0.5

    # calculate viscous sublayer
    Sv = C1*kinV/uTanS  # effective thickness of the aqueous viscous sublayer
    eu_N = uTanS**3      # e_u(0) = (tau_t/rho)**1.5/(vonK*Sv)
    eu_D = vonK*Sv      # denominator
    eu_0 = eu_N/eu_D    # in m2/s3
    ec_0 = -1.0*Bflx       # buoyancy flux, but only when outward

    #ewave_0 turbulence due to wave breaking
    A0 = A0/1e6 # convert surface area to km
    Fetch = 2*np.sqrt(A0/np.pi) # fetch in km (assuming a conical lake)
    Hs = 0.0163*(Fetch**0.5)*wnd # significant wave height - Woolf (2005)
    Aw = (1/(2*np.pi))*(( (g*Hs*rhoAir)/(0.062*rho_w*
            uSt**2))**(2/3)) # wave age - eqn 1.11 Soloviev et al. (2007)

    W = 3.8e-6*wnd**3.4 # simplified whitecap fraction (Fariall et al. 2000)


    Ap = 2.45*W*((1/(W**0.25))-1)
    alphaW = 100 # p. 185 - Soloviev et al. (2007)
    B = 16.6 # p. 185 - Soloviev et al. (2007)
    Sq = 0.2 # p. 185 - Soloviev et al. (2007)
    cT = 0.6 # p. 188 - Soloviev et al. (2007)
    ewave_0 = ((Ap**4)*alphaW)*((3/(B*Sq))**0.5) * \
                    (((Ke/KeCrit)**1.5)/((1+Ke/KeCrit)**1.5))* \
                    (uSt*g*kinV)/(0.062*vonK*cT*((2*np.pi*Aw)**1.5)) * \
                    (rhoAir/rho_w)

    #------------------------------------
    e_0 = ec_0+eu_0+ewave_0    # e(0) from Soloviev (w/o wave effects)
    Kc = ec_0*kinV*100**4*3600**4      # convective component now in cm4/hr4  (Total)
    Ku = eu_0*kinV*100**4*3600**4 # shear component now in cm4/hr4  (Total)
    Kwave = ewave_0*kinV*100**4*3600**4 # wave component now in cm4/hr4  (Total)
    Kall = e_0*kinV*100**4*3600**4       # turbulent kinetic energy now in cm4/hr4  (Total)

    #Schmidt number could be calculated as temperature dependent
    #Sc = 1568+(-86.04*Ts)+(2.142*Ts**2)+(-0.0216*Ts**3)
    k600org = nu*600**(-0.5)*(Kc+Ku)**(1/4)   # in cm/hr (Total)
    k600org = k600org*24/100 #now in units in m d-1

    k600 = nu*600**(-0.5)*Kall**(1/4)   # in cm/hr (Total)
    k600 = k600*24/100 #now in units in m d-1

    # ---Breaking Wave Component, Author: R I Woolway, 2014-11-13 ---
    # bubble mediated component - Woolf 1997
    kbi = W*2450
    beta_0 = 2.71*1e-2 # Ostwald gas solubility (Emerson and Hedges, 2008)
    Sc = 1568+(-86.04*Ts)+(2.142*Ts**2)+(-0.0216*Ts**3) # Schmidt number
    kbiii = (1+(1/(14*beta_0*Sc**(-0.5))**(1/1.2)))**1.2
    kb = kbi/((beta_0*kbiii))
    kb = kb*24/100 #units in m d-1
    #----------------------------------------------------------------

    k600b = k600+kb
    #allks = pd.DataFrame(data =(Ku,Kc,Kwave,kb,k600org,k600,k600b), columns= ["shear","convective","wave","bubble","k600org","k600",'k600b'])
    return k600b

def k_vachon(wnd, A0, params=np.array([2.51,1.48,0.39])):
    """Estimate ``k600`` from wind speed and lake surface area.

    Method
    ----------
    The Vachon empirical relationship combines an intercept, a linear wind
    term, and an interaction between wind speed and the base-10 logarithm of
    lake area in square kilometres. The result is converted from centimetres
    per hour to metres per day.

    Parameters
    ----------
    wnd : array_like
        Wind speed in metres per second.
    A0 : float
        Lake surface area in square metres.
    params : array_like, default: [2.51, 1.48, 0.39]
        Intercept, wind, and wind-area coefficients.

    Returns
    -------
    array_like
        Standardized gas-transfer velocity in metres per day.
    """
    U10 = wnd  #This function uses just the wind speed it is supplied
    k600 = params[0] + params[1]*U10 + params[2]*U10*np.log10(A0/1000000) # units in cm h-1
    k600 = k600*24/100 #units in m d-1
    return(k600)
    

def k600_2_kGAS(k600,temperature,gas="O2"):
    """Convert ``k600`` to a gas-specific transfer velocity.

    Method
    ----------
    The gas Schmidt number is evaluated from water temperature using a cubic
    empirical relationship. ``k600`` is then scaled by the square root of the
    ratio between that Schmidt number and 600.

    Parameters
    ----------
    k600 : array_like
        Gas-transfer velocity standardized to a Schmidt number of 600.
    temperature : array_like
        Water temperature in degrees Celsius.
    gas : str, default: "O2"
        Gas identifier supported by :func:`getSchmidt`.

    Returns
    -------
    array_like
        Transfer velocity in the same units as ``k600``.

    Notes
    -----
    A Schmidt-number exponent of ``-0.5`` is used.
    """
    #'@title Returns the gas exchange velocity for gas of interest w/ no unit conversions
    #'@description 
    #'Returns the gas exchange velocity for gas of interest w/ no unit conversions
    #'@usage
    #'k600.2.kGAS.base(k600,temperature,gas="O2")
    #'
    #'k600.2.kGAS(ts.data, gas="O2")
    #'
    #'@param ts.data Object of class data.frame with named columns datetime and k600 and wtr (water temp in deg C). Other columns are ignored
    #'@param k600 k600 as vector array of numbers or single number
    #'@param temperature Water temperature (deg C) as vector array of numbers or single number
    #'@param gas gas for conversion, as string (e.g., 'CO2' or 'O2')
    #'@return Numeric value of gas exchange velocity for gas
    #'@author Jordan S. Read
    #'@seealso \link{k.read} and \link{k.read.base} for functions that calculate k600 estimates
    n	=	0.5
    schmidt	=	getSchmidt(temperature,gas)
    Sc600	=	schmidt/600

    kGAS	=	k600*(Sc600**-n)
    return(kGAS)
