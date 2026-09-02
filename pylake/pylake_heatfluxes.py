import numpy as np
import pandas as pd
from pysolar.solar import get_altitude
import warnings
import xarray as xr
from datetime import datetime, timezone
from .functions import *
from .functions_heat import *

def calculate_cloud_cover(date, LON, LAT, ELEV, Q, T, Rh, P):
    # Woolway 2015 procedure to calculate DAILY cloud cover using the Pysolar package to obtain Zenith
    # The DOY calc and the daily-mean windowing below assume date is in
    # ascending chronological order. We don't silently sort it ourselves -
    # that risks desynchronising it from Q/T/Rh, which the caller must sort
    # to match. Raise instead so out-of-order input is caught at the call
    # site.
    date = check_sorted(ensure_utc(date))

    Zenit = np.full(len(date), np.nan)
    DOY = np.full(len(date), np.nan)
    for i in range(len(date)):
        Zenit[i] = 90 - get_altitude(LAT, LON, date[i])  # , ELEV, T[i], P[i])
        DOY[i] = (date[i] - datetime(date[i].year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 24. / 60. / 60.
        # CSRad[i] = solar.radiation.GetRadiationDirect(date[i],altitude)
    # Effective solar constant: squared eccentricity-correction factor,
    # matching Woolway et al. (2015)'s own implementation (calc_lwnet.m:
    # I0 = 1353*cosN.*cosN where cosN = 1+0.034*cos(...)) - not the
    # unsquared version this file had before.
    Ieff = 1353 * (1 + 0.034 * np.cos(2 * np.pi * DOY / 365)) ** 2
    Ieff[Ieff < 0] = 0.
    cosZ = np.cos(np.pi / 180. * Zenit)
    cosZ[cosZ < 0] = 0.

    # Rayleigh scattering
    p = (101325. * (1 - ELEV * 2.25577e-5) ** 5.25588) / 100.  # surface pressure
    # air-mass thickness: the 1224 constant matches calc_lwnet.m
    # (m1 = 1224*cosZ.^2+1) - this file previously had 1244.
    m = 35. * cosZ * (1224. * cosZ ** 2 + 1) ** -0.5
    TrTpg = 1.021 - 0.084 * (m * (0.000949 * p + 0.051)) ** 0.5

    # Water vapour absortion. Td (dewpoint, deg C) follows the standard
    # Magnus formula with no added offset - calc_lwnet.m's T_d has none
    # either; this file previously added a spurious +33.8 here. The
    # precipitable-water term (pw) needs Td in Fahrenheit
    # (calc_lwnet.m: T_d = T_d*9/5+32 right before it's used) - this file
    # previously used Td in Celsius directly, unconverted.
    es = saturation_vapour_pressure(T)
    ez = Rh * es / 100.
    Td = (243.5 * np.log(ez / 6.112)) / (17.67 - np.log(ez / 6.112))
    Td_F = Td * 9. / 5. + 32.
    G = G_constant(DOY, LAT)
    pw = np.exp((0.1133 - np.log(G + 1)) + 0.0393 * Td_F)
    Tw = 1. - 0.077 * (pw * m) ** 0.3

    # Aerosol attenuation (Meyers & Dale) - calc_lwnet.m uses 0.935, not
    # the 0.95 this file previously had.
    Ta = 0.935 ** m

    # clear-sky irradiance
    Ic = Ieff * cosZ * TrTpg * Tw * Ta

    # calculates daily cloud cover using a time-based rolling mean: a +-12h
    # window around each point (so midnight uses everything from the
    # previous noon to the same day's noon), based on actual elapsed time
    # rather than sample count, so gaps in the series widen the effective
    # averaging window instead of silently covering more or less than a
    # day's worth of real time.
    #
    # Only daytime samples feed the average - Ic is exactly 0 at night, so
    # including nighttime rows would dilute (or, if a window is all-night,
    # zero out) the mean. Masking them to NaN before rolling means
    # pandas' rolling mean skips them and uses only the nearby daytime
    # samples - which is what lets a *nighttime* point still get a
    # meaningful cloud-cover value, from the daytime data in its window,
    # instead of being undefined.
    night = cosZ <= 0
    Ic_day = np.where(night, np.nan, Ic)
    Q_day = np.where(night, np.nan, np.asarray(Q, dtype=float))

    rolled = pd.DataFrame(
        {"Ic": Ic_day, "Q": Q_day}, index=date
    ).rolling("1D", center=True, min_periods=1).mean()
    mIc = rolled["Ic"].to_numpy()
    mQ = rolled["Q"].to_numpy()

    # percentage of clear sky radiation: an absolute ratio of measured to
    # theoretical clear-sky irradiance, NOT rescaled relative to whatever
    # else is in this dataset. csf == 1 means measured radiation equals the
    # clear-sky model's prediction (0% cloud); csf == 0 means no radiation
    # got through at all (100% cloud). mIc/mQ can still be NaN here if a
    # window happens to contain no daytime samples at all (e.g. very sparse
    # or polar-night data) - there's genuinely nothing to compute from in
    # that case, so it stays NaN rather than being forced to a value.
    with np.errstate(divide="ignore", invalid="ignore"):
        csf = mQ / mIc
    # measurement noise or clear-sky model underestimation can occasionally
    # push the ratio slightly above 1 (or, less commonly given Q was
    # clipped >= 0, below 0) - clip to keep it physically bounded rather
    # than letting clf run negative or past 1. NaN passes through both
    # comparisons unchanged, so night-with-no-data stays NaN.
    csf = np.clip(csf, 0., 1.)
    clf = 1 - csf

    return clf

def shortwave_radiation(date, SRad, cloud_cover, lat):
    # albedo for absorbed short-wave from Fink et al. (2014) based on Cogley (1979)
    date = check_sorted(ensure_utc(date))
    # Coerce to a plain array: callers sometimes pass a pandas Index (e.g. an
    # array built via arithmetic on date.hour/date.minute keeps returning an
    # Index all the way through), and Index objects are immutable so the
    # boolean-mask assignment below would raise
    # "TypeError: Index does not support mutable operations".
    SRad = np.asarray(SRad, dtype=float)
    albedo_diff = 0.066
    albedo_dir_array = calculate_albedo_dir(lat)

    # Direct and diffusive fraction based on cloud cover
    Fdir = (1. - cloud_cover) / ((1. - cloud_cover) + 0.5 * cloud_cover)
    Fdiff = 0.5 * cloud_cover / ((1. - cloud_cover) + 0.5 * cloud_cover)

    SRad[SRad < 0.] = 0.

    # Vectorized month-of-year (1-12) and day-of-month for every timestamp -
    # `date` is a DatetimeIndex, so this replaces the old per-element loop
    # that referenced an undefined `i`/`d`.
    month = date.month.to_numpy()
    day = date.day.to_numpy()

    # Piecewise linear interpolation between the two nearest mid-month
    # (day-15) climatological anchors - matches the two-segment reference
    # scheme: days 1-15 interpolate between the previous and current
    # month, days 16-31 interpolate between the current and next month.
    # (Both segments meet exactly at day 15, which reproduces the current
    # month's own table value.)
    prev_month = np.where(month == 1, 12, month - 1)
    next_month = np.where(month == 12, 1, month + 1)

    before_16 = day < 16
    albedo_start = np.where(before_16, albedo_dir_array[prev_month - 1], albedo_dir_array[month - 1])
    albedo_end = np.where(before_16, albedo_dir_array[month - 1], albedo_dir_array[next_month - 1])
    weight = np.where(before_16, day + 15, day - 15)
    albedo_dir = albedo_start + weight * (albedo_end - albedo_start) / 30

    Qsw = SRad * (Fdir * (1. - albedo_dir) + Fdiff * (1. - albedo_diff))
    return Qsw

def longwave_in(Ta, RH, C):
    # Absorbed atmospheric long-wave radiation, following Fink et al.
    # Sign convention: positive = heat flux INTO the lake (this term is
    # always a gain, so it is always >= 0) - matches shortwave_radiation's
    # Qsw and longwave_out's sign convention below.
    sigma = 5.67e-8  # Stefan-Boltzmann constant, W m-2 K-4
    AL = 0.03        # long-wave surface albedo
    a = 1.0592
    Cc = 0.17

    es = saturation_vapour_pressure(Ta)
    ea = RH * es / 100.
    Ea = a * (1 + Cc * C ** 2) * 1.24 * (ea / (Ta + 273.16)) ** (1 / 7.)  # atmospheric emissivity
    Qlw_in = (1 - AL) * Ea * sigma * (Ta + 273.16) ** 4
    return Qlw_in

def longwave_out(Tw):
    # Emitted long-wave radiation from the water surface, following Fink et al.
    # Sign convention: positive = heat flux INTO the lake, so emission (a
    # loss) is returned negative (this term is always a loss, so always <= 0).
    sigma = 5.67e-8  # Stefan-Boltzmann constant, W m-2 K-4
    Qlw_out = -0.972 * sigma * (Tw + 273.16) ** 4
    return Qlw_out

def latent(Ta, RH, Tw, Wsp):
    # Latent (evaporative) heat flux, following Fink et al.
    # Sign convention: positive = heat flux INTO the lake, so evaporation
    # (esw > ea, the usual case) comes out negative (a loss).
    es = saturation_vapour_pressure(Ta)
    ea = RH * es / 100.
    esw = saturation_vapour_pressure(Tw)  # attention: it is with water temperature
    f = 4.8 + 1.98 * Wsp + 0.28 * (Tw - Ta)
    Qlat = -f * (esw - ea)
    return Qlat

def sensible(Ta, Tw, Wsp, P):
    # Sensible heat flux, following Fink et al.
    # Sign convention: positive = heat flux INTO the lake, so a warmer
    # water surface than air (Tw > Ta, the usual case) comes out negative
    # (a loss).
    # P: air pressure [mbar/hPa], consistent with the mbar/hPa convention
    # used throughout this module (e.g. calculate_cloud_cover's P).
    Cpa = 1005.0  # specific heat of air at constant pressure [J kg-1 K-1] -
                  # not given in the original formula, using the standard
                  # dry-air value.
    f = 4.8 + 1.98 * Wsp + 0.28 * (Tw - Ta)
    Lv = latent_heat_vap(Tw)
    gamma = (Cpa * P) / (0.622 * Lv)
    Qsen = -gamma * f * (Tw - Ta)
    return Qsen