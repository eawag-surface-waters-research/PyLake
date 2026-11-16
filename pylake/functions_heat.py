import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Empirical "G" constant used in the precipitable-water term of the
# clear-sky irradiance model, from Smith (1966), table 1 - transcribed from
# Woolway et al. (2015)'s Lake Heat Flux Analyzer (calc_lwnet.m /
# getSmithGamma), including that source's own season breakpoints
# (solstice/equinox-anchored, with a repeated point at each end for annual
# wraparound) and lat bins (band centers every 10 degrees).
_G_LAT = np.array([5., 15., 25., 35., 45., 55., 65., 75., 85.])
_G_DOY = np.array([-10., 81., 173., 264., 355., 446.])
_G_TABLE = np.array([
    [3.37, 2.85, 2.80, 2.64, 3.37, 2.85],
    [2.99, 3.02, 2.70, 2.93, 2.99, 3.02],
    [3.60, 3.00, 2.98, 2.93, 3.60, 3.00],   # last value: 2.98 in calc_lwnet.m, a typo (the
                                            # repeated spring value must equal column 2)
    [3.04, 3.11, 2.92, 2.94, 3.04, 3.11],
    [2.70, 2.95, 2.77, 2.71, 2.70, 2.95],
    [2.52, 3.07, 2.67, 2.93, 2.52, 3.07],
    [1.76, 2.69, 2.61, 2.61, 1.76, 2.69],
    [1.60, 1.67, 2.24, 2.63, 1.60, 1.67],
    [1.11, 1.44, 1.94, 2.02, 1.11, 1.44],
])
_G_INTERPOLATOR = RegularGridInterpolator(
    (_G_LAT, _G_DOY), _G_TABLE, bounds_error=False, fill_value=None
)

def G_constant(DOY, LAT):
    # As getSmithGamma in calc_lwnet.m: southern hemisphere = northern hemisphere shifted by
    # 182 days (seasons) with |lat|; latitudes outside the table (5-85 deg) use the nearest band
    # instead of being extrapolated linearly
    DOY = np.asarray(DOY, dtype=float)
    LAT = np.broadcast_to(np.asarray(LAT, dtype=float), DOY.shape)
    DOY = np.where(LAT < 0, (DOY + 182.) % 365., DOY)
    LAT = np.clip(np.abs(LAT), _G_LAT[0], _G_LAT[-1])
    return _G_INTERPOLATOR(np.column_stack([LAT, DOY]))

def saturation_vapour_pressure(T):
    # Saturation vapour pressure (hPa)
    # Input:
    # T: air temperature in degree C
    # Outout
    # es: saturation vapour pressure (mbar/hPa)
    es = 6.11 * np.exp((17.27 * T) / (237.3 + T))
    return es

def dew_point(e):
    # Dew point [degC] from the vapour pressure e [hPa]: inverse of saturation_vapour_pressure
    x = np.log(e / 6.11)
    return 237.3 * x / (17.27 - x)

def wind_function(Wsp, Tw, Ta):
    # Wind function f(u) [W m-2 hPa-1] of the latent and sensible heat flux (Fink et al. 2014):
    # f = 4.8 + 1.98 u10 + 0.28 (Tw - Ta). Limited to >= 0: with calm wind and air much warmer
    # than the water (Tw - Ta < about -17 degC) the formula would turn negative and reverse
    # the direction of both fluxes.
    return np.maximum(4.8 + 1.98 * Wsp + 0.28 * (Tw - Ta), 0.)

def latent_heat_vap(T):
    # latent heat of vaporiyation.
    # Inputs
    # T [degC]
    # Output:
    # Lv in J/kg
    Lv = 2.501e6 - 2370. * T
    return Lv

def calculate_albedo_dir(lat):
    """
    Monthly albedo of the water surface for direct short-wave radiation (Cogley 1979),
    for the 10-degree latitude band of `lat`.

    Parameters
    ----------
    lat : float
        Latitude in degrees (negative: southern hemisphere).

    Returns
    -------
    albedo : np.ndarray
        12 monthly values (January to December).
    """

    # Northern hemisphere
    if lat > 0:

        albedo_data = np.array([
            [0.069, 0.065, 0.063, 0.063, 0.065, 0.066, 0.065, 0.063, 0.063, 0.065, 0.068, 0.070],
            [0.076, 0.070, 0.065, 0.063, 0.063, 0.064, 0.063, 0.063, 0.064, 0.069, 0.075, 0.079],
            [0.091, 0.079, 0.070, 0.065, 0.064, 0.064, 0.064, 0.064, 0.068, 0.076, 0.089, 0.097],
            [0.121, 0.097, 0.078, 0.069, 0.066, 0.065, 0.066, 0.068, 0.075, 0.091, 0.116, 0.132],
            [0.178, 0.131, 0.095, 0.077, 0.071, 0.070, 0.070, 0.075, 0.088, 0.120, 0.169, 0.198],
            [0.263, 0.193, 0.127, 0.093, 0.080, 0.077, 0.079, 0.088, 0.114, 0.174, 0.249, 0.294],
            [0.340, 0.281, 0.185, 0.122, 0.099, 0.095, 0.097, 0.113, 0.163, 0.254, 0.336, 0.325],
            [0.301, 0.337, 0.266, 0.178, 0.138, 0.123, 0.132, 0.163, 0.238, 0.329, 0.301, 1.000],
            [1.000, 0.301, 0.333, 0.253, 0.167, 0.133, 0.150, 0.226, 0.317, 0.301, 1.000, 1.000]
        ])

        if lat < 10:
            lat_number = 1
        elif lat < 20:
            lat_number = 2
        elif lat < 30:
            lat_number = 3
        elif lat < 40:
            lat_number = 4
        elif lat < 50:
            lat_number = 5
        elif lat < 60:
            lat_number = 6
        elif lat < 70:
            lat_number = 7
        elif lat < 80:
            lat_number = 8
        else:
            lat_number = 9

    # Southern hemisphere
    else:

        albedo_data = np.array([
            [0.065, 0.063, 0.063, 0.065, 0.068, 0.070, 0.069, 0.065, 0.063, 0.063, 0.065, 0.066],
            [0.063, 0.063, 0.064, 0.069, 0.075, 0.079, 0.076, 0.070, 0.065, 0.063, 0.063, 0.064],
            [0.064, 0.064, 0.068, 0.076, 0.089, 0.097, 0.091, 0.079, 0.070, 0.065, 0.064, 0.064],
            [0.066, 0.068, 0.075, 0.091, 0.116, 0.132, 0.121, 0.097, 0.078, 0.069, 0.066, 0.065],
            [0.070, 0.075, 0.088, 0.120, 0.169, 0.198, 0.178, 0.131, 0.095, 0.077, 0.071, 0.070],
            [0.079, 0.088, 0.114, 0.174, 0.249, 0.294, 0.263, 0.193, 0.127, 0.093, 0.080, 0.077],
            [0.097, 0.113, 0.163, 0.254, 0.336, 0.325, 0.340, 0.281, 0.185, 0.122, 0.099, 0.095],
            [0.132, 0.163, 0.238, 0.329, 0.301, 1.000, 0.301, 0.337, 0.266, 0.178, 0.138, 0.123],
            [0.150, 0.226, 0.317, 0.301, 1.000, 1.000, 1.000, 0.301, 0.333, 0.253, 0.167, 0.133]
        ])

        if lat > -10:
            lat_number = 1
        elif lat > -20:
            lat_number = 2
        elif lat > -30:
            lat_number = 3
        elif lat > -40:
            lat_number = 4
        elif lat > -50:
            lat_number = 5
        elif lat > -60:
            lat_number = 6
        elif lat > -70:
            lat_number = 7
        elif lat > -80:
            lat_number = 8
        else:
            lat_number = 9

    return albedo_data[lat_number - 1]