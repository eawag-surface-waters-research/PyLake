import numpy as np
import xarray as xr
import warnings
from scipy.signal import find_peaks


def control(Temp, depths):
    """Validate the minimum depth requirements of a temperature profile.

    Method
    ----------
    Thermocline calculations require at least three measurements at distinct
    depths. This function performs these two checks before a profile is passed
    to a stratification algorithm. Invalid profiles produce a warning and a
    missing result instead of an arbitrary thermocline depth.

    Parameters
    ----------
    Temp : xarray.DataArray
        Temperature data containing a ``depth`` dimension.
    depths : array_like
        Measurement depths in metres.

    Returns
    -------
    int or float
        ``1`` for a valid profile, otherwise ``numpy.nan``.

    Warns
    -----
    UserWarning
        If fewer than three measurements are available or depths are repeated.

    Examples
    ----------
    >>> import xarray as xr
    >>> profile = xr.DataArray([20, 15, 10], dims="depth")
    >>> control(profile, [1, 2, 3])
    1
    """
    if Temp.sizes["depth"] < 3:
        warnings.warn(
            "At least 3 measurements are required"
        )
        return np.nan
    elif len(depths) != len(np.unique(depths)):
        warnings.warn("depths must be unique")
        return np.nan
    else:
        return 1


def to_xarray(Temp, depths=None, time=None):
    """Convert temperature profiles to PyLake's standard DataArray layout.

    Method
    ----------
    PyLake represents temperature as a two-dimensional array whose rows are
    profiles in time and whose columns are measurement depths. One-dimensional
    input is interpreted as one profile. Two-dimensional NumPy input is
    transposed when the depth axis is detected in the first dimension. Existing
    xarray coordinates are preserved.

    Parameters
    ----------
    Temp : array_like or xarray.DataArray
        One profile or a collection of profiles. Output order is always
        ``(time, depth)``.
    depths : array_like or None, default: None
        Measurement depths in metres. Required for non-xarray input.
    time : array_like or None, default: None
        Profile timestamps. Sequential integers are used when omitted.

    Returns
    -------
    Temp : xarray.DataArray
        Temperature data with ``time`` and ``depth`` dimensions.
    depths : numpy.ndarray
        Depth coordinate extracted from or assigned to ``Temp``.

    Raises
    ------
    ValueError
        If the depth dimension is absent or array shapes are incompatible.

    Examples
    --------
    >>> temp, depth = to_xarray([14.3, 14.0, 12.1], [1, 2, 3])
    >>> temp.shape
    (1, 3)
    """
    if isinstance(Temp, xr.DataArray):
        if "depth" not in Temp.dims:
            raise ValueError(
                "The input DataArray must contain a 'depth' dimension."
            )

        depths = Temp["depth"].to_numpy()

        if Temp.ndim == 1:
            Temp = Temp.expand_dims(time=[0])

        return Temp, depths

    if depths is None:
        raise ValueError(
            "depths must be provided when Temp is not an xarray.DataArray."
        )

    Temp = np.asarray(Temp)
    depths = np.asarray(depths)

    Temp = format_Temp(depths, Temp)

    if Temp.shape[1] != len(depths):
        raise ValueError(
            "Temperature data and depths have incompatible shapes."
        )

    if time is None:
        time = np.arange(Temp.shape[0])
    else:
        time = np.asarray(time)

    if len(time) != Temp.shape[0]:
        raise ValueError(
            "Time and temperature data have incompatible shapes."
        )

    Temp = xr.DataArray(
        Temp,
        dims=("time", "depth"),
        coords={
            "time": time,
            "depth": depths,
        },
    )

    return Temp, depths


def smooth_1D(Temp, smooth):
    """Smooth a one-dimensional profile with a Savitzky-Golay filter.

    Method
    ----------
    A local polynomial is fitted with :func:`scipy.signal.savgol_filter`.
    Unless custom settings are supplied, the window is the next odd integer
    close to one tenth of the series length, the polynomial order is at most
    three, and boundary values are extended using ``nearest`` mode.

    Parameters
    ----------
    Temp : array_like
        Values to smooth.
    smooth : bool or dict
        A dictionary may define ``window_size``, ``order``, and ``method``.
        Other truthy values use an odd window near one tenth of the profile
        length, polynomial order up to three, and ``nearest`` boundaries.

    Returns
    -------
    numpy.ndarray
        Smoothed values with the same shape as the input.
    """
    from scipy.signal import savgol_filter

    if type(smooth) == dict:
        window_size = smooth.get(
            "window_size",
            round_up_to_odd(len(Temp) / 10),
        )
        mode = smooth.get("method", "nearest")
        polyorder = min(3, window_size - 1)
        order = smooth.get("order", polyorder)

        new_Temp = savgol_filter(
            Temp,
            window_size,
            order,
            mode=mode,
        )
    else:
        window_size = round_up_to_odd(len(Temp) / 10)
        mode = "nearest"
        polyorder = min(3, window_size - 1)

        new_Temp = savgol_filter(
            Temp,
            window_size,
            polyorder,
            mode=mode,
        )

    return new_Temp


def smooth_temp(Temp, depths, smooth):
    """Smooth temperature along the depth dimension.

    Method
    ----------
    The Savitzky-Golay filter is applied independently to every temperature
    profile along its ``depth`` axis. Time coordinates and all non-depth
    dimensions are preserved. Smoothing can reduce biases caused by irregular
    sensor spacing, but it may also suppress narrow physical structures.

    Parameters
    ----------
    Temp : xarray.DataArray
        Temperature profiles containing a ``depth`` dimension.
    depths : array_like
        Depth coordinate, used to determine the default window length.
    smooth : bool or dict
        Optional Savitzky-Golay settings: ``window_size``, ``order``, and
        ``method``.

    Returns
    -------
    xarray.DataArray
        A copy of ``Temp`` with smoothed values and preserved coordinates.
    """
    from scipy.signal import savgol_filter

    if type(smooth) == dict:
        window_size = smooth.get(
            "window_size",
            round_up_to_odd(len(depths) / 10),
        )
        mode = smooth.get("method", "nearest")
        polyorder = min(3, window_size - 1)
        order = smooth.get("order", polyorder)
    else:
        window_size = round_up_to_odd(len(depths) / 10)
        mode = "nearest"
        polyorder = min(3, window_size - 1)
        order = polyorder

    axis = Temp.get_axis_num("depth")

    values = savgol_filter(
        Temp.data,
        window_size,
        order,
        axis=axis,
        mode=mode,
    )

    return Temp.copy(data=values)


def weighted_method(depths, rho, z_idx):
    """
    Refine a target depth between measurement depths.

    Method
    ----------
    A maximum density gradient initially locates the target between two
    sensors. The gradients immediately above and below that interval are then
    used to weight its two bounding depths. This produces a continuous depth
    estimate rather than forcing the result onto a sensor or interval midpoint.
    At the top and bottom boundaries, where both neighbouring gradients are not
    available, the midpoint of the adjacent sensors is returned.

    Parameters
    ----------
    depths : array_like
        Depth array retained for API compatibility. The coordinate stored in
        ``rho`` is used by the calculation.
    rho : xarray.DataArray
        Water density with a depth dimension.
    z_idx : xarray.DataArray
        Index of the maximum density gradient.

    Returns
    -------
    weighted_depth : xarray.DataArray
        Refined depth.

    Raises
    ------
    TypeError
        If ``rho`` is not an :class:`xarray.DataArray`.

    Notes
    -----
    Boundary maxima fall back to the midpoint of the adjacent sensors.

    References
    ----------
    Read, J. S. et al. (2011). Derivation of lake mixing and stratification
    indices from high-resolution lake buoy data. Environmental Modelling &
    Software, 26, 1325-1336.
    """

    if not isinstance(rho, xr.DataArray):
        raise TypeError("weighted_method expects rho to be an xarray.DataArray")

    depths = rho["depth"]

    drho_dz = (
        rho.diff("depth")
        / rho.depth.diff("depth")
    )

    mask_up = z_idx == 0
    mask_down = z_idx >= len(depths) - 2

    z_masked = z_idx.copy()

    z_masked = z_masked.where(
        ~mask_up,
        z_masked + 1,
    )

    z_masked = z_masked.where(
        ~mask_down,
        z_masked - 1,
    )

    drho = drho_dz.isel(
        depth=z_masked
    )

    drho_plus = drho_dz.isel(
        depth=z_masked + 1
    )

    drho_minu = drho_dz.isel(
        depth=z_masked - 1
    )

    Sdn = -(
        depths.isel(depth=z_masked + 1)
        - depths.isel(depth=z_masked)
    ) / (
        drho_plus - drho
    )

    Sup = (
        depths.isel(depth=z_masked)
        - depths.isel(depth=z_masked - 1)
    ) / (
        drho - drho_minu
    )

    weighted_depth = (
        depths.isel(depth=z_masked + 1)
        * (Sdn / (Sdn + Sup))
        + depths.isel(depth=z_masked)
        * (Sup / (Sdn + Sup))
    )

    midpoint = (
        depths.isel(depth=z_masked)
        + depths.isel(depth=z_masked + 1)
    ) / 2

    invalid_weight = (
        ~np.isfinite(Sdn)
        | ~np.isfinite(Sup)
        | ~np.isfinite(weighted_depth)
    )

    weighted_depth = weighted_depth.where(
        ~invalid_weight,
        midpoint,
    )

    weighted_depth = weighted_depth.where(
        ~mask_up,
        (depths[0] + depths[1]) / 2,
    )

    weighted_depth = weighted_depth.where(
        ~mask_down,
        (depths[-1] + depths[-2]) / 2,
    )

    try:
        weighted_depth = weighted_depth.drop_vars("depth")
    except (ValueError, KeyError):
        pass

    return weighted_depth


def find_peak_index(values, min_height, fallback_index):
    """Return the deepest peak above a threshold or a fallback index.

    Parameters
    ----------
    values : array_like
        One-dimensional density-gradient values.
    min_height : float
        Minimum accepted peak height.
    fallback_index : int
        Index returned when no qualifying peak exists.

    Returns
    -------
    int
        Index of the deepest qualifying peak.
    """
    values = np.asarray(values, dtype=float)
    locations, _ = find_peaks(values, height=min_height)

    if locations.size:
        return int(locations[-1])

    return int(fallback_index)

def check_bathy(Temp, bthA, bthD, depth):
    """Align temperature and bathymetry depth ranges.

    Temperature is extended with its nearest boundary value when bathymetry
    extends beyond the sampled profile. Bathymetry is extended with zero area
    when the temperature profile is deeper than the supplied lake geometry.

    Parameters
    ----------
    Temp : array_like
        Temperature profiles arranged as ``(time, depth)``.
    bthA, bthD : array_like
        Cross-sectional lake areas in square metres and corresponding depths
        in metres.
    depth : array_like
        Temperature measurement depths in metres.

    Returns
    -------
    tuple
        Adjusted ``Temp``, bathymetric areas, bathymetric depths, and profile
        depths.
    """
    numD = Temp.shape[1] - 1

    if max(bthD) > depth[numD]:
        Temp = np.append(
            Temp,
            Temp[:, numD],
        )

        depth = np.append(
            depth,
            max(bthD),
        )

    elif max(bthD) < depth[numD]:
        bthD = np.append(
            bthD,
            depth[numD],
        )

        bthA = np.append(
            bthA,
            0,
        )

    if min(bthD) < depth[0]:
        Temp = np.hstack(
            (
                Temp[:, 0].reshape(-1, 1),
                Temp,
            )
        )

        depth = np.append(
            np.min(bthD),
            depth,
        )

    return Temp, bthA, bthD, depth


def format_Temp(depths, Temp):
    """Arrange temperature input as rows of time and columns of depth.

    One-dimensional input becomes a single-row profile. Two-dimensional input
    is transposed when its first axis matches the depth coordinate.

    Parameters
    ----------
    depths : array_like
        Measurement depths used to infer the profile axis.
    Temp : numpy.ndarray
        One- or two-dimensional temperature data.

    Returns
    -------
    numpy.ndarray
        Temperature array with shape ``(time, depth)`` when orientation can be
        inferred.
    """
    if Temp.ndim == 2:
        if Temp.shape[0] == depths.shape[0]:
            Temp = Temp.T

    elif Temp.ndim == 1:
        Temp = Temp.reshape(-1, 1).T

    return Temp


def find_nearest_index(old_depths, target_depth):
    """Find indices of the sampled depths nearest to target depths.

    Parameters
    ----------
    old_depths, target_depth : array_like
        Available and requested depths in metres.

    Returns
    -------
    numpy.ndarray
        Index in ``old_depths`` for each target depth.
    """
    depth_index = np.argmin(
        np.abs(
            target_depth
            - old_depths.reshape(-1, 1)
        ),
        axis=0,
    )

    return depth_index


def find_nearest(old_depths, target_depth):
    """Return sampled depths nearest to requested target depths.

    Missing target depths remain missing in the returned array.

    Parameters
    ----------
    old_depths : array_like
        Available measurement depths in metres.
    target_depth : array_like
        Requested depths in metres.

    Returns
    -------
    numpy.ndarray
        Nearest depths, with the shape of ``target_depth``.
    """
    depth_index = find_nearest_index(
        old_depths,
        target_depth,
    )

    nearest_depth = old_depths[
        depth_index
    ]

    nearest_depth = set_nan(
        target_depth,
        nearest_depth,
    )

    return nearest_depth


def set_nan(vec1, vec2):
    """Copy missing-value positions from one vector to another.

    Parameters
    ----------
    vec1 : array_like
        Source vector whose ``NaN`` positions are copied.
    vec2 : array_like
        Values to update.

    Returns
    -------
    array_like
        ``vec2`` with missing values propagated from ``vec1``.
    """
    source = np.asarray(vec1)
    target = np.asarray(vec2, dtype=float).copy()
    missing = np.isnan(source)

    if source.ndim == 0:
        if bool(missing):
            return np.nan
        return vec2

    target[missing] = np.nan
    return target


def round_up_to_odd(f):
    """Round a number upward to the next odd integer.

    Parameters
    ----------
    f : float
        Value to round.

    Returns
    -------
    int
        Smallest odd integer produced by rounding ``f`` upward.
    """
    return int(
        np.ceil(f) // 2 * 2 + 1
    )

def depth_filter(depth, run_length=20, index=False):
    """Extract the monotonic downcast portion of a depth series.

    Non-finite observations are removed, data after the deepest observation are
    discarded, and reversals shallower than the running maximum are excluded.

    Method
    ----------
    CTD files may contain measurements taken before the probe enters the water,
    short upward movements during descent, and the final upcast. The algorithm
    removes non-finite depths, stops at the deepest observation, locates a
    sustained descending run, and retains only observations that do not move
    above the deepest depth already reached.

    Parameters
    ----------
    depth : array_like
        Recorded depths in metres, typically from a CTD cast.
    run_length : int, default: 20
        Consecutive non-decreasing differences used to locate the downcast.
    index : bool, default: False
        Return indices into the original array instead of depth values.

    Returns
    -------
    numpy.ndarray
        Filtered depths or their original integer indices.

    Examples
    ----------
    >>> depth_filter([0, 1, 2, 1.5, 3], run_length=2).tolist()
    [0.0, 1.0, 2.0, 3.0]
    """
    depth = np.asarray(depth, dtype=float)

    valid = np.isfinite(depth)
    original_index = np.flatnonzero(valid)
    z = depth[valid]

    if len(z) == 0:
        return original_index if index else z

    deepest = np.argmax(z)

    z = z[:deepest + 1]
    original_index = original_index[:deepest + 1]

    if len(z) <= run_length:
        return original_index if index else z

    dz = np.diff(z)

    start = 0

    for i in range(len(dz) - run_length + 1):
        if np.all(dz[i:i + run_length] >= 0):
            start = i
            break

    z = z[start:]
    original_index = original_index[start:]

    keep = np.ones(len(z), dtype=bool)

    max_depth = z[0]

    for i in range(1, len(z)):
        if z[i] < max_depth:
            keep[i] = False
        else:
            max_depth = z[i]

    z = z[keep]
    original_index = original_index[keep]

    if index:
        return original_index

    return z


def depth_average(depth, values):
    """Average observations recorded at identical depths.

    Non-finite depth-value pairs are removed before grouping.

    Method
    ----------
    Repeated CTD measurements are grouped by their exact depth value. The mean
    of the finite observations in each group is returned together with the
    sorted unique depths. This produces one value per depth before a profile is
    interpolated or passed to a lake-stratification function.

    Parameters
    ----------
    depth, values : array_like
        Recorded depths and corresponding observations. Both are converted to
        floating-point arrays and must be broadcastable to the same mask.

    Returns
    -------
    unique_depth : numpy.ndarray
        Sorted unique finite depths.
    average : numpy.ndarray
        Mean value at each unique depth.

    Examples
    ----------
    >>> depth, value = depth_average([1, 1, 2], [10, 12, 8])
    >>> depth.tolist(), value.tolist()
    ([1.0, 2.0], [11.0, 8.0])
    """
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = (
        np.isfinite(depth)
        & np.isfinite(values)
    )

    depth = depth[valid]
    values = values[valid]

    unique_depth, inverse = np.unique(
        depth,
        return_inverse=True
    )

    average = np.array([
        values[inverse == i].mean()
        for i in range(len(unique_depth))
    ])

    return unique_depth, average


def center_buoyancy(Temp, depth):
    """Calculate the depth-weighted centre of positive buoyancy frequency.

    Method
    ----------
    The squared buoyancy frequency is calculated between adjacent temperature
    measurements. Negative values and missing observations are discarded, then
    the remaining interval depths are averaged using buoyancy frequency as the
    weight. The result identifies the vertical centre of stable stratification.

    Parameters
    ----------
    Temp, depth : array_like
        Temperature in degrees Celsius and corresponding depth in metres.

    Returns
    -------
    float
        Centre of buoyancy in metres, or ``numpy.nan`` when no positive finite
        buoyancy frequency is present.
    """
    from .pylake import buoyancy_freq

    n2 = buoyancy_freq(
        Temp,
        depth
    )

    values = np.asarray(
        n2
    ).squeeze()

    buoyancy_depth = np.asarray(
        n2["avg_depth"]
    )

    values = np.maximum(
        values,
        0
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(buoyancy_depth)
    )

    values = values[valid]
    buoyancy_depth = buoyancy_depth[valid]

    if np.sum(values) == 0:
        return np.nan

    return np.sum(
        buoyancy_depth * values
    ) / np.sum(values)



def layer_average(top, bottom, values, depth, bthA, bthD, dz=0.1):
    """Calculate a volume-weighted mean over a lake layer.

    Values and lake area are linearly interpolated onto a regular vertical grid
    and weighted by the volume represented by each slice.

    Method
    ----------
    The profile and the lake cross-sectional area are linearly interpolated at
    a constant vertical resolution ``dz``. Each value is multiplied by the
    volume of its horizontal lake slice. Dividing the sum of weighted values by
    total layer volume gives the volume-weighted mean.

    Parameters
    ----------
    top, bottom : float
        Upper and lower layer boundaries in metres.
    values, depth : array_like
        Profile values and measurement depths.
    bthA, bthD : array_like
        Lake areas and their bathymetric depths.
    dz : float, default: 0.1
        Vertical integration step in metres.

    Returns
    -------
    float
        Volume-weighted layer mean.

    Raises
    ------
    ValueError
        If boundaries are reversed or profile arrays have different lengths.

    Examples
    ----------
    >>> value = layer_average(0, 2, [20, 15, 10], [0, 1, 2],
    ...                       [100, 80, 0], [0, 1, 2], dz=1)
    >>> round(float(value), 6)
    17.777778
    """
    values = np.asarray(values, dtype=float)
    depth = np.asarray(depth, dtype=float)
    bthA = np.asarray(bthA, dtype=float)
    bthD = np.asarray(bthD, dtype=float)

    if top > bottom:
        raise ValueError("bottom depth must be greater than top")

    if len(values) != len(depth):
        raise ValueError("values and depth must have the same length")

    if depth[-1] < bottom:
        values = np.append(
            values,
            values[-1]
        )

        depth = np.append(
            depth,
            bottom
        )

    if np.max(bthD) < bottom:
        bthD = np.append(
            bthD,
            bottom
        )

        bthA = np.append(
            bthA,
            0
        )

    layer_depth = np.arange(
        top,
        bottom + dz,
        dz
    )

    layer_values = np.interp(
        layer_depth,
        depth,
        values
    )

    layer_area = np.interp(
        layer_depth,
        bthD,
        bthA
    )

    return np.sum(
        layer_area
        * layer_values
        * dz
    ) / np.sum(
        layer_area
        * dz
    )


def layer_density(
    top,
    bottom,
    Temp,
    depth,
    bthA,
    bthD,
    sal=0,
    dz=0.1
):
    """Calculate volume-weighted water density within a lake layer.

    ``Temp`` is converted to density using :func:`pylake.water_density`, then
    integrated between ``top`` and ``bottom`` using bathymetric area weights.

    Method
    ----------
    Temperature and salinity are first converted to water density with the
    PyLake density equation. :func:`layer_average` then integrates density over
    the selected layer using the lake bathymetry.

    Parameters
    ----------
    top, bottom : float
        Layer boundaries in metres.
    Temp, depth : array_like
        Temperature in degrees Celsius and measurement depths.
    bthA, bthD : array_like
        Bathymetric areas and depths.
    sal : float or array_like, default: 0
        Salinity in practical salinity units.
    dz : float, default: 0.1
        Vertical integration step in metres.

    Returns
    -------
    float
        Volume-weighted density.
    """
    from .pylake import water_density

    Temp = np.asarray(
        Temp,
        dtype=float
    )

    sal = np.asarray(
        sal,
        dtype=float
    )

    if sal.ndim == 0:
        sal = np.full_like(
            Temp,
            sal
        )

    density = water_density(
        Temp,
        sal
    )

    return layer_average(
        top,
        bottom,
        density,
        depth,
        bthA,
        bthD,
        dz=dz
    )



def layer_temperature(
    top,
    bottom,
    Temp,
    depth,
    bthA,
    bthD,
    dz=0.1
):
    """Calculate volume-weighted temperature within a lake layer.

    Method
    ----------
    Temperature is interpolated vertically and weighted by bathymetric slice
    volume through :func:`layer_average`.

    Parameters
    ----------
    top, bottom : float
        Upper and lower layer boundaries in metres.
    Temp, depth : array_like
        Water temperature in degrees Celsius and measurement depths in metres.
    bthA, bthD : array_like
        Cross-sectional lake areas and corresponding bathymetric depths.
    dz : float, default: 0.1
        Vertical integration step in metres.

    Returns
    -------
    float
        Volume-weighted layer temperature in degrees Celsius.
    """
    return layer_average(
        top,
        bottom,
        Temp,
        depth,
        bthA,
        bthD,
        dz=dz
    )


def whole_lake_temperature(
    Temp,
    depth,
    bthA,
    bthD,
    dz=0.1
):
    """Calculate volume-weighted temperature over the full lake depth.

    Method
    ----------
    This is a convenience wrapper around :func:`layer_temperature`, with the
    upper boundary fixed at the surface and the lower boundary fixed at maximum
    bathymetric depth.

    Parameters
    ----------
    Temp, depth : array_like
        Water temperature in degrees Celsius and measurement depths in metres.
    bthA, bthD : array_like
        Cross-sectional lake areas and corresponding bathymetric depths.
    dz : float, default: 0.1
        Vertical integration step in metres.

    Returns
    -------
    float
        Whole-lake volume-weighted temperature in degrees Celsius.
    """
    return layer_temperature(
        0,
        np.max(bthD),
        Temp,
        depth,
        bthA,
        bthD,
        dz=dz
    )


def epi_temperature(
    Temp,
    depth,
    bthA,
    bthD,
    meta_top,
    dz=0.1
):
    """Calculate volume-weighted epilimnion temperature.

    The epilimnion is integrated from the surface to ``meta_top``.

    Method
    ----------
    The calculation calls :func:`layer_temperature` between 0 m and the top of
    the metalimnion.

    Parameters
    ----------
    Temp, depth : array_like
        Water temperature and measurement depths.
    bthA, bthD : array_like
        Cross-sectional lake areas and corresponding bathymetric depths.
    meta_top : float
        Bottom boundary of the epilimnion in metres.
    dz : float, default: 0.1
        Vertical integration step in metres.

    Returns
    -------
    float
        Volume-weighted epilimnion temperature in degrees Celsius.
    """
    return layer_temperature(
        0,
        meta_top,
        Temp,
        depth,
        bthA,
        bthD,
        dz=dz
    )


def hypo_temperature(
    Temp,
    depth,
    bthA,
    bthD,
    meta_bottom,
    dz=0.1
):
    """Calculate volume-weighted hypolimnion temperature.

    The hypolimnion is integrated from ``meta_bottom`` to maximum lake depth.

    Method
    ----------
    The calculation calls :func:`layer_temperature` between the bottom of the
    metalimnion and maximum bathymetric depth.

    Parameters
    ----------
    Temp, depth : array_like
        Water temperature and measurement depths.
    bthA, bthD : array_like
        Cross-sectional lake areas and corresponding bathymetric depths.
    meta_bottom : float
        Upper boundary of the hypolimnion in metres.
    dz : float, default: 0.1
        Vertical integration step in metres.

    Returns
    -------
    float
        Volume-weighted hypolimnion temperature in degrees Celsius.
    """
    return layer_temperature(
        meta_bottom,
        np.max(bthD),
        Temp,
        depth,
        bthA,
        bthD,
        dz=dz
    )


def ustar(wind_speed, wind_height, average_epi_density):
    """Calculate water-side friction velocity from wind forcing.

    Method
    ----------
    Wind speed is adjusted to a standard height of 10 m with a logarithmic
    wind profile when necessary. Wind stress is calculated from air density,
    wind speed, and a piecewise drag coefficient. Water-side friction velocity
    is the square root of wind stress divided by epilimnion density.

    Parameters
    ----------
    wind_speed : array_like
        Wind speed in metres per second.
    wind_height : float
        Wind measurement height in metres. Speeds measured away from 10 m are
        adjusted logarithmically.
    average_epi_density : array_like
        Mean epilimnion water density in kilograms per cubic metre.

    Returns
    -------
    numpy.ndarray
        Friction velocity in metres per second.

    Notes
    -----
    Air density is fixed at 1.2 kg m-3 and the drag coefficient is 0.001 below
    5 m s-1 and 0.0015 otherwise.

    Examples
    ----------
    >>> round(float(ustar(5, 10, 1000)), 6)
    0.006708
    """
    wind_speed = np.asarray(
        wind_speed,
        dtype=float
    )

    average_epi_density = np.asarray(
        average_epi_density,
        dtype=float
    )

    rho_air = 1.2
    von_karman = 0.4

    cd = np.where(
        wind_speed < 5,
        0.001,
        0.0015
    )

    if wind_height != 10:
        wind_speed = wind_speed / (
            1
            - np.sqrt(cd)
            / von_karman
            * np.log(10 / wind_height)
        )

    tau = (
        cd
        * rho_air
        * wind_speed**2
    )

    return np.sqrt(
        tau
        / average_epi_density
    )
