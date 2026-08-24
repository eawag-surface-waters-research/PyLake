import numpy as np
import xarray as xr
import warnings


def control(Temp, depths):
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

    Parameters
    ----------
    depths : array_like
        Depth array.
    rho : xarray.DataArray
        Water density with a depth dimension.
    z_idx : xarray.DataArray
        Index of the maximum density gradient.

    Returns
    -------
    weighted_depth : xarray.DataArray
        Refined depth.
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

    hplus = (
        depths.isel(depth=z_masked)
        - depths.isel(depth=z_masked + 2)
    ) / 2

    hminu = (
        depths.isel(depth=z_masked - 1)
        - depths.isel(depth=z_masked + 1)
    ) / 2

    drho = drho_dz.isel(
        depth=z_masked
    )

    drho_plus = drho_dz.isel(
        depth=z_masked + 1
    )

    drho_minu = drho_dz.isel(
        depth=z_masked - 1
    )

    Dplus = hplus / (
        drho - drho_plus
    )

    Dminu = hminu / (
        drho - drho_minu
    )

    weighted_depth = (
        depths.isel(depth=z_masked + 1)
        * (Dplus / (Dminu + Dplus))
        + depths.isel(depth=z_masked)
        * (Dminu / (Dminu + Dplus))
    )

    mask_inf = (
        np.isinf(Dplus / (Dminu + Dplus))
        & np.isinf(Dminu / (Dminu + Dplus))
    )

    weighted_depth = weighted_depth.where(
        ~mask_up,
        (depths[0] + depths[1]) / 2,
    )

    weighted_depth = weighted_depth.where(
        ~mask_down,
        (depths[-1] + depths[-2]) / 2,
    )

    weighted_depth = weighted_depth.where(
        ~mask_inf,
        np.nan,
    )

    try:
        weighted_depth = weighted_depth.drop_vars("depth")
    except (ValueError, KeyError):
        pass

    return weighted_depth

def check_bathy(Temp, bthA, bthD, depth):
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
    if Temp.ndim == 2:
        if Temp.shape[0] == depths.shape[0]:
            Temp = Temp.T

    elif Temp.ndim == 1:
        Temp = Temp.reshape(-1, 1).T

    return Temp


def find_nearest_index(old_depths, target_depth):
    depth_index = np.argmin(
        np.abs(
            target_depth
            - old_depths.reshape(-1, 1)
        ),
        axis=0,
    )

    return depth_index


def find_nearest(old_depths, target_depth):
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
    NaN = np.isnan(vec1)

    if any(NaN):
        if len(NaN) == 1:
            vec2 = np.array([np.nan])
        else:
            vec2[NaN] = np.nan

    return vec2


def round_up_to_odd(f):
    return int(
        np.ceil(f) // 2 * 2 + 1
    )

def depth_filter(depth, run_length=20, index=False):
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

