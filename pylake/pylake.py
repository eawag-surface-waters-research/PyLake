import numpy as np
from .functions import *
import gsw
from scipy.interpolate import interp1d
import warnings
from scipy.signal import find_peaks,savgol_filter
import xarray as xr
    
def thermocline(Temp, depth=None, time=None, s=0.2, mixed_cutoff=1, smooth=False):
    '''
    Calculate the thermocline depth from one or various temperature profiles.
    It uses the method of the maximum gradient, the results can be interpreted
    as a diurnal thermocline (see pylake.seasonal_thermocline for the seasonal).

    Method
    ----------
    The thermocline is calculated using the maximum gradient of density.
    If the temperature profile have a variability in depth resolution, it is recommended 
    to smooth the profile to avoid resolution influence on the algorithm.
    Once the maximum gradient of density is found, a special technique to refine the 
    thermocline depth is used and presented in (Read et al., 2011).

    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depth : array_like
        a numeric vector corresponding to the depth (in m) of the Temp
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    mixed_cutoff : scalar, default: 1
        The difference between the maximum and minimum of the
        temperature profile should be higher than this cutoff.
    smooth : bool, default: False
        Smooth the curve following the scipy savgol filter (window size: 1/10 of the 
        depth length, order:3, method=nearest)
        Smoothing is recommended when the thermocline is located at a lower resolution 
        sensors (sensors are more spaced at the thermocline)

    Returns 
    ----------
    thermoD: array_like, scalar
        thermocline depth (m)
    thermoInd: array_like, scalar
        thermocline index corresponding to the thermocline depth
    
    Examples
    ----------
    >>>     import pylake
    ...     temp = temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
    ...     depth = depth = np.array([1,2,3,4,5,6,7,8])
    ...     thermo, thermoInd = pylake.thermocline(temp,depth)
    ...     print(f"thermocline depth: {thermo}\n")
    ...     print(f"thermocline depth index: {thermoInd}\n")
    ...     thermocline depth: 2.878790569183019
    ...     thermocline depth index: 2
    '''

    Temp, depth = to_xarray(Temp, depth,time)

    is_not_significant = Temp.max('depth')-Temp.min('depth')<mixed_cutoff
    if smooth:
        time = Temp.time
        Temp = smooth_temp(Temp, depth, smooth)
        Temp, depth = to_xarray(Temp, depth,time)

    rhoVar = water_density(Temp,s)
    drho_dz = rhoVar.diff('depth')/rhoVar.depth.diff('depth')
    inf_mask = np.isinf(drho_dz)
    drho_dz = drho_dz.where(~inf_mask, np.nan)
    thermoInd = drho_dz.fillna(-999).argmax('depth')

    thermoD = weighted_method(depth, rhoVar, thermoInd)

    thermoInd = abs(thermoD-Temp.depth).fillna(999).argmin('depth')

    thermoD = thermoD.where(~is_not_significant, np.nan)

    if thermoD.size==1:
        thermoD = thermoD.values.item()
        thermoInd = thermoInd.data.item()
    return thermoD, thermoInd



def seasonal_thermocline(Temp, depth=None, time=None, s=0.2, mixed_cutoff=1, Smin=0.1, seasonal_smoothed=True, smooth=False):
    '''
    Calculate depth of the thermocline from a temperature profile.
    
    Method
    ---------
    The seasonal thermocline uses the find_peaks from scipy to find the first 
    local maximum (higher than a certain threshold, based on Smin) starting 
    from the bottom of the profile.
    By default, if no peak is found, the seasonal thermocline is set equal to 
    the diurnal thermocline. 
    The same refining technique is used.
    A savgol filter (special moving averaged filter) is used to smooth the data 
    and discard extremums.
    The seasonal is set to be more or equal to the diurnal thermocline.
    Keep in mind that if the seasonal thermocline is often assimilated 
    with the diurnal, with the exception that the seasonal is artificially smoothed.

    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depth : array_like
        a numeric vector corresponding to the depth of the temperature.
        Depth is defined as positive, and is minimum at the surface. 
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    mixed_cutoff : scalar, default: 1
        The difference between the maximum and minimum of the
        temperature profile should be higher than this cutoff.
    Smin : float, default: 0.1 kg/m3/m
        Optional parameter defining minimum density gradient for
        thermocline. Threshold for the peak height of the scipy.signal.find_peaks(...).
    seasonal_smoothed: bool, default: True
        Smooth the seasonal thermocline on the entire time serie. The smooth depends on the time serie length.
    smooth : bool, default: False
        Smooth the curve following the scipy savgol filter (window size: 1/10 of the 
        depth length, order:3, method=nearest)
        Smoothing is recommended when the thermocline is located at a lower resolution 
        sensors (sensors can be more spaced at the thermocline, resulting in a bias).
    Returns 
    ----------
    thermoD: array_like
        thermocline depth (m)
    thermoInd: array_like
        thermocline index corresponding to the thermocline depth
    
    Examples
    ----------
    >>>     import pylake
    ...     temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
    ...     depth = np.array([1,2,3,4,5,6,7,8])
    ...     Sthermo, SthermoInd = pylake.seasonal_thermocline(temp,depth)
    ...     print(f"Seasonal thermocline depth: {Sthermo}\n")
    ...     print(f"Seasonal thermocline depth index: {SthermoInd}\n")
    ...     Seasonal thermocline depth: 6.510728817460102
    ...     Seasonal thermocline depth index: 6
    '''

    Temp, depth = to_xarray(Temp, depth, time)

    is_not_significant = (Temp.max('depth') - Temp.min('depth')) < mixed_cutoff

    time = Temp.time
    rhoVar = water_density(Temp,s)
    drho_dz = rhoVar.diff('depth')/rhoVar.depth.diff('depth')

    # It seems that this is not used...
    dRhoCut = Smin*np.ones(time.size)

    thermoD, thermoInd = thermocline(Temp, depth, smooth=smooth, mixed_cutoff=mixed_cutoff)        

    def process_peaks(arr, Smin, thermoInd):
        #If no peak is found, set SthermoInd to thermoInd
        arr = arr.copy()
        locs, peaks = find_peaks(arr, height=Smin)
        if (len(locs)and(~np.isnan(locs[-1]))):
            SthermoInd = locs[-1].astype(int)
        else:
            SthermoInd = thermoInd
        return SthermoInd

    SthermoInd = xr.apply_ufunc(process_peaks, drho_dz, Smin, thermoInd, input_core_dims=[['depth'],[],[]],output_core_dims=[[]], vectorize=True)

    SthermoD = weighted_method(depth, rhoVar, SthermoInd)
    SthermoD = SthermoD.where(~is_not_significant, np.nan)

    #Compare the seasonal and the diurnal thermocline, seasonal should be at higher depth than the diurnal, if not, both are set equal.
    mask = (SthermoD<thermoD)
    SthermoD = SthermoD.where(~mask, thermoD)
    SthermoInd = SthermoInd.where(~mask, thermoInd)

    SthermoInd = abs(SthermoD-Temp.depth).fillna(999).argmin('depth')

    if len(SthermoD)!=1:
        if seasonal_smoothed:
            SthermoD = smooth_1D(SthermoD, seasonal_smoothed)
            SthermoD = xr.DataArray(SthermoD, coords={'time':time})
    else:
        SthermoD = SthermoD.values.item()
        SthermoInd = SthermoInd.values.item()
    return SthermoD, SthermoInd



def robust_thermocline(Temp, depth=None, time=None, s=0.2, mixed_cutoff=1):
    '''
    Calculate the thermocline depth from one or various temperature profiles in a more robust way than gradient methods.
    This is especially useful, if the depth resolution of the data is not very high.

    Method
    ----------
    Thermocline depth is defined as the middle point between the upper and the lower bound of
    the metalimnion. The middle point is calculated by taking the depth where temperature is
    the average between the temperatures at the upper and lower bound of the metalimnion

    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depth : array_like
        a numeric vector corresponding to the depth (in m) of the Temp
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    mixed_cutoff : scalar, default: 1
        The difference between the maximum and minimum of the
        temperature profile should be higher than this cutoff.

    Returns 
    ----------
    thermoD: array_like, scalar
        thermocline depth (m)
    
    Examples
    ----------
    >>>     import pylake
    ...     temp = temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
    ...     depth = depth = np.array([1,2,3,4,5,6,7,8])
    ...     thermo = pylake.robust_thermocline(temp,depth)
    ...     print(f"thermocline depth: {thermo}\n")
    ...     thermocline depth: 6.50536441
    '''

    Temp, depth = to_xarray(Temp, depth, time)

    # Ensure Temp is (time, depth)
    Temp = Temp.transpose('time', 'depth')

    # Compute significance
    is_not_significant = Temp.max('depth') - Temp.min('depth') < mixed_cutoff

    # Metalimnion boundaries
    epi_depth, hypo_depth = metalimnion(Temp, depth, slope=0.5, slope_calc='relative',
                                        seasonal=True, mixed_cutoff=mixed_cutoff, smooth=True, s=s)

    # Force epi_depth and hypo_depth to 1D numpy arrays
    epi_depth = np.atleast_1d(epi_depth)
    hypo_depth = np.atleast_1d(hypo_depth)

    # Initialize output
    thermoD = np.full(epi_depth.shape[0], np.nan)

    for i in range(epi_depth.shape[0]):
        temp_profile = Temp.values[i, :]  # (depths)
        if np.isnan(epi_depth[i]) or np.isnan(hypo_depth[i]):
            continue
        epi_temp = np.interp(epi_depth[i], depth, temp_profile)
        hypo_temp = np.interp(hypo_depth[i], depth, temp_profile)
        meta_temp = (epi_temp + hypo_temp) / 2
        thermoD[i] = crossing_depth(depth, temp_profile, meta_temp, epi_depth[i], hypo_depth[i])

    # Apply significance mask
    thermoD = np.where(~is_not_significant.values, thermoD, np.nan)

    if thermoD.size == 1:
        thermoD = thermoD[0]
    return thermoD


def metalimnion(Temp, depth=None, slope=0.25, slope_calc="relative", seasonal=False, mixed_cutoff=1, smooth=False, s=0.2):
    '''
    Calculates the top and bottom depth of the metalimnion in a stratified
    lake. The metalimnion is defined as the water stratum in a stratified lake
    with the steepest thermal gradient and is demarcated by the bottom of the
    epilimnion and top of the hypolimnion.
    
    Method
    ----------
    
    Parameters
    ----------
    Temp :  array_like
        a numeric vector of water temperature in degrees C
    depth : array_like
        a numeric vector corresponding to the depth (in m) of the Temp
        measurements. Depth is defined as positive, and is minimum at the surface. 
    slope : scalar, str, default: 0.1
        a numeric vector corresponding to the minimum slope. Can be set to "relative", if it's the case,
        the threshold will be 10% of the max slope density gradient.
    seasonal : bool, default: False
        Calculates the metalimnion based on the seasonal thermocline if set to True.
    mixed_cutoff : scalar, default: 1
        A cutoff (deg C) where below this threshold, thermo.depth and meta.depth are 
        not calculated (NaN is returned). Defaults to 1 deg C.
    smooth : bool, default: False
        Smooth the curve following the scipy savgol filter (window size: 1/5 of the 
        depth length, order:3, method=nearest)
        Smoothing is recommended when the thermocline is located at a lower resolution 
        sensors (sensors are more spaced at the thermocline)
    s : array_like, default : 0.2
        Salinity of the water column in PSU
    thermocline_output : bool, default : False
        Return the calculated thermocline depth if set to True 

    Returns 
    ----------
    epilimnion : array_like, scalar
        A numeric vector of the epilimnion depth.
        Returns the thermocline depth if no epilimnion depth is found
    hypolimnion : array_like, scalar
        A numeric vector of the hypolimnion depth.
        Returns the thermocline depth if no hypolimnion depth is found
    
    See also
    ----------
    pylake.thermocline

    Examples
    ----------
    >>>     import pylake
    ...     temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
    ...     depth = np.array([1,2,3,4,5,6,7,8])
    ...     epilimnion, hypolimnion = pylake.metalimnion(temp, depth)
    ...     print(f"Epilimnion: {epilimnion} \n Hypolimnion: {hypolimnion}.")
    Epilimnion: 2.5
    Hypolimnion: 3.5

    References
    ----------
    Wetzel, R. G. 2001. Limnology: Lake and River Ecosystems, 3rd ed. Academic Press.'''

    Temp, depth = to_xarray(Temp, depth)
    
    if smooth:
        time = Temp.time
        Temp = smooth_temp(Temp, depth, smooth)
        Temp, depth = to_xarray(Temp, depth, time)

    if seasonal:
        thermoD, thermoInd = seasonal_thermocline(Temp, depth, mixed_cutoff=mixed_cutoff, smooth=False, seasonal_smoothed=False)
    else:
        thermoD, thermoInd = thermocline(Temp, depth, mixed_cutoff=mixed_cutoff, smooth=False)

    #thermoD, thermoInd = list(map(np.asanyarray, (thermoD, thermoInd)))
    Temp["thermoInd"] = thermoInd
    Temp["thermoD"] = thermoD

    rhoVar = water_density(Temp,s)
    drho_dz = rhoVar.diff('depth')/rhoVar.depth.diff('depth')
    drho_dz["depth"] = [(a+b)/2 for a,b in zip(depth, depth[1:])]

    drho_dz["thermoInd"] = abs(drho_dz["thermoD"]-drho_dz["depth"]).fillna(999).argmin('depth')
    mark =  drho_dz["depth"]-drho_dz["thermoD"]

    if slope_calc == "relative":
        slope = drho_dz.max('depth') * slope
    elif slope_calc == "absolute":
        slope = slope

    cond = drho_dz<slope

    mark_value = mark.where(~cond)
    e_mark = mark_value.where(mark_value<0)
    h_mark = mark_value.where(mark_value>0)
    
    hyp_idx = h_mark.fillna(999).diff('depth').argmax('depth')
    hyp_depth = h_mark.isel(depth=hyp_idx).depth

    e_marked_rev = abs(e_mark)[::-1]
    ep_idx_rev = e_marked_rev.fillna(999).diff('depth').argmax(dim='depth')
    ep_depth = e_marked_rev.isel(depth=ep_idx_rev).depth

    only_NaN_h_mark = (~np.isnan(h_mark)).sum('depth')>0
    only_NaN_e_mark = (~np.isnan(e_mark)).sum('depth')>0
    epi_depth = ep_depth.where(only_NaN_e_mark, ep_depth["thermoD"])
    hypo_depth = hyp_depth.where(only_NaN_h_mark, hyp_depth["thermoD"])

    hypo_depth_filt = hypo_depth.where(hypo_depth>hypo_depth["thermoD"],hypo_depth["thermoD"])
    epi_depth_filt = epi_depth.where(epi_depth<epi_depth["thermoD"],epi_depth["thermoD"])
    
    if epi_depth_filt.size==1:
        epi_depth_filt = epi_depth_filt.values.item()
        hypo_depth_filt = hypo_depth_filt.values.item()
    return epi_depth_filt, hypo_depth_filt

def mixed_layer(Temp, depth=None, s=0.2, threshold=None, *, method):
    '''
    Calculates the mixed layer depth by the difference method, based on density or temperature (method is required).
    The depth of the mixed layer is the shallowest depth where the difference to the surface exceeds a threshold:
    - method="density": density at depth minus surface density > threshold (kg/m3, default 0.01)
    - method="temperature": absolute temperature difference to the surface > threshold (degC, default 0.02); the absolute value
      also catches inverse (winter) stratification
    The depth is interpolated linearly between the two depths around the threshold. If the threshold is never
    exceeded, the deepest depth is returned.
    A density threshold corresponds to very different temperature differences depending on the temperature
    (0.001 kg/m3: ~0.005 degC at 20 degC, ~0.06 degC at 5 degC, infinite at 4 degC), a temperature threshold does not.
    Surface values might be NaN. If it's the case, the second depth is used as surface reference.

    Parameters
    -----------
    Temp : array_like
        Temperature matrix (degrees)
    depth : array_like
        depth vector (m)
    s   : array_like
        Salinity matrix (psu), only used for method="density"
    threshold : scalar, optional
        threshold for the mixed layer detection (kg/m3 for method="density", default 0.01;
        degC for method="temperature", default 0.02)
    method : str, required (keyword)
        "density" or "temperature"

    Returns
    -----------
    hML array_like
        Mixed layer depth (m)

    Example
    ----------
    >>>    temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
    ...    depth = np.array([1,2,3,4,5,6,7,8])
    ...    hML = pylake.mixed_layer(temp, depth, threshold=0.5, method="temperature")
    ...    print(hML)
    ...    2.1052631578947363
    '''
    Temp, depth = to_xarray(Temp, depth)
    Temp = Temp.transpose('time', 'depth')
    if method == "density":
        var = water_density(Temp, s)
        threshold = 0.01 if threshold is None else threshold
    elif method == "temperature":
        var = Temp.copy()
        threshold = 0.02 if threshold is None else threshold
    else:
        raise ValueError('method must be "density" or "temperature"')

    surf = var.isel(depth=0).copy()

    #If surface sensor is NaN, check the second (could potentially be iterated further down)
    NaN = np.where(np.isnan(surf))[0]
    surf[NaN] = var.isel(time=NaN, depth=1)

    diff = var - surf
    if method == "temperature":
        diff = abs(diff)
    # excess over the threshold; the mixed layer ends where it first (from the surface) becomes positive.
    # (The minimum of the positive excess is not used: with small inversions it can be deeper.)
    excess = (diff - threshold).values
    z = np.asarray(var.depth.values, dtype=float)
    exceeds = excess > 0
    hML = np.full(excess.shape[0], z[-1])     # deepest depth if the threshold is never exceeded
    for i in np.where(exceeds.any(axis=1))[0]:
        j = np.argmax(exceeds[i])
        if j == 0 or np.isnan(excess[i, j - 1]):
            hML[i] = z[j]
        else:
            hML[i] = z[j - 1] + (-excess[i, j - 1]) / (excess[i, j] - excess[i, j - 1]) * (z[j] - z[j - 1])

    hML = xr.DataArray(hML, coords={'time': var.time})
    if hML.size == 1:
        hML = hML.values.item()
    return hML

def wedderburn(delta_rho, metaT, uSt, AvHyp_rho, Lo=False, Ao=False, g=9.81):
    ''' 
    Wedderburn Number (Wn) is a dimensionless parameter measuring the balance
    between wind stress and bouyancy force and is used to estimate the amount of
    upwelling occuring in a lake.  When Wn is much greater than 1, the buoyancy
    force is much greater than the wind stress and therefore there is a strong
    vertical stratification with little horizontal variation in the
    stratification. When Wn is much less than 1, the wind stress is much greater
    than the bouyancy force and upwelling is likely occuring at the upwind end
    of the lake. When Wn is near 1, the bouyance force and wind stress are
    nearly equal and horizontal mixing is considered important

    Parameters
    ----------
    delta_rho : array_like 
        density difference between the epilimnion and the hypolimnion (kg/m3).
    metaT : array_like:
        thickness of the surface layer (m)
    uSt : array_like
        water friction velocity due to wind stress (m/s)
    AvHyp_rho : array_like, scalar
        average water density of the hypolimnion layer (kg/m3)
    Lo : bool, scalar, default : False
        fetch length in the direction of the wind (m). If Lo=False, calculate it based on Ao
    Ao : bool, scalar, default : False
        Lake surface (m2). Used to calculate Lo (if not given), assume the lake as a perfect circle.
    g : scalar
        gravity acceleration (m/s2)

    Returns
    ----------
    Wedderburn Number
    
    Examples 
    ----------
    >>>    delta_rho = np.array([3.1,1.5])
    ...    metaT = np.array([5.5,2.4])
    ...    uSt = np.array([0.0028,0.0032])
    ...    Ao = np.array([80300,120000])
    ...    AvHyp_rho = np.array([999.31,999.1])
    ...    pylake.wedderburn(delta_rho, metaT, uSt, AvHyp_rho, Ao=Ao)
    array([367.22052925  21.19474286])

    Equation
    ----------   
    W = (g*delta_rho*(h**2))/(pHyp*(uSt**2)*Lo)
    where
    g = gravity acceleration
    delta_rho (kg/m3) = density difference beTempeen the epilimnion and the hypolimnion 
    metaT (m)= thickness of the surface layer
    uSt (m/s)= water friction velocity due to wind stress 
    Lo = fetch length in the direction of the wind.

    References
    ----------
    Read, J.S. et al., 2011. Derivation of lake mixing and stratification indices from high-resolution lake
    buoy data. Environ. Model. Software 26, 1325–1336. https://doi.org/10.1016/j.
    Imberger, J., Patterson, J.C., 1990. Physical limnology. Advances in Applied Mechanics 27, 353-370.
    '''
    #Must exist a better way to do this
    try:
        Lo_cond = len(Lo)
    except:
        Lo_cond = Lo
    try:
        Ao_cond = len(Ao)
    except:
        Ao_cond = Ao

    if not Lo_cond:
        if Ao_cond:
            Lo = 2 * np.sqrt(Ao/np.pi);      #Length at thermocline depth
        else:
            warnings.warn("Please indicate either Lo or Ao")

    go = g*delta_rho/AvHyp_rho
    W = go*metaT**2/(uSt**2*Lo)
    return W 

def _layers(z0, z1, dz):
    '''Midpoints and thickness of layers that exactly fill [z0, z1] with a thickness close to dz.'''
    if z1 <= z0:
        raise ValueError("the bottom of the layer must be deeper than the top")
    n = max(int(np.ceil((z1 - z0) / dz - 1e-9)), 1)
    dz_eff = (z1 - z0) / n
    return z0 + dz_eff * (np.arange(n) + 0.5), dz_eff


def _bathymetry(bthA, bthD):
    '''Bathymetry as float arrays sorted by depth.'''
    bthA, bthD = np.asarray(bthA, dtype=float), np.asarray(bthD, dtype=float)
    order = np.argsort(bthD)
    return bthA[order], bthD[order]


def schmidt_stability(Temp, depth=None, time=None, bthA=None, bthD=None, sal = 0.2, g=9.81, dz=0.1):
    '''
    Schmidt stability, or the resistance to mechanical mixing due to the potential energy inherent in the stratification of the water column.

    Parameters
    -----------
    Temp: array_like
        water temperature in degrees C
    depth:  array_like, default: None
        depth of the Temp measurements (m)
    bthA: array_like: 
        cross sectional areas (m**2) corresponding to bthD depth
    bthD: array_like
        depth (m) which correspond to areal measures in bthA
    sal: scalar,array_like, default: 0.2
        Salinity in Practical Salinity Scale units
    g: scalar, defaults: 9.81
        gravity acceleration (m/s2)
    dz: scalar, default: 0.1
        depth resolution for the integral calculus

    Gaps (NaN) inside a profile are interpolated linearly in depth; missing values at the top or
    bottom of the profile give NaN. The integral runs from the shallowest to the deepest depth of
    Temp (layer midpoints) and is divided by the area at the shallowest depth (interpolated from
    the bathymetry, which may be given on any depth axis).
    
    Returns
    ----------
        - Schmidt stability (J.m-2)

    Example
    ----------
    >>>    bthA	=	np.array([1000,900,864,820,200,10])
    ...    bthD	=	np.array([0,2.3,2.5,4.2,5.8,7]) 
    ...    wtr	=	np.array([28,27,26.4,26,25.4,24,23.3])
    ...    depth=   np.array([0,1,2,3,4,5,6])  
    ...    pylake.schmidt_stability(wtr, depth, bthA=bthA, bthD=bthD, sal=.2)
    20.77

    equation
    ----------
    g/A0 int(z0, zmax, (z-zv) rho(z) A(z) dz),   zv = centre of volume
    (equivalent to g/A0 int((z-zv)(rho(z)-rho_v) A(z) dz), because int((z-zv) A(z) dz) = 0)

    '''
    Temp, depth = to_xarray(Temp, depth, time)
    Temp = Temp.sortby("depth").interpolate_na(dim='depth')
    bthA, bthD = _bathymetry(bthA, bthD)

    z0, z1 = float(np.min(depth)), float(np.max(depth))
    layerD, dz_eff = _layers(z0, z1, dz)
    layerA = np.interp(layerD, bthD, bthA)
    A0 = float(np.interp(z0, bthD, bthA))
    if A0 <= 0:
        raise ValueError("surface area cannot be zero, check bathymetric file")

    layerP = water_density(Temp, sal).interp(depth=layerD)
    Zcv = np.sum(layerD * layerA) / np.sum(layerA)            # centre of volume
    w = xr.DataArray((layerD - Zcv) * layerA * dz_eff * g / A0, coords={"depth": layerD}, dims="depth")
    St = (layerP * w).sum("depth", skipna=False)
    if St.size == 1:
        St = St.values.item()
    return St 

def heat_content(Temp, bthA, bthD, depth=None, s=0.2, time=None, dz=0.1):
    '''
    Calculates the heat content of the water column with temperature and hypsography
    Heat content is the thermal energy in the water column, which is
    calculated by multiplying the specific heat of water (J kg-1 K-1) by the
    temperature and mass of the water in the lake, divided by the area at the
    shallowest depth (relative to 0 degC).

    Parameters
    -----------
    Temp: array_like: 
        water temperature in degrees C
    depth: array_like:
        depth (in m) of the Temp measurements
    bthA: array_like: 
        cross sectional areas (m^2) corresponding to bthD depth
    bthD array_like:
        depth (m) which correspond to areal measures in bthA
    s: scalar, array_like, default: 0.2
        salinity (PSU) for the density
    time: array_like, default: None
        times of the profiles (2D input only)
    dz: scalar, default: 0.1
        layer thickness (m) for the integration

    Computed with content() (C = cw * rho * T): layer midpoints from the shallowest to the deepest
    depth of Temp, divided by the area at the shallowest depth. Gaps (NaN) inside a profile are
    interpolated linearly in depth; missing values at the top or bottom of the profile give NaN.

    Returns
    ---------
        U: array_like
            internal energy in Joules m-2

    Example
    ---------
    >>>    bthA =np.array([1000,900,864,820,200,10])
    ...    bthD=np.array([0,2.3,2.5,4.2,5.8,7])
    ...    wtr	=np.array([28,27,26.4,26,25.4,24,23.3])
    ...    depth	= np.array([0,1,2,3,4,5,6])
    ...    pylake.heat_content(wtr, bthA, bthD, depth=depth, s=0.2)
    ...    505946464
    '''
    cw = 4186
    Temp, depth = to_xarray(Temp, depth, time)
    return content(cw * water_density(Temp, s) * Temp, bthA, bthD, dz=dz)


def content(C, bthA, bthD, depth=None, time=None, depth_range=None, per_area=True, mean=False,
            A_ref=None, dz=0.1, extrapolate=False):
    '''
    Content of a dissolved quantity (e.g. salt, oxygen, nutrients) in the lake, from a
    concentration profile and the hypsography (same approach as heat_content).

    The profile is interpolated onto layers of thickness ~dz and every layer is weighted with
    its area A(z), i.e. its volume A(z)*dz:

        total content     M = sum( C(z) * A(z) * dz )             (per_area=False)
        content per area  M / A_ref                                (per_area=True, default)
        mean conc.        M / sum( A(z) * dz )                     (mean=True)

    The function does not convert units: with C in g m-3 (= mg L-1) the result is in g m-2
    (per area), g (total) or g m-3 (mean). Salinity in g kg-1 is approximately kg m-3 in
    fresh water, which gives kg m-2 / kg / kg m-3.

    Parameters
    -----------
    C: array_like, xarray
        concentration profile(s): 1D (depth) or 2D (time x depth or depth x time), or an
        xarray DataArray with a "depth" dimension
    bthA: array_like
        cross sectional areas (m2) corresponding to bthD
    bthD: array_like
        depths (m) corresponding to bthA
    depth: array_like, default: None
        depths (m) of the C values. Necessary if C is not an xarray
    time: array_like, default: None
        times of the profiles (2D input only)
    depth_range: (float, float), default: None
        top and bottom (m) of the layer to integrate, e.g. (15, 79) for the hypolimnion.
        Default: from the shallowest to the deepest depth of C
    per_area: bool, default: True
        divide the content by the reference area A_ref (as heat_content, which gives J m-2)
    mean: bool, default: False
        return the volume-weighted mean concentration instead of the content
    A_ref: float, default: None
        reference area (m2) for per_area. Default: area at the top of the integrated layer,
        i.e. at the shallowest depth of C, as in heat_content. Use bthA at 0 m for the lake
        surface area
    dz: float, default: 0.1
        layer thickness (m) for the integration (adjusted slightly so that the layers exactly
        fill depth_range)
    extrapolate: bool, default: False
        gaps (NaN) inside a profile are always interpolated linearly in depth. Missing values at
        the top or bottom of a profile (or a depth_range beyond the data) give NaN, unless
        extrapolate=True, which uses the nearest valid value of the profile there

    Returns
    -----------
    content: float or xarray.DataArray
        content per area, total content or mean concentration (one value per profile)

    Examples
    -----------
    >>> bthA = np.array([1000, 900, 864, 820, 200, 10])
    ... bthD = np.array([0, 2.3, 2.5, 4.2, 5.8, 7])
    ... oxy = np.array([10, 10, 9.5, 9, 7, 5, 3])            # g m-3
    ... depth = np.array([0, 1, 2, 3, 4, 5, 6])
    ... pylake.content(oxy, bthA, bthD, depth=depth)                   # g m-2
    ... pylake.content(oxy, bthA, bthD, depth=depth, per_area=False)   # g
    ... pylake.content(oxy, bthA, bthD, depth=depth, mean=True)        # g m-3
    '''
    C, depth = to_xarray(C, depth, time)
    C = C.sortby("depth")
    bthA, bthD = _bathymetry(bthA, bthD)

    C = C.interpolate_na(dim="depth")                         # gaps inside the profile
    if extrapolate:
        # nearest valid value at the top / bottom (interior gaps are already filled linearly)
        C = C.interpolate_na(dim="depth", method="nearest", fill_value="extrapolate")

    z0, z1 = (float(np.min(depth)), float(np.max(depth))) if depth_range is None else map(float, depth_range)
    layerD, dz_eff = _layers(z0, z1, dz)                      # layer midpoints
    layerA = np.interp(layerD, bthD, bthA)
    if extrapolate:                                           # depth_range beyond the data
        layerC = C.interp(depth=np.clip(layerD, float(C.depth.min()), float(C.depth.max())))
        layerC = layerC.assign_coords(depth=layerD)
    else:
        layerC = C.interp(depth=layerD)                       # NaN outside the data

    weights = xr.DataArray(layerA * dz_eff, coords={"depth": layerD}, dims="depth")
    M = (layerC * weights).sum("depth", skipna=False)
    if mean:
        out = M / weights.sum()
    elif per_area:
        A0 = float(np.interp(z0, bthD, bthA)) if A_ref is None else float(A_ref)
        if A0 <= 0:
            raise ValueError("reference area must be > 0 (check the bathymetry or A_ref)")
        out = M / A0
    else:
        out = M
    if out.size == 1:
        out = out.values.item()
    return out


def seiche_period_1(depth, Zt, Lt, delta_rho, AvHyp_rho, g= 9.81) :
    '''
    Estimation of the Seiche periode based on: G. Monismith, 1985; Münnich, Wüest, & Imboden, 1992.

    Parameters
    -----------
    depth: array_like:
        depth (in m) of the Temp measurements
    Zt: array_like, scalar
        Thermocline depth 
    At: scalar 
        Surface at the thermocline depth 
    delta_rho: scalar, array_like
        Density difference between the epilimion and the hypolimnion 
    AvHyp_rho: array_like, scalar
        Average density of the hypolimnion
    g: scalar, default: 9.81
        gravity acceleration (m/s2)

    Returns
    ---------
    T1: array_like, scalar
        Mode-1 vertical seiche period (h)

    example
    ---------
    >>>    bthA =np.array([1000,900,864,820,200,10])
    ...    bthD=np.array([0,2.3,2.5,4.2,5.8,7])
    ...    depth	= np.array([0,1,2,3,4,5,6])
    ...    Zt = 4.5
    ...    Lt = 4000
    ...    delta_rho = 0.5
    ...    AvHyp_rho = 997
    ...    lw.seiche_period_1(depth, Zt, Lt, delta_rho, AvHyp_rho, g= 9.81)
    ...    1445418 
    '''
    g_reduced = g*delta_rho/AvHyp_rho
    Zd = depth[-1]
    h2 = Zd-Zt
    h1 = Zt
    T1 = 2*Lt*np.sqrt((h1+h2)/(g_reduced*h1*h2))
    return T1*60*60

def Lake_number(bthA, bthD, ustar, St, metaT, metaB, averageHypoDense, g=9.81):
    '''
    Description: The Lake Number, defined by Imberger and Patterson (1990), has been used to
    describe processes relevant to the internal mixing of lakes induced by wind
    forcings. Lower values of Lake Number represent a higher potential for
    increased diapycnal mixing, which increases the vertical flux of mass and
    energy across the metalimnion through the action of non-linear internal
    waves. Lake Number is a dimensionless index.

    Lake Number has been used, for example, to estimate the flux of oxygen
    across the thermocline in a small lake (Robertson and Imberger, 1994), and
    to explain the magnitude of the vertical flux of ammonium in a lake (Romero
    et al., 1998).
    In Imberger and Patterson (1990), Lake Number was defined as
    Ln = (g * St * (zm - zT)) / (rho_0 * u*^2 * A0^3/2 * (zm - zg))
    with all values referenced from the lake bottom, e.g.,
    zm being the height of the water level, zT the height of metalimnion
    and zg the height of center volume.
    Our calculations assume that the reference is at the lake surface, therefore:
    height of metalimnion becomes metalimnion depth (average of meta top and bot): 
    (zm - zT) --> (metaT + metaB)/2 
    height of center of volume depth becomes center of volume depth Zcv:
    (zm - zg) --> Zcv
    Further, we note that in that original work St was defined as
    St = int (z - zg) A(z) rho(z) dz
    and rLakeAnalyzer defines St as
    St = g/A0 int (z - zg) rho(z) dz
    Therefore, we calculate St_uC = St * Ao / g

    Parameters
    -----------
    bthA: array_like:
        a numeric vector of cross sectional areas (m2) corresponding to bthD depth, hypsographic areas
    bthD: array_like:
        a numeric vector of depth (m) which correspond to areal measures in bthA, hypsographic depth
    uStar: array_like:
        a numeric array of u* (m/s), water friction velocity due to wind stress
    St: array_like
        a numeric array of Schmidt stability (J/m2), as defined by Idso 1973
    metaT: array_like
        a numeric array of the top of the metalimnion depth (m from the surface)
    metaB: array_like
        a numeric array of the bottom of the metalimnion depth (m from the surface)
    averageHypoDense: array_like:
        a numeric array of the average density of the hypolimnion (kg/m3)
    g: scalar, default: 9.81
        gravity acceleration (m/s2)
    Returns
    -----------
        Ln: A numeric vector of Lake Number [dimensionless]

    Example
    -----------
    >>>    bthA =np.array([1000,900,864,820,200,10])
    ...    bthD=np.array([0,2.3,2.5,4.2,5.8,7])
    ...    uStar = np.array([0.0032,0.0024])
    ...    St = np.array([140,153])
    ...    metaT = np.array([1.34,1.54])
    ...    metaB = np.array([4.32,4.33])
    ...    averageHypoDense = np.array([999.3,999.32])
    ...    lw.Lake_number(bthA, bthD, uStar, St, metaT,metaB,averageHypoDense,self.g)
    ...    [472.30373072, 951.64555323]
    references
    -----------
    Imberger, J., Patterson, J.C., 1990. Advances in Applied Mechanics 27, 303-475.
    '''
    dz	=	0.1
    A0 = bthA[0]
    Z0 = bthD[0]

    layerD = np.arange(Z0, np.max(bthD),dz)
    layerA = np.interp(layerD, bthD, bthA)
    Zv = layerD*layerA*dz                    
    Zcv = sum(Zv)/sum(layerA)/dz
    St_uC = St*A0/g
    Ln = g*St_uC*(metaT+metaB)/(2*averageHypoDense*ustar**2*A0**(3/2)*Zcv)
    return Ln 

def buoyancy_freq(Temp, depth=None, g=9.81):
    '''
    Description: Calculate the buoyancy frequency (Brunt-Vaisala frequency) for a temperature profile.

    Parameters
    ----------- 
    Temp: array_like
        A numeric vector of water temperature in degrees C
    depth: array_like
        a numeric vector corresponding to the depth (in m) of the Temp measurements
    g: scalar, default: 9.81
        gravity acceleration (m/s2)
    
    Returns
    ----------
    n2: xarray with buoyancy frequency in units {sec^-2} and associated average lake depths (can be different from input depths).
    
    Example
    ----------
    >>>     wtr = np.array([22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42,15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43])
    ...    depth = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ...    avg_depth, buoy_freq = lw.buoyancy_freq(wtr, depth, self.g)
    ...    plt.pcolormesh(buoy_freq)
    ... array([4.10503572e-04, 9.10051748e-05, 0.00000000e+00, 0.00000000e+00,
        9.08868567e-05, 1.36034054e-04, 2.03383759e-04, 2.25040545e-04,
        1.93749334e-03, 9.17368987e-03, 2.00052155e-03, 1.31951341e-03,
        1.19625956e-03, 2.75490678e-03, 4.89153665e-04, 6.45127824e-05,
        3.73658150e-04, 4.06586584e-04, 2.70840415e-04, 1.40129203e-04,
        2.31752785e-04, 1.00430572e-04])
    '''
    Temp, depth = to_xarray(Temp,depth)
    rho = water_density(Temp, 0.2)
    numdepth = len(depth)
    rho_2 = rho.isel(depth=slice(0,numdepth-1))
    drho_dz = rho.diff('depth')/Temp.depth.diff('depth')
    rho_2["depth"] = drho_dz.depth
    n2 = g/rho_2*drho_dz
    n2["depth"] = [(a+b)/2 for a,b in zip(depth, depth[1:])]
    n2 = n2.rename({"depth":"avg_depth"})
    return n2 

def Average_layer_temp(Temp, depth_ref, layer='top', depth=None):
    '''
    Perform the layer average temperature based on the thermocline depth 

    Parameters
    -----------
    Temp: array_like, xarray: 
        A dataset containing the temperature. Must be of the same dimensions than the thermocline depth.
    depth_ref: array_like:
        reference depth in which the averaged temperature above or under is calculated.
    layer: str, default: 'top'
        layer='top' will do the average temperature above the depth_ref
        layer='bot' will do the average temperature under the depth_ref
    depth: array_like
        a numeric vector corresponding to the depth (in m) of the Temp measurements. Necessary if not using xarray

    Returns
    -----------
    mean_temp: array_like
        dataset with the mean layer temperature

    Examples
    -----------
    >>> wtr = np.array([22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42,15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43])
    ... depth = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ... depth_ref = 9 
    ... lw.Average_layer_temp(wtr, depth, depth_ref, top=True)
    ... 21.70
    ... lw.Average_layer_temp(wtr, depth, depth_ref, top=True)
    ... 10.33
    '''
    Temp, depth = to_xarray(Temp,depth)

    if layer=='top':
        mask = (Temp.depth<depth_ref)
    elif layer=='bot':
        mask = (Temp.depth>depth_ref)
    else:
        warnings.warnings("Temperature average of the whole water column")
        mean_temp = Temp.mean(dim="depth")
        return mean_temp
    masked_temp = Temp.where(mask)
    mean_temp = masked_temp.mean(dim="depth")
    return mean_temp

def Surface_Buoyancy_Flux(swT,sHFturb0,rho0):
    '''
    Calculate the surface buyoancy flux

    Parameters
    -------------
    swT: array_like, unit: °C
        Surface water temperature in °C 
    sHFturb0: array_like, unit: W/m2
        Net Surface Heat Flux - Absorbed Shortwave Radiation 
    rho0: array_like
        Lake Surface Density kg/m3

    returns
    -------------
    JB0: array_like
        Surface buoyancy flux W/m2
    '''
    JB0 = alpha(swT)*9.81/cp(swT)*sHFturb0/rho0
    return JB0

def Monin_Obukhov(ustar, JB0):
    '''
    Calculate Monin-Obhukov length

    Parameters
    -------------
    ustar: array_like, unit: m/s
        Shear Velocity on water
    JB0: array_like, unit: W/m2
        Surface buoyancy flux 
    '''
    LMO = ustar**3/0.4/JB0
    return LMO

def oxygen_solubility(T, saturation=100):
    """
    Oxygen solubility in freshwater (mg/L) according to Benson & Krause (1984),
    without salinity effect (suitable for lakes).

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in °C
    saturation : float or np.ndarray, optional
        Oxygen saturation in percent. Default is 100 for full saturation.

    Returns
    -------
    O2 : float or np.ndarray
        Oxygen concentration in mg/L corresponding to the given saturation.
        If saturation=100, this is the solubility at equilibrium with air.
    """
    T, saturation = as_array(T), as_array(saturation)
    T_K = T + 273.15
    # Benson & Krause formula (logarithm of solubility)
    ln_O2 = (-139.34411 + 1.575701e5/T_K
             - 6.642308e7/T_K**2
             + 1.243800e10/T_K**3
             - 8.621949e11/T_K**4)
    
    O2_100 = np.exp(ln_O2)  # mg/L at 100% saturation

    # Scale by percent saturation
    O2_actual = O2_100 * saturation / 100.0
    
    return O2_actual

def barometric_pressure(altitude_m):
    """
    Compute air pressure at altitude assuming isothermal atmosphere.

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level [m]

    Returns
    -------
    float
        Air pressure [Pa]
    """

    P0 = 101325.0              # Pa, standard sea-level pressure
    T0 = 288.15                # K, standard temperature (15 °C)
    g = 9.80665                # m/s², standard gravity
    M = 0.0289644              # kg/mol, molar mass of dry air
    R = 8.31447                # J/(mol·K), universal gas constant

    return P0 * np.exp(-g * M * altitude_m / (R * T0))

def water_vapor_pressure(T_C):
    """
    Water vapor pressure using Antoine equation.

    Parameters
    ----------
    T_C : float
        Temperature [°C]

    Returns
    -------
    float
        Vapor pressure [Pa]
    """
    p_mmHg = 10 ** (8.10765 - 1750.286 / (235 + T_C))
    return p_mmHg * 133.322  # mmHg → Pa

def altitude_correction(altitude_m=None, T_C=15, pressure_hPa=None):
    """
    Correction factor for the oxygen solubility at the local air pressure, for water-saturated air.

    The solubility from oxygen_solubility() refers to 1013.25 hPa. At the local air pressure P the oxygen
    partial pressure of water-saturated air is lower, and the saturation concentration is
        oxygen_solubility(T) * (P - e_w(T)) / (1013.25 hPa - e_w(T))
    with the water vapour pressure e_w. The air pressure is either a measured value (pressure_hPa) or estimated
    from the altitude (barometric_pressure(), mean pressure without weather variations).

    Parameters
    ----------
    altitude_m : float or array_like, optional
        Altitude above sea level (m). Used if pressure_hPa is not given.
    T_C : float or array_like, default: 15
        Temperature (degC) for the water vapour pressure.
    pressure_hPa : float or array_like, optional
        Measured absolute air pressure at the site (hPa), i.e. station pressure (QFE), not reduced to sea level.
        Takes precedence over altitude_m.

    Returns
    -------
    float or np.ndarray
        Dimensionless correction factor

    Example
    -------
    >>> pylake.altitude_correction(724)                       # mean pressure at 724 m
    >>> pylake.altitude_correction(pressure_hPa=955, T_C=21)  # measured pressure
    """

    P0 = 101325.0              # Pa, standard sea-level pressure
    if pressure_hPa is not None:
        p_air = np.asarray(pressure_hPa, dtype=float) * 100.0
    elif altitude_m is not None:
        p_air = barometric_pressure(altitude_m)
    else:
        raise ValueError("Either altitude_m or pressure_hPa must be given.")
    p_vapor = water_vapor_pressure(np.asarray(T_C, dtype=float))

    return (p_air - p_vapor) / (P0 - p_vapor)


def calculate_depth(press,rho,lat=45,air_press=0,integrate="down"):
    """
    Calculates the depth of measurements from pressure and density data.
    
    Parameters
    ----------
    press : float or array
        Pressure (dbar)
    rho: float or array (same dimension as press)
        Water density, independent of pressure (kg/m3)
    lat: float
        Latitude (°) to compute the gravitational acceleration
    air_press: float
        Atmospheric pressure (dbar)
    integrate : string
        Density-integration option:
            "down": integrates density values measured before the pressure value
            "up": integrates density values measured after the pressure value
            "None": uses density value measured at the same time as pressure
    
    Returns
    -------
    depth : float or array (same dimension as press)
        Depth (m) 
    """
    indrho=np.full(len(rho),1.0)
    indrho[np.isnan(rho)]=np.nan
    if integrate=="down": 
        # computes the cumulative average density for each pressure value
        rho_avg=np.nancumsum(rho)/np.nancumsum(indrho) 
    elif integrate=="up":
        # computes the reversed cumulative average density for each pressure value
        rho_avg=np.nancumsum(rho[::-1])/np.nancumsum(indrho[::-1])
        rho_avg=rho_avg[::-1]
    else:
        # uses the instantaneous density
        rho_avg=np.copy(rho)
        
    depth = 1e4 * (press-air_press) / (rho_avg*gsw.grav(lat,0)) # [m]

    return depth

def water_density(T,S=0.2):
    """
    Calculates lake water density rho as a function of water temperature and salinity (ignoring pressure effects, i.e. at sea level pressure)
    According to Chen and Millero (1986)
    
    Parameters
    ----------
    T : float or array
        Temperature in °C
    S: float or array, default: 0.2
        Salinity in Practical Salinity Scale units
    
    Returns
    -------
    density : float or array (same dimension as T and S)
        lake water density (kg m-3) 
    """    
    T, S = as_array(T), as_array(S)
    density = 1e3 * (
                    0.9998395 + 6.7914e-5 * T - 9.0894e-6 * T ** 2 + 1.0171e-7 * T ** 3 -
                    1.2846e-9 * T ** 4 + 1.1592e-11 * T ** 5 - 5.0125e-14 * T ** 6 + (
                        8.181e-4 - 3.85e-6 * T + 4.96e-8 * T ** 2) * S)

    return density

def cp(T):
    """
    Calculates the isobaric specific heat coefficient cp as a function of water temperature (ignoring salinity and pressure effects, i.e. at zero salinity and sea level pressure)
    According to Chen and Millero (1986)
    
    Parameters
    ----------
    T : float or array
        Temperature in °C
    
    Returns
    -------
    cp : float or array (same dimension as T)
        specific heat coefficient (J kg-1 K-1) 
    """    
    T = as_array(T)
    cp = (4.2174 - 3.6608e-3*T + 1.3129e-4*T**2 - 2.210e-6*T**3 + 1.508e-8*T**4)*1e3

    return cp

def alpha(T):
    """
    Calculates the thermal expansion coefficient alpha as a function of water temperature (ignoring salinity and pressure effects, i.e. at zero salinity and sea level pressure)
    According to Chen and Millero (1986)
    
    Parameters
    ----------
    T : float or array
        Temperature in °C
    
    Returns
    -------
    alpha : float or array (same dimension as T)
        thermal expansion coefficient (K-1) 
    """    
    T = as_array(T)
    alpha = (-68.00 + 18.2091*T - 0.30866*T**2 + 5.3445e-3*T**3 - 6.0721e-5*T**4 + 3.1441e-7*T**5)*1e-6

    return alpha