import pylake
import numpy as np

Temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
depth = np.array([1,2,3,4,5,6,7,8])

thermoD, thermoInd = pylake.thermocline(Temp, depth)
epilimnion, hypolimnion = pylake.metalimnion(Temp, depth)
SthermoD, SthermoInd = pylake.seasonal_thermocline(Temp,depth)
thermoR = pylake.robust_thermocline(Temp, depth)
assert abs(thermoR - (1.9*2.5 + 2.1*3.5 + 3.5*6.5 + 1.0*7.5) / 8.5) < 1e-9   # centroid of the strong intervals
assert np.isnan(pylake.robust_thermocline(np.full(8, 10.0), depth))           # mixed: below mixed_cutoff
assert abs(pylake.robust_thermocline(np.array([0.5, 2.5, 3.5, 3.8, 3.9, 3.9, 4.0, 4.0]), depth) - (2.0*1.5 + 1.0*2.5) / 3.0) < 1e-9   # inverse (under ice): only increases >= 25 % of the max count
hML = pylake.mixed_layer(Temp,depth, threshold=0.4, method="density")
hML_T = pylake.mixed_layer(Temp, depth, threshold=0.5, method="temperature")
assert abs(hML_T - (2 + 0.2 / 1.9)) < 1e-9          # first exceedance between 2 and 3 m, interpolated
assert pylake.mixed_layer(np.full(8, 10.0), depth, method="temperature") == 8   # never exceeded -> deepest depth
assert pylake.mixed_layer(np.array([13.52, 13.44, 13.42, 13.42, 13.40, 13.39, 13.41, 13.41]), depth, threshold=0.1, method="temperature") == 4.0   # differences of exactly 0.10 (13.52 - 13.42) do not exceed the threshold
assert abs(pylake.mixed_layer(Temp, depth, method="temperature") - (1 + 0.02 / 0.3)) < 1e-9   # default 0.02 degC
try:
    pylake.mixed_layer(Temp, depth, threshold=0.4)
    raise AssertionError("mixed_layer without method must fail")
except TypeError:
    pass
n2 = pylake.buoyancy_freq(Temp, depth, g=9.81)
avg_ep_T = pylake.Average_layer_temp(Temp, depth=depth, depth_ref=epilimnion, layer='top')
avg_ep_rho = pylake.water_density(avg_ep_T)
avg_hyp_T = pylake.Average_layer_temp(Temp, depth=depth, depth_ref=hypolimnion, layer='bot')
avg_hyp_rho = pylake.water_density(avg_hyp_T)
delta_rho = avg_hyp_rho-avg_ep_rho

bthA = np.array([100,90,86,82,20,1])
bthD = np.array([0,2.3,2.5,4.2,5.8,7])
Lo = 50
ustar = 0.5
W = pylake.wedderburn(delta_rho, metaT=epilimnion, uSt=ustar, AvHyp_rho=avg_hyp_rho, Lo=Lo, g=9.81)
Schmidt_stability = pylake.schmidt_stability(Temp, depth=depth, bthA=bthA, bthD=bthD, sal = 0.2, g=9.81, dz=0.1)
heat_content = pylake.heat_content(Temp, bthA=bthA, bthD=bthD, depth=depth, s=0.2)
seiche_period_1 = pylake.seiche_period_1(depth=depth, Zt=thermoD, Lt=Lo, delta_rho=delta_rho, AvHyp_rho=avg_hyp_rho, g= 9.81)
Lake_number = pylake.Lake_number(bthA=bthA, bthD=bthD, ustar=ustar, St=Schmidt_stability, metaT=epilimnion, metaB=hypolimnion, averageHypoDense=avg_hyp_rho, g=9.81)

# content: hypsometric content of a dissolved quantity (here oxygen in g m-3)
oxy = np.array([10, 10, 9.5, 9, 7, 5, 3, 2])
oxy_area = pylake.content(oxy, bthA=bthA, bthD=bthD, depth=depth)                    # g m-2
oxy_total = pylake.content(oxy, bthA=bthA, bthD=bthD, depth=depth, per_area=False)   # g
oxy_mean = pylake.content(oxy, bthA=bthA, bthD=bthD, depth=depth, mean=True)         # g m-3
oxy_hypo = pylake.content(oxy, bthA=bthA, bthD=bthD, depth=depth, depth_range=(4, 7))
# constant concentration in a box-shaped lake: content per area = C * layer thickness
box = pylake.content(np.full(8, 2.0), bthA=np.array([100, 100]), bthD=np.array([0, 10]), depth=depth)
assert abs(box - 2.0 * 7) < 1e-9
assert abs(pylake.content(np.full(8, 2.0), bthA=bthA, bthD=[0, 2.3, 2.5, 4.2, 5.8, 7], depth=depth, mean=True) - 2.0) < 1e-9
# a gap inside the profile is interpolated, a missing bottom value gives NaN unless extrapolate=True
oxy_gap = oxy.astype(float); oxy_gap[-1] = np.nan
assert np.isnan(pylake.content(oxy_gap, bthA=bthA, bthD=bthD, depth=depth))
assert np.isfinite(pylake.content(oxy_gap, bthA=bthA, bthD=bthD, depth=depth, extrapolate=True))

# altitude_correction: from altitude (mean pressure) or from a measured pressure
assert abs(pylake.altitude_correction(0) - 1) < 1e-9
assert abs(pylake.altitude_correction(pressure_hPa=1013.25, T_C=20) - 1) < 1e-12
p724 = pylake.barometric_pressure(724) / 100
assert abs(pylake.altitude_correction(724) - pylake.altitude_correction(pressure_hPa=p724)) < 1e-12
assert np.allclose(pylake.altitude_correction(pressure_hPa=[950, 960], T_C=[20, 21]),
                   [(950 - pylake.water_vapor_pressure(20) / 100) / (1013.25 - pylake.water_vapor_pressure(20) / 100),
                    (960 - pylake.water_vapor_pressure(21) / 100) / (1013.25 - pylake.water_vapor_pressure(21) / 100)])

# longwave_in: default (Fink) unchanged, Dilley & O'Brien + Crawford & Duchon as in Simstrat
Ta, RH = np.array([0., 10., 20.]), np.array([90., 70., 60.])
assert np.allclose(pylake.longwave_in(Ta, RH, 0.5), pylake.longwave_in(Ta, RH, 0.5, method='fink'))
lw_dc = pylake.longwave_in(Ta, RH, 0.5, method='dilley_crawford')
ea = RH * pylake.saturation_vapour_pressure(Ta) / 100; TK = Ta + 273.15
L_clear = 59.38 + 113.7 * (TK / 273.16) ** 6 + 96.96 * np.sqrt(465 * ea / TK * 0.04)   # as in Simstrat
assert np.allclose(lw_dc, 0.97 * (0.5 * L_clear + 0.5 * 5.67e-8 * TK ** 4))
assert np.allclose(pylake.longwave_in(Ta, RH, 1.0, method='dilley_crawford'), 0.97 * 5.67e-8 * TK ** 4)   # overcast: black body
