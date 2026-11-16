import pylake
import numpy as np

Temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
depth = np.array([1,2,3,4,5,6,7,8])

thermoD, thermoInd = pylake.thermocline(Temp, depth)
epilimnion, hypolimnion = pylake.metalimnion(Temp, depth)
SthermoD, SthermoInd = pylake.seasonal_thermocline(Temp,depth)
hML = pylake.mixed_layer(Temp,depth, threshold=0.4)
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
