import data
import simulation as sim
import matplotlib.pyplot as plt
import numpy as np

exp = data.Experiment('Temp_Control_log_22-05_12-52_103456.88_device-8_isHyst_True_bandwidth_0.4_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_100.csv')
exp.limit_range(0, 500)

sim.fill_fraction = 0.4
sim.h_loss = 2
sim.Q_heater_const = 5
sim.calculate_geometry()
sim.T_initial = exp.Tw[0]
sim.temp_control_type = sim.control_types['Hysterisis']
sim.hysteresis_band = 0.2
sim.od_control_type = sim.control_types['None']
sim.dt = exp.t[len(exp.t)//2]/(len(exp.t)//2)
sim.t_end = exp.t[len(exp.t)//2]
sim.run()
# plt.plot(sim.t/60, sim.Tw, label='Simulated Temperature 0')
plt.plot(sim.t/60, 29.0+0.2*sim.Rs, label='Simulated Relay State')

print([(exp.t[end-1]-exp.t[start])/60 for (start, end) in data.get_ranges_from_mask(exp.Rs==1)])
exp_off_period = np.array([(exp.t[end-1]-exp.t[start])/60 for (start, end) in data.get_ranges_from_mask(exp.Rs==1)])[2:-1].mean()
exp_on_period = np.array([(exp.t[end-1]-exp.t[start])/60 for (start, end) in data.get_ranges_from_mask(exp.Rs==0)])[2:-1].mean()

import scipy.optimize as scopt
def temp_to_optimize(values):
    (sim.fill_fraction, sim.h_loss, sim.Q_heater_const) = values
    sim.calculate_geometry()
    sim.run()
    sim_off_period = np.array([(sim.t[end-1]-sim.t[start])/60 for (start, end) in data.get_ranges_from_mask(sim.Rs==1)])[2:-1].mean()
    sim_on_period = np.array([(sim.t[end-1]-sim.t[start])/60 for (start, end) in data.get_ranges_from_mask(sim.Rs==0)])[2:-1].mean()
    peak = np.max(sim.Tw[sim.t<50*60])
    cutoff = sim.Tw[sim.t>50*60]
    themin = np.min(cutoff)
    themax = np.max(cutoff)
    return 0*(peak - 31.8)**2 + 5*(themax-30.4)**2 + (themin - 29.8)**2 + 0.8*(sim_off_period - 18.38)**2 + (exp_on_period - 3.11)**2

res = scopt.minimize(temp_to_optimize, x0=(0.46, 2, 4), method='Nelder-Mead')
print(res.x)
temp_to_optimize(res.x)

sim_off_period = np.array([(sim.t[end-1]-sim.t[start])/60 for (start, end) in data.get_ranges_from_mask(sim.Rs==1)])[2:-1].mean()
sim_on_period = np.array([(sim.t[end-1]-sim.t[start])/60 for (start, end) in data.get_ranges_from_mask(sim.Rs==0)])[2:-1].mean()
peak = np.max(sim.Tw[sim.t<50*60])
cutoff = sim.Tw[sim.t>50*60]
themin = np.max(cutoff)
themax = np.min(cutoff)
plt.plot(sim.t/60, sim.Tw, label='Simulated Temperature N')

print(f"""sim_off_period = {sim_off_period:.2f}
sim_on_period = {sim_on_period:.2f}
exp_off_period = {exp_off_period:.2f}
exp_on_period = {exp_on_period:.2f}
peak = {peak:.2f}
themin = {themin:.2f}
themax = {themax:.2f}""")
exp.plot_temperature_control()
exit(1)

exp.smooth_data()
exp.calculate_derivatives(smooth_n=40)
exp.limit_range(start=100)

sim.fill_fraction = 0.5
sim.h_loss = 3
sim.Q_heater_const = 3.5
sim.calculate_geometry()
sim.T_initial = exp.Tw[0]
sim.temp_control_type = sim.control_types['Hysterisis']
sim.hysteresis_band = 0.2
sim.od_control_type = sim.control_types['None']
sim.dt = exp.t[-1]/len(exp.t)
sim.t_end = exp.t[-1]
sim.run()
plt.plot(exp.Tw[:-10], exp.Twd[10:], label='Measured')
plt.plot(sim.Tw, sim.Twd, label='Simulated')

# ranges = data.get_ranges_from_mask(exp.Rs==1)
# for i, [start, end] in enumerate(ranges):
#     if i == 0:
#         plt.plot(exp.Tw[start:end], exp.Twd[start:end], color='blue', label='Relay Off')
#     else:
#         plt.plot(exp.Tw[start:end], exp.Twd[start:end], color='blue')

# ranges = data.get_ranges_from_mask(exp.Rs==0)
# for i, [start, end] in enumerate(ranges):
#     if i == 0:
#         plt.plot(exp.Tw[start:end], exp.Twd[start:end], color='green', label='Relay On')
#     else:
#         plt.plot(exp.Tw[start:end], exp.Twd[start:end], color='green')

# plt.plot(exp.Tw[exp.Rs == 0]/60, exp.Twd[exp.Rs == 0], label='Relay Off')
data.show_plot('Temperature Derivative', 'Temp Derivative ($^\\circ C s^{-1}$)', 'Temperature ($^\\circ C$)')