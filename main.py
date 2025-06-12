import data
import simulation as sim
import matplotlib.pyplot as plt
import numpy as np

exp = data.Experiment('Temp_Control_log_22-05_12-52_103456.88_device-8_isHyst_True_bandwidth_0.4_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_100.csv')
exp.limit_range(0, 500)

sim.fill_fraction = 0.28
sim.h_loss = 2
sim.Q_heater_const = 4.48
sim.calculate_geometry()
sim.T_initial = exp.Tw[0]
sim.temp_control_type = sim.control_types['Hysterisis']
sim.hysteresis_band = 0.2
sim.od_control_type = sim.control_types['None']
sim.dt = exp.t[-1]/len(exp.t)
sim.t_end = exp.t[-1]
sim.run()

plt.plot(sim.t/60, sim.Tw, label='Simulated Temperature')
exp.plot_temperature_control()

