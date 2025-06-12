import data
import simulation as sim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def setup_sliders(labels, values, slider_height = 0.03, slider_spacing = 0.07, bottom_start = 0.05):
    fig, axs = plt.subplots(len(labels)+1, 1)
    axs[0].set_position([0.2, bottom_start + (len(labels)+1) * slider_spacing, 0.65, 1 - bottom_start - (len(labels)+2) * slider_spacing])
    sliders = []
    for i in range(len(labels)):
        axs[i+1].set_position([0.2, bottom_start + i * slider_spacing, 0.65, slider_height])
        sliders.append(Slider(
            ax=axs[i+1],
            label=labels[i],
            valmin = values[i][0],
            valmax= values[i][1],
            valinit = values[i][2]
        ))
    return fig, axs, sliders


def temperature_control_simple():
    exp = data.Experiment('Temp_Control_log_22-05_12-52_103456.88_device-8_isHyst_True_bandwidth_0.4_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_100.csv')
    exp.limit_range(0, 500)

    sim.T_initial = exp.Tw[0]
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.2
    sim.od_control_type = sim.control_types['None']
    sim.dt = exp.t[-1]/len(exp.t)
    sim.t_end = exp.t[-1]
    sim.run()

    exp.plot_temperature(plt, label='Measured Temperaure')
    exp.smooth_data()
    exp.plot_temperature(plt, label='Smoothed Temperature')
    exp.plot_constant_in_time(plt, 29.8, label='Hysterisis Bounds', color='black')
    exp.plot_constant_in_time(plt, 30.2, color='black')
    exp.plot_Rs(plt, 29.3, 0.4, label='Relay state')

    plt.plot(sim.t/60, sim.Tw, label='Simulated Temperature')
    data.show_plot(title='Temperature Control Experiment', ylabel='Temperature ($^\\circ$C)')

def temperature_control():
    exp = data.Experiment('Temp_Control_log_22-05_12-52_103456.88_device-8_isHyst_True_bandwidth_0.4_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_100.csv')
    exp.limit_range(0, 500)

    fig, axs, sliders = setup_sliders(
        ['Fill fraction', 'Loss heat transfer coefficient', 'Heater power', 'Hysterisis Band'], [
        (0.1, 0.9, sim.fill_fraction), 
        (0.1, 20, sim.h_loss),
        (0.1, 15, sim.Q_heater_const),
        (0.05, 0.5, 0.2)
    ])
    exp.smooth_data()
    exp.plot_temperature(axs[0], label='Smoothed Temperature')
    [linet] = exp.plot_constant_in_time(axs[0], 29.8, label='Hysterisis Bounds', color='black')
    [lineb] = exp.plot_constant_in_time(axs[0], 30.2, color='black')
    # exp.plot_Rs(plt, 29.3, 0.4, label='Relay state')

    sim.T_initial = exp.Tw[0]
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.2
    sim.od_control_type = sim.control_types['None']
    sim.dt = min(exp.t[-1]/len(exp.t), 0.5)
    sim.t_end = exp.t[-1]
    sim.run()

    [simlinew] = axs[0].plot(sim.t/60, sim.Tw, label='Simulated Water Temperatures')
    [simlines] = axs[0].plot(sim.t/60, sim.Ts, label='Simulated Sensor Temperatures')

    axs[0].set_title('Temperature Control Experiment')
    axs[0].set_ylabel('Temperatures ($^\\circ C$)')
    axs[0].set_xlabel('Time (min)')
    axs[0].grid()
    axs[0].legend()

    def update_func(val):
        sim.fill_fraction = sliders[0].val
        sim.calculate_geometry()
        sim.h_loss = sliders[1].val
        sim.Q_heater_const = sliders[2].val
        sim.hysteresis_band = sliders[3].val
        sim.run()
        simlinew.set_ydata(sim.Tw)
        simlines.set_ydata(sim.Ts)
        linet.set_ydata([sim.target_temp - sim.hysteresis_band, sim.target_temp - sim.hysteresis_band])
        lineb.set_ydata([sim.target_temp + sim.hysteresis_band, sim.target_temp + sim.hysteresis_band])
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update_func)
    
    plt.show()


def chemostat():
    exp = data.Experiment('Chemostat_log_29-05_12-57_87438.19_device-8_isHyst_True_bandwidth_0.4_isBang_False_bDelay_5.0_motorspeed_1_stirspeed_0.csv')
    exp.limit_range(240, 360)
    exp.t -= exp.t[0]
    exp.OD = 0.25-5.1*exp.OD
    plt.plot(exp.t/60, exp.OD, label='Measured OD')
    exp.remove_Rs_effect()
    exp.smooth_data(smooth_n=60, thin_n=2)
    plt.plot(exp.t/60, exp.OD, label='Smoothed OD')

    sim.A_initial = 0.1e6
    sim.T_initial = exp.Tw[0]
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.2
    sim.od_control_type = sim.control_types['Open-Loop']
    sim.dt = exp.t[-1]/len(exp.t)
    sim.t_end = exp.t[-1]
    sim.run()

    plt.plot(sim.t/60, sim.OD, label='Simulated OD')

    plt.title('Chemostat Experiment')
    plt.ylabel('Optical Density Measurements')
    plt.xlabel('Time (min)')
    plt.grid()
    plt.legend()
    plt.show()
    

    # plt.plot(sim.t/60, sim.Tw, label='Simulated Temperature')
    # exp.plot_temperature_control()

def turbidostat_part1():
    exp = data.Experiment('OD_Control_log_02-06_10-14_21119.3_device-8_isHyst_True_bandwidth_0.3_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_80.csv')
    exp.t -= exp.t[0]
    
    fig, axs = plt.subplots(5, 1)
    
    axs[0].plot(exp.t/60, exp.OD, label='Measured OD')
    exp.remove_Rs_effect(20, 184)
    exp.OD = data.minimum_filter(exp.OD, 5)
    exp.smooth_data(smooth_n=10, thin_n=1)
    exp.OD = data.minimum_filter(exp.OD, 5)
    exp.smooth_data(smooth_n=5, thin_n=1)
    axs[0].plot(exp.t/60, exp.OD, label='Smooth OD')

    sim.A_initial = 0.95e5
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.15
    sim.od_control_type = sim.control_types['Closed-Loop']
    sim.dt = min(exp.t[-1]/len(exp.t), 0.5)
    sim.t_end = exp.t[-1]
    sim.run()

    [simline] = axs[0].plot(sim.t/60, sim.OD, label='Simulated OD')
    axs[0].set_title('Turbidostat Experiment')
    axs[0].set_ylabel('OD Readings')
    axs[0].set_xlabel('Time (min)')
    axs[0].legend()
    axs[0].grid()
    

    slider_height = 0.03; slider_spacing = 0.07; bottom_start = 0.05
    axs[0].set_position([0.2, bottom_start + 5 * slider_spacing, 0.65, 1 - bottom_start - 6 * slider_spacing])
    axs[1].set_position([0.2, bottom_start + 3 * slider_spacing, 0.65, slider_height])
    axs[2].set_position([0.2, bottom_start + 2 * slider_spacing, 0.65, slider_height])
    axs[3].set_position([0.2, bottom_start + 1 * slider_spacing, 0.65, slider_height])
    axs[4].set_position([0.2, bottom_start + 0 * slider_spacing, 0.65, slider_height])

    
    slider_fill_fraction = Slider(
        ax=axs[1],
        label='Fill fraction',
        valmin = 0.05,
        valmax= 0.9,
        valinit = sim.fill_fraction
    )
    slider_h_loss = Slider(
        ax=axs[2],
        label='Heat transfer coefficient',
        valmin = 0.1,
        valmax= 10,
        valinit = sim.h_loss
    )
    slider_Q = Slider(
        ax=axs[3],
        label='Heater power',
        valmin = 0.1,
        valmax= 10,
        valinit = sim.Q_heater_const
    )
    slider_Ai = Slider(
        ax=axs[4],
        label='log A(0) (initial population)',
        valmin = 1,
        valmax= 20,
        valinit = np.log(sim.A_initial)
    )

    def update_func(val):
        sim.A_initial = np.exp(slider_Ai.val)
        sim.fill_fraction = slider_fill_fraction.val
        sim.calculate_geometry()
        sim.h_loss = slider_h_loss.val
        sim.Q_heater_const = slider_Q.val
        sim.run()
        simline.set_ydata(sim.OD)
        fig.canvas.draw_idle()

    slider_fill_fraction.on_changed(update_func)
    slider_h_loss.on_changed(update_func)
    slider_Q.on_changed(update_func)
    slider_Ai.on_changed(update_func)

    plt.show()
  
def turbidostat_part2():
    exp = data.Experiment('OD_Control_log_02-06_10-14_21119.3_device-8_isHyst_True_bandwidth_0.3_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_80.csv')
    exp.t -= exp.t[0]

    fig, axs = plt.subplots(5, 1)
    slider_height = 0.03 
    slider_spacing = 0.07
    bottom_start = 0.05
    axs[0].set_position([0.2, bottom_start + 5 * slider_spacing, 0.65, 1 - bottom_start - 6 * slider_spacing])
    axs[1].set_position([0.2, bottom_start + 3 * slider_spacing, 0.65, slider_height])
    axs[2].set_position([0.2, bottom_start + 2 * slider_spacing, 0.65, slider_height])
    axs[3].set_position([0.2, bottom_start + 1 * slider_spacing, 0.65, slider_height])
    axs[4].set_position([0.2, bottom_start + 0 * slider_spacing, 0.65, slider_height])

    exp.plot_temperature(axs[0], label='Measured Temperaure')
    exp.smooth_data()
    exp.plot_temperature(axs[0], label='Smoothed Temperature')
    exp.plot_constant_in_time(axs[0], 29.8, label='Hysterisis Bounds', color='black')
    exp.plot_constant_in_time(axs[0], 30.2, color='black')

    sim.fill_fraction = 0.5145
    sim.h_loss = 14.56
    sim.Q_heater_const = 12.23
    sim.A_initial = np.exp(11.48)
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.15
    sim.od_control_type = sim.control_types['Closed-Loop']
    sim.dt = min(exp.t[-1]/len(exp.t), 0.5)
    sim.t_end = exp.t[-1]
    sim.run()

    slider_fill_fraction = Slider(
        ax=axs[1],
        label='Fill fraction',
        valmin = 0.05,
        valmax= 0.9,
        valinit = sim.fill_fraction
    )
    slider_h_loss = Slider(
        ax=axs[2],
        label='Heat transfer coefficient',
        valmin = 0.1,
        valmax= 20,
        valinit = sim.h_loss
    )
    slider_Q = Slider(
        ax=axs[3],
        label='Heater power',
        valmin = 0.1,
        valmax= 15,
        valinit = sim.Q_heater_const
    )
    slider_Ai = Slider(
        ax=axs[4],
        label='log A(0) (initial population)',
        valmin = 5,
        valmax= 20,
        valinit = np.log(sim.A_initial)
    )

    # [simlinew] = axs[0].plot(sim.t/60, sim.Tw, label='Simulated Water Temperatures')
    [simlines] = axs[0].plot(sim.t/60, sim.Ts, label='Simulated Sensor Temperatures')
    axs[0].set_title('Turbidostat Experiment')
    axs[0].set_ylabel('Temperatures ($^\\circ C$)')
    axs[0].set_xlabel('Time (min)')
    axs[0].legend()
    axs[0].grid()

    def update_func(val):
        sim.A_initial = np.exp(slider_Ai.val)
        sim.fill_fraction = slider_fill_fraction.val
        sim.calculate_geometry()
        sim.h_loss = slider_h_loss.val
        sim.Q_heater_const = slider_Q.val
        sim.run()
        # simlinew.set_ydata(sim.Tw)
        simlines.set_ydata(sim.Ts)
        fig.canvas.draw_idle()

    slider_fill_fraction.on_changed(update_func)
    slider_h_loss.on_changed(update_func)
    slider_Q.on_changed(update_func)
    slider_Ai.on_changed(update_func)

    plt.show()


temperature_control()
# chemostat()
turbidostat_part1()
turbidostat_part2()
