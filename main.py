import data
import simulation as sim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, CheckButtons

slider_values = {
    'fill_fraction': ('Fill fraction', 0.05, 1.1),
    'h_gw': ('Glass-Water heat transfer coef.', 1, 50),
    'h_loss': ('Glass-Environment heat transfer coef.', 1, 50),
    'h_top_bottom': ('Water-Environment heat transfer coef.', 1, 50),
    'doubling_time': ('Doubling Time', 15, 50),
    'noise': ('Noise level', 0, 2),
    'target_temp': ('Target temperature', 25, 45),
    'turbidostat_rate': ('Turbidostat flow rate', 0.1e-6, 0.3e-5),
    'chemostat_rate': ('Chemostat flow rate', 0.002e-6, 0.030e-6),
    'A_initial': ('Initial bacterial population (log)', 1, 20),   
    'Q_heater_const': ('Heater power', 1, 20),
    'hysteresis_band': ('Hysteresis Band', 0.01, 1.5)
}

def setup_sliders(mvars, ncurves=1):
    fig, axs = plt.subplots(len(mvars)+ncurves, 1)
    bottomp = 0.04
    topp = 0.08
    midp = 0.05
    alloced_space = 0.5
    plot_width = 0.65
    if ncurves == 1:
        alloced_space = 0.5
        axs[0].set_position([0.2, 1-(topp+alloced_space)+midp, plot_width, alloced_space-midp])
    else:
        alloced_space = 0.70
        axs[0].set_position([0.2, 1-(topp+alloced_space) + midp, plot_width, alloced_space/2 - midp])
        axs[1].set_position([0.2, 1-(topp+alloced_space) + alloced_space/2 + midp , plot_width, (alloced_space-midp)/2])
    
    slider_spacing = (1-(topp+alloced_space+bottomp)) / len(mvars)
    slider_height = min(slider_spacing/2, 0.03)
    slider_width = 0.65
    sliders = []
    for i in range(len(mvars)):
        axs[i+ncurves].set_position([0.2, bottomp + i * slider_spacing, slider_width, slider_height])
        label, valmin, valmax = slider_values[mvars[i]]
        sliders.append(Slider(
            ax=axs[i+ncurves],
            label = label,
            valmin = valmin,
            valmax= valmax,
            valinit = getattr(sim, mvars[i]) if mvars[i] != 'A_initial' else np.log(sim.A_initial)
        ))
    return fig, axs, sliders

def temperature_control():
    exp = data.Experiment('Temp_Control_log_22-05_12-52_103456.88_device-8_isHyst_True_bandwidth_0.4_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_100.csv')
    exp.limit_range(0, 500)

    slidervars = ['fill_fraction', 'h_loss', 'h_gw', 'h_top_bottom', 'Q_heater_const', 'hysteresis_band', 'noise', 'target_temp']
    fig, axs, sliders = setup_sliders(slidervars, ncurves=1)
    exp.plot_temperature(axs[0], label='Measured Temperature')
    exp.smooth_data()
    exp.plot_temperature(axs[0], label='Smoothed Temperature')
    [linet] = exp.plot_constant_in_time(axs[0], 29.8, label='Hysterisis Bounds', color='black')
    [lineb] = exp.plot_constant_in_time(axs[0], 30.2, color='black')

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
        for i in range(len(sliders)):
            setattr(sim, slidervars[i], sliders[i].val)
        sim.calculate_noise()
        sim.calculate_geometry()
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

    slidevars = ['chemostat_rate', 'doubling_time', 'A_initial', 'noise'] 
    fig, axs, sliders = setup_sliders(slidevars, ncurves=2)
    
    axs[1].plot(exp.t/60, exp.Tw, label='Measured Temperature')
    axs[0].plot(exp.t/60, exp.OD, label='Measured OD')
    exp.remove_Rs_effect()
    exp.smooth_data(smooth_n=60, thin_n=2)
    axs[1].plot(exp.t/60, exp.Tw, label='Smooth Temperature')
    axs[0].plot(exp.t/60, exp.OD, label='Smoothed OD')

    sim.A_initial = 0.1e6
    sim.T_initial = exp.Tw[0]
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.2
    sim.od_control_type = sim.control_types['Open-Loop']
    sim.dt = exp.t[-1]/len(exp.t)
    sim.t_end = exp.t[-1]
    sim.run()

    [simodl] = axs[0].plot(sim.t/60, sim.OD, label='Simulated OD')
    [simtempl] = axs[1].plot(sim.t/60, sim.Tw, label='Simulated Temperature')

    axs[1].set_title('Chemostat Experiment')
    axs[0].set_ylabel('Optical Density')
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel('Time (min)')
    axs[1].set_ylabel('Temperatures ($^\\circ C$)')
    axs[1].grid()
    axs[1].legend()

    def update_func(val):
        for i in range(len(sliders)):
            setattr(sim, slidevars[i], sliders[i].val)
            if slidevars[i] == 'A_initial':
                sim.A_initial = np.exp(sim.A_initial)
        sim.calculate_noise()
        sim.calculate_geometry()
        sim.run()
        simodl.set_ydata(sim.OD)
        simtempl.set_ydata(sim.Tw)
        fig.canvas.draw_idle()
    for slider in sliders:
        slider.on_changed(update_func)

    plt.show()
    
def turbidostat():
    
    exp = data.Experiment('OD_Control_log_02-06_10-14_21119.3_device-8_isHyst_True_bandwidth_0.3_isBang_False_bDelay_5.0_motorspeed_200_stirspeed_80.csv')
    exp.t -= exp.t[0]

    sim.A_initial = 0.95e5
    sim.temp_control_type = sim.control_types['Hysterisis']
    sim.hysteresis_band = 0.15
    sim.od_control_type = sim.control_types['Closed-Loop']
    sim.dt = min(exp.t[-1]/len(exp.t), 0.5)
    sim.t_end = exp.t[-1]
    sim.run()

    slidevars = ['fill_fraction', 'h_loss', 'h_gw', 'h_top_bottom', 'Q_heater_const', 'hysteresis_band', 'noise', 'turbidostat_rate', 'doubling_time', 'A_initial']
    fig, axs, sliders = setup_sliders(slidevars, ncurves = 2)

    axs[0].plot(exp.t/60, exp.OD, label='Measured OD')
    axs[1].plot(exp.t/60, exp.Tw, label='Measured Temperature')
    exp.remove_Rs_effect(20, 184)
    exp.OD = data.minimum_filter(exp.OD, 5)
    exp.smooth_data(smooth_n=10, thin_n=2)
    exp.OD = data.minimum_filter(exp.OD, 5)
    exp.smooth_data(smooth_n=5, thin_n=2)
    axs[0].plot(exp.t/60, exp.OD, label='Smooth OD')
    axs[1].plot(exp.t/60, exp.Tw, label='Smooth Temperature')
        
    [simodl] = axs[0].plot(sim.t/60, sim.OD, label='Simulated OD')
    [simtempl] = axs[1].plot(sim.t/60, sim.Tw, label='Simulated Temperature')
    
    axs[1].set_title('Turbidostat Experiment')
    axs[0].set_ylabel('Optical Density')
    axs[0].set_xlabel('Time (min)')
    axs[1].set_ylabel('Temperatures ($^\\circ C$)')
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()

    def update_func(val):
        for i in range(len(sliders)):
            setattr(sim, slidevars[i], sliders[i].val)
            if slidevars[i] == 'A_initial':
                sim.A_initial = np.exp(sim.A_initial)
        sim.calculate_noise()
        sim.calculate_geometry()
        sim.run()
        simodl.set_ydata(sim.OD)
        simtempl.set_ydata(sim.Tw)
        fig.canvas.draw_idle()
    for slider in sliders:
        slider.on_changed(update_func)

    plt.show()

temperature_control()
chemostat()
turbidostat()