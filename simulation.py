import numpy as np
import openpyxl
import random
import matplotlib.pyplot as plt

# Vessel Geometry Parameters (SI units)
R = 0.03             # Radius of cylinder (m)
L = 0.06;            # Height of cylinder (m)
fill_fraction = 0.5  #Fraction of total volume filled with water

# Water Properties
rho_water = 1000               # Water density (kg/m^3)
c_water = 4186                 # Specific heat capacity (J/(kg*K))
    
# Vessel (Glass) Properties
mass_vessel = 0.05     # Mass of the glass vessel (kg)
c_vessel = 840         # Specific heat capacity of glass (J/(kg*K))
C_glass = mass_vessel * c_vessel   # Glass heat capacity (J/K)

#Calculate vessel volumes
def calculate_geometry():

    global V_vessel, V_water, A_side_total, h_water, A_side_water, A_bottom, A_water_contact
    global A_side_env, A_top, A_env, m_water, C_water, A_water_contact

    V_vessel = np.pi * R**2 * L          # Total vessel volume (m^3)
    V_water = fill_fraction * V_vessel   # Water volume (m^3)
    
    # Area Calculations
    # Cylindrical (side) surface (where the heating mat is wrapped)
    A_side_total = 2 * np.pi * R * L
        
    # Water contacts the vessel on the bottom and the side up to the water level.
    h_water = fill_fraction * L                # height of the water column (m)
    A_side_water = 2 * np.pi * R * h_water     # portion of side in contact with water
    A_bottom = np.pi * R**2                    # bottom area (always water contact)
    A_water_contact = A_side_water + A_bottom  # Total area where glass contacts water
        
    # The environmental losses occur from the vessel surfaces not in contact with water.
    # Side area not in
    A_side_env = A_side_total - A_side_water
    A_top = np.pi * R**2 if fill_fraction < 1 else 0
    A_env = A_side_env + A_top + A_bottom  # Total area exposed to ambient

    m_water = rho_water * V_water  # Mass of water    
    C_water = m_water * c_water    # Water heat capacity (J/K)

calculate_geometry()
    
# Heat Transfer Parameters
# Heater: a fixed power is applied when ON.
Q_heater_const = 3.5  # Heater power (W) when on (Orignal = 10)
    
# Heat transfer coefficients (W/(m^2*K))
h_gw = 25    # From glass to water (Original = 50)
h_loss = 3.8 # From glass to ambient (Original = 10)

T_ambient = 22   # Ambient temperature (°C)
T_initial = 28.5

# Bacterial Growth Parameters
doubling_time = 46.5 # Doubling timein minutes
alpha_max = np.log(2)/(doubling_time*60) # Maximum Growth Rate
B = 2e6              # Maximum Population Size
A_initial = 2000     # Initial Popuation, Usually 2000
# A_initial = 25000  # Comment out if not looking at chemostat
C = 4.7e-7           # Proporionality Constant between Popluation size and OD Readings

# Noise Parameters
noise = False
if noise:
    Tg_sigma = 0.04
    Tm_sigma = 0.03
    Growth_Rate_sigma_mu = 1     # Models intrinsic noise in gene expression - always set to 1!!
    OD_measurement_sigma = 0.001
else:
    Tg_sigma, Tm_sigma, Growth_Rate_sigma_mu, OD_measurement_sigma = 0,0,1,0

# Control variables
target_temp = 30      # Target temperature
hysteresis_band = 0.5 # For hysterisis control
control_delay = 5     # For bang-bang control, seconds
control_types = { 'None': 0, 'Bang-Bang': 1, 'Hysterisis': 2 }
temp_control_type = control_types['Bang-Bang'] # 0 for None, 1 for Bang-Bang, 2 for Hysterisis
od_control_type = control_types['Hysterisis']  # 0 for None, 1 for Open-Loop, 2 for Hysterisis 

def temperature_control(Tw, Rs, k, dt):
    # None
    if temp_control_type == 0:
        Rs[k] = Rs[k-1]
    # Hysterisis
    elif temp_control_type == 1:
        if np.sum(Tw[max(int(k-1-control_delay/dt), 0)]) > target_temp:
            Rs[k] = 0.0
        else:
            Rs[k] = 1.0
    # Bang-Bang
    elif temp_control_type == 2:
        if Rs[k-1] == 1.0 and Tw[k] > target_temp + hysteresis_band:
            Rs[k] = 0.0
        elif Rs[k-1] == 0.0 and Tw[k] < target_temp - hysteresis_band:
            Rs[k] = 1.0
        else:
            Rs[k] = Rs[k-1]

def OD_control(OD_readings, k, dt):
    if od_control_type == 0:
        return 0
    # Open_Loop control
    # Set both motor powers to 65, corresponds to fluid inflow/outflow rate of 0.02ml/s
    if od_control_type == 1:
        return 0.02e-6  # Measured in m^3/s

# Simulation length variables
dt = 1
t_end = 1000
n = int(t_end/dt)

# Output values
t = np.linspace(0, t_end, n) # In seconds
Tg = np.ones(n)*T_ambient    # Glass vessel temperature (°C)
Tw = np.ones(n)*T_initial    # Water temperature (°C)
Twd = np.zeros(n)            # Water temperature derivative
Tm = np.copy(Tw)             # Measured Temerature(Added noise)
Rs = np.zeros(n)             # Record of heater power over time
A = np.ones(n)*A_initial     # Record of bacteria population over time (Proportional to OD)
OD = np.ones(n)*A_initial*C  # Optical Denisity

def run():
    n = int(t_end/dt)
    global t, Tg, Tw, Twd, Tm, Rs, A, OD
    t = np.linspace(0, t_end, n)
    Tg = np.ones(n)*T_ambient
    Tw = np.ones(n)*T_initial
    Twd = np.zeros(n)
    Tm = np.copy(Tw)
    Rs = np.zeros(n)
    A = np.ones(n)*A_initial 
    OD = np.ones(n)*A_initial*C 

    # Initialization of Arrays
    for k in range(1, n):
   
        # Heater energy
        Q_heater = Rs[max(k-1, 0)] * Q_heater_const
        # Glass energy balance
        Q_to_water = h_gw * A_water_contact * (Tg[k-1] - Tw[k-1])
        Q_loss_env = h_loss * A_env * (Tg[k-1] - T_ambient)
        dTg =  (Q_heater - Q_to_water - Q_loss_env)  * dt / C_glass
        Tg[k] = Tg[k-1] + dTg + random.gauss(0, Tg_sigma)*np.sqrt(dt)
        # Water energy balance
        Twd[k] = (Q_to_water - Q_loss_env) / C_water
        Tw[k] = Tw[k-1] + Twd[k] * dt
        # Measurement noise for water temperature readings
        Tm[k] = Tw[k] + random.gauss(0,Tm_sigma)

        # Update Bacteria Population (Due to bacteria growth)
        dt_A = A[k-1] * alpha_max * (1-A[k-1]/B)
        A[k] = A[k-1] + dt_A + random.gauss(0, Growth_Rate_sigma_mu/A[k-1])*np.sqrt(dt)
        # Update OD Measurement
        OD[k] = C*A[k] + random.gauss(0, OD_measurement_sigma)

        # Update Heater State due to temperature control
        temperature_control(Tm, Rs, k, dt)
        # Update Temperature/OD Due to parameter readings
        flowrate = OD_control(OD,k, dt)
        Tw[k] = Tw[k] + (T_ambient-Tw[k]) * flowrate*dt / V_water
        A[k] = A[k] * (1-flowrate*dt/V_water)

    # return t, Tg, Tw, Tm, Rs, A, OD

# def compare_plots_chemostat():
    
#     global temp_control_type, od_control_type
#     temp_control_type = 1
#     od_control_type = 1

#     mOD, mTm, mt, mRs = get_data('Chemostat_Data.xlsx')
#     mask = np.logical_and(mt > mt[0] + 220*60, mt < mt[0]+320*60)
#     # mask = np.ones(mt.shape, dtype=np.bool)
#     mt = mt[mask]; mOD = mOD[mask]; mTm = mTm[mask]; mRs = mRs[mask]
#     mt -= mt[0]
#     t, Tg, Tw, Tm, Rs, A, OD = run_sim(mt[-1]/len(mt), mt[-1])
#     fig, ax1 = plt.subplots()
#     ax1.set_xlabel('Time (min)')
#     ax1.set_ylabel("Temperatures (°C)")
#     ax1.plot(mt/60, mTm, label = 'Measured Temperature', color='blue')
#     ax1.plot(t/60, Tm, label = 'Simulated Temperaure', color='black')
    
#     ax2 = ax1.twinx()
    
#     ax2.plot(mt/60, mOD, label = 'Measured OD', color='orange')
#     ax2.plot(t/60, OD, label = 'Simulated OD', color='red')
#     ax2.set_ylabel('Optical Denisty Readings')
#     plt.legend()
#     plt.title("OD Readings & Temperature over time")
#     plt.grid()
#     plt.show()

# if __name__ == "__main__":
#     compare_plots_chemostat()

 
