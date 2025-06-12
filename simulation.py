import numpy as np
import openpyxl
import random
import matplotlib.pyplot as plt

# Vessel Geometry Parameters (SI units)
R = 0.03             # Radius of cylinder (m)
L = 0.06;            # Height of cylinder (m)
fill_fraction = 0.281  #Fraction of total volume filled with water

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
    global A_side_env, A_top, A_env, m_water, C_water, A_water_contact, A_water_env

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
    A_side_env = A_side_total - A_side_water   # Total area of jacket exposed to ambient
    A_top = np.pi * R**2 if fill_fraction < 1 else 0
    A_env = A_side_env + A_bottom
    A_water_env = A_top # Total area of water exposed to ambient

    m_water = rho_water * V_water  # Mass of water    
    C_water = m_water * c_water    # Water heat capacity (J/K)

calculate_geometry()
    
# Heat Transfer Parameters
# Heater: a fixed power is applied when ON.
Q_heater_const = 3.78  # Heater power (W) when on (Orignal = 10)
    
# Heat transfer coefficients (W/(m^2*K))
h_gw = 25    # From glass to water (Original = 50)
h_loss = 4.162 # From glass to ambient (Original = 10)
h_top_bottom = 2

mdot_sensor = 1e-5 # Mass transfer rate throigh the sensor pump, in SI units
m_sensor = 1e-3      # Mass of the water in the sensor chamber, in SI units

T_ambient = 22   # Ambient temperature (°C)
T_initial = 28.5

# Bacterial Growth Parameters
doubling_time = 46.5 # Doubling timein minutes
B = 2e8              # Maximum Population Size
A_initial = 2000     # Initial Popuation, Usually 2000
C = 4.7e-7           # Proporionality Constant between Popluation size and OD Readings

chemostat_rate = 0.012e-6
turbidostat_rate = 0.2e-5

# Noise Parameters
noise = True

def calculate_noise():
    global noise, Tg_sigma, Tm_sigma, Growth_Rate_sigma_mu, OD_measurement_sigma, alpha_max
    alpha_max = np.log(2)/(doubling_time*60) # Maximum Growth Rate
    if noise == True:
        Tg_sigma = 0.04
        Tm_sigma = 0.03
        Growth_Rate_sigma_mu = 1     # Models intrinsic noise in gene expression - always set to 1!!
        OD_measurement_sigma = 0.001
    elif noise == False:
        Tg_sigma, Tm_sigma, Growth_Rate_sigma_mu, OD_measurement_sigma = 0,0,1,0
    else:
        Tg_sigma             = noise * 0.04
        Tm_sigma             = noise * 0.03
        Growth_Rate_sigma_mu = 1 + noise
        OD_measurement_sigma = noise * 0.001

calculate_noise()

# Control variables
target_temp = 30      # Target temperature
hysteresis_band = 0.5 # For hysterisis control
control_delay = 5     # For bang-bang control, seconds
control_types = { 'None': 0, 'Bang-Bang': 1, 'Hysterisis': 2, 'Open-Loop': 1, 'Closed-Loop': 2 }
temp_control_type = control_types['Bang-Bang'] # 0 for None, 1 for Bang-Bang, 2 for Hysterisis
od_control_type = control_types['Closed-Loop'] # 0 for None, 1 for Open-Loop, 2 for Closed-Loop 

def temperature_control(k):
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

pump_state = False

def OD_control(k):
    if od_control_type == 0:
        return 0
    # Open Loop control
    # Set both motor powers to 65, corresponds to fluid inflow/outflow rate of 0.02ml/s
    elif od_control_type == 1:
        return chemostat_rate  # Measured in m^3/s
    # Closed Loop control
    elif od_control_type == 2:
        global pump_state
        if pump_state and OD[k] <= 0.4:
            pump_state = False
        elif not pump_state and OD[k] >= 0.7:
            pump_state = True
        
        return turbidostat_rate if pump_state else 0

# Simulation length variables
dt = 1
t_end = 1000
n = int(t_end/dt)

# Output values
t = np.linspace(0, t_end, n) # In seconds
Tg = np.ones(n)*T_ambient    # Glass vessel temperature (°C)
Tw = np.ones(n)*T_initial    # Water temperature (°C)
Twd = np.zeros(n)            # Water temperature derivative
Ts = np.ones(n)*T_initial
Tm = np.copy(Tw)             # Measured Temerature(Added noise)
Rs = np.zeros(n)             # Record of heater power over time
A = np.ones(n)*A_initial     # Record of bacteria population over time (Proportional to OD)
OD = np.ones(n)*A_initial*C  # Optical Denisity

def run():
    n = int(t_end/dt)
    global t, Tg, Tw, Twd, Ts, Tm, Rs, A, OD
    t = np.linspace(0, t_end, n)
    Tg = np.ones(n)*T_ambient
    Tw = np.ones(n)*T_initial
    Twd = np.zeros(n)
    Tm = np.copy(Tw)
    Rs = np.zeros(n)
    A = np.ones(n)*A_initial 
    OD = np.ones(n)*A_initial*C 
    Ts = np.ones(n)*T_initial

    # Initialization of Arrays
    for k in range(1, n):
   
        # Tranfered heat terms
        Q_heater = Rs[max(k-1, 0)] * Q_heater_const
        Q_g2w = h_gw * A_water_contact * (Tg[k-1] - Tw[k-1])
        Q_loss_env_g = h_loss * A_env * (Tg[k-1] - T_ambient)
        Q_loss_env_w = h_top_bottom * A_water_env * (Tw[k-1] - T_ambient)
        Q_s2w = c_water * mdot_sensor * (Ts[k-1] - Tw[k-1])

        # Jacket energy equation
        dTgdt =  (Q_heater - Q_g2w - Q_loss_env_g) / C_glass
        Tg[k] = Tg[k-1] + dTgdt * dt + random.gauss(0, Tg_sigma)*np.sqrt(dt)

        # Water energy equation
        Twd[k] = (Q_g2w - Q_loss_env_w + Q_s2w) / C_water
        Tw[k] = Tw[k-1] + Twd[k] * dt
        
        # Inline sensor equation
        dTsdt = -Q_s2w / (m_sensor * c_water)
        Ts[k] = Ts[k-1] + dTsdt*dt

        # Measurement noise for water temperature readings
        Tm[k] = Ts[k] + random.gauss(0, Tm_sigma)
        
        # Update Bacteria Population (Due to bacteria growth)
        # shape * scale = 1
        # shape * scale * scale = 
        alpha = alpha_max * np.random.gamma(1/Growth_Rate_sigma_mu, Growth_Rate_sigma_mu)
        dA = A[k-1] * alpha * (1-A[k-1]/B) * dt
        A[k] = A[k-1] + dA
        # Update OD Measurement
        OD[k] = C*A[k] + random.gauss(0, OD_measurement_sigma) - C*A_initial

        # Update Heater State due to temperature control
        temperature_control(k)
        
        # Update Temperature/OD Due to parameter readings
        flowrate = OD_control(k)
        Tw[k] += (T_ambient-Tw[k]) * flowrate*dt / V_water
        A[k] *= 1-flowrate*dt/V_water




 
