import numpy as np
import scipy.signal as scsignal
import scipy.interpolate as scinterp
import scipy.ndimage as scimage
import csv
import matplotlib.pyplot as plt

def moving_averages(data, n):
    if n == 0:
        return data
    cs = np.cumsum(data)
    cs[n:] = cs[n:] - cs[:-n]
    cs[n:] /= n
    cs[:n] /= np.array(range(1, n+1))
    return cs

def median_filter(data, n):
    if n == 0:
        return data
    return scimage.median_filter(data, n)
    
def minimum_filter(data, n):
    if n == 0:
        return data
    return scimage.minimum_filter(data, n)

def thin(data, n):
    if n < 2:
        return data
    return data[::n]

def extend_mask(mask, n, nn, inverted=False):
    """Extends a boolean mask numpy array such that ranges of True increase in size"""
    if inverted:
        mask = np.logical_not(mask)
    shift_mask_1 = np.pad(mask, n)[:len(mask)]
    shift_mask_2 = np.pad(mask, nn)[(2*nn):]
    new_mask = np.logical_or(shift_mask_1, mask)
    new_mask = np.logical_or(shift_mask_2, new_mask)
    if inverted:
        new_mask = np.logical_not(new_mask)
    return new_mask

def interpolate_mask(mask, x, y):
    """Keeps values where the mask was True and interpolates the values of y based on what remains from x"""
    newx = x[mask]
    newx = np.insert(newx, 0, x[0])
    newx = np.insert(newx, len(newx), x[-1])
    newy = y[mask]
    newy = np.insert(newy, 0, y[0])
    newy = np.insert(newy, len(newy), y[-1])
    y_interp = scinterp.interp1d(newx, newy)
    return y_interp(x)

def least_squares(x, y):
    """Does linear least squares"""
    xm = x.mean()
    ym = y.mean()
    n = len(x)
    Sxx = np.sum((x - xm)*(x - xm))
    Sxy = np.sum((x - xm)*(y - ym))
    m = Sxy / Sxx
    c = ym - m*xm
    residual = np.sum((y - m*x-c)*(y - m*x-c)) / (n-1)
    stdm = np.sqrt(residual / Sxx)
    stdc = np.sqrt(residual * (1/n + xm**2 / Sxx))
    return m, c, stdm, stdc

def derivative(x, y):
    """Calculates the derivative, with clamping when it's infinity"""
    ydif = y[1:] - y[:-1]
    xdif = x[1:] - x[:-1]
    der = ydif / xdif
    mask = extend_mask(xdif != 0, 2, 2, inverted=True)
    der = interpolate_mask(mask, x[:-1], der)
    der = np.insert(der, 0, der[0])
    return der

def get_ranges_from_mask(mask):
    """Transforms a boolean mask into a list of ranges, represented by tuples"""
    start = 0
    end = 0
    ranges = []
    while True:
        while start < len(mask) and mask[start]:
            start+=1
        if start >= len(mask):
            break
        end = start
        while end < len(mask) and not mask[end]:
            end+=1
        ranges.append((start, end))
        if end >= len(mask):
            break
        start = end
    return ranges

def full_least_squares(func, guess):
    """Does least squares on an arbitrary function"""
    vals = scopt.least_squares(func, x0=guess)
    return vals.x

def get_data(filename):
    """Gets the data from one of the log files"""
    file = open(filename)
    reader = csv.reader(file)
    times = []
    ods = []
    Temps = []
    Rs = []
    Ms = []
    for row in reader:
        ods.append(float(row[3]))
        Temps.append(float(row[4]))
        Rs.append(float(row[5]))
        Ms.append(float(row[6]))
        times.append(float(row[7]))
    file.close()
    return np.array(times), np.array(ods), np.array(Temps), np.array(Rs), np.array(Ms)

def show_plot(title='', ylabel='', xlabel='Time (min)'):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    plt.legend()
    plt.show()

class Experiment:

    def __init__(self, filename):
        self.filename = filename
        self.t, self.OD, self.Tw, self.Rs, self.Ms = get_data(filename)

    def remove_Rs_effect(self, good_start_t, good_end_t):
        range_mask = np.logical_and(self.t > good_start_t*60, self.t < good_end_t*60)
        new_Rs = self.Rs[range_mask]
        Rs_mask = extend_mask(new_Rs == 0, 10, 1, inverted=True)
        new_ods = interpolate_mask(Rs_mask, self.t[range_mask], self.OD[range_mask])
        od_offsets = self.OD[range_mask] - new_ods
        od_offset = np.mean(od_offsets[new_Rs != 0])
        self.OD = median_filter(self.OD - od_offset*(Rs != 0), 20)
    
    def calculate_derivatives(self, smooth_n = 150):
        very_smooth_ods = moving_averages(self.OD, smooth_n)
        very_smooth_Temps = moving_averages(self.Tw, smooth_n)
        self.Twd = derivative(self.t, very_smooth_Temps)
        self.ODd = derivative(self.t, very_smooth_ods)

    def smooth_data(self, smooth_n = 50, thin_n = 10):
        self.t = thin(self.t, thin_n)
        self.Rs = thin(self.Rs, thin_n)
        self.Ms = thin(self.Ms, thin_n)
        self.OD = thin(moving_averages(self.OD, smooth_n), thin_n)
        self.Tw = thin(moving_averages(self.Tw, smooth_n), thin_n)
        if hasattr(self, 'Twd'):
            self.Twd = thin(moving_averages(self.Twd, smooth_n), thin_n)
            self.ODd = thin(moving_averages(self.ODd, smooth_n), thin_n)

    def limit_range(self, start=-10, end=1e4):
        mask = np.logical_and(self.t >= start*60, self.t <= end*60)
        self.t = self.t[mask]
        self.OD = self.OD[mask]
        self.Tw = self.Tw[mask]
        self.Rs = self.Rs[mask]
        self.Ms = self.Ms[mask]
        if hasattr(self, 'Twd'):    
            self.Odd = self.ODd[mask]
            self.Twd = self.Twd[mask]

    def plot_temperature(self, ax, **kwargs):
        ax.plot(self.t/60, self.Tw, **kwargs)

    def plot_Rs(self, ax, offset=0, size=1, **kwargs):
        ax.plot(self.t/60, offset + size*self.Rs, **kwargs)

    def plot_constant_in_time(self, ax, value, **kwargs):
        ax.plot([self.t[0]/60, self.t[-1]/60], [value, value], **kwargs)

    def plot_temperature_control(self):
        self.plot_temperature(plt, label='Measured Temperaure')
        self.smooth_data()
        self.plot_temperature(plt, label='Smoothed Temperature')
        self.plot_constant_in_time(plt, 29.8, label='Hysterisis Bounds', color='black')
        self.plot_constant_in_time(plt, 30.2, color='black')
        self.plot_Rs(plt, 29.3, 0.4, label='Relay state')
        show_plot(title='Temperature Control Experiment', ylabel='Temperature ($^\\circ$C)')
        


