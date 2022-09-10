#(x1)' = 10465.031 1 + -0.002 x1 + -1711.993 x2 + 70.022 x2^2
#(x2)' = 0.000

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import sg_model as gv

A_CON_S7 = 97.64 # kg/s
A_CON_A6 = 0.006 # m/kg/s
A_CON_S6 = 12.2 # m

def filter(x):
    #b, a = scipy.signal.butter(3, 0.4)
    #filtered = scipy.signal.filtfilt(b, a, x)
    filtered = scipy.signal.savgol_filter(x, 9, 3)
    
    return filtered

def setpoint_ll():
    X = gv.states()

    t = X["t"]
    x1_den = filter(X["x1"])
    x2_den = filter(X["x2"])
    u_corr_den = filter(X["u"])

    _, ax = plt.subplots(3, 1, figsize=(10,10))    
    ax[0].plot(t, x1_den)
    ax[0].plot(t, X["x1"], linewidth=.5)

    ax[1].plot(t, x2_den)
    ax[1].plot(t, X["x2"], linewidth=.5)

    ax[2].plot(t, u_corr_den)
    ax[2].plot(t, X["u"], linewidth=.5)

    #ax[3].plot(t, setpoint)

    plt.show()

setpoint_ll()


