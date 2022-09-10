import pandas as pd
import datetime
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy import interpolate
import scipy.signal
import pysindy as ps

def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.4)
    filtered = scipy.signal.filtfilt(b, a, x)

    return filtered

def interpolate_data(data, t, tnew):
    y = data
    x = t
    f = interpolate.interp1d(x, y)
    ynew = f(tnew)

    #plt.plot(x, y, 'o', tnew, ynew, '-')
    #plt.show()

    return ynew

def data_conditioning(df):
    dt_ = 0.1
    df.drop(df[df["LBA10CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["JEA10CL901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF001A"] <= 0.].index, inplace=True)

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*(0.1), num=n)

    #t = np.arange(0, len(df["Data_Hora"])*10, 10)  
    #tnew = np.arange(t[0], t[-1], dt_)

    #x1 = interpolate_data(df["LBA10CF901"], t, tnew)
    #x2 = interpolate_data(df["JEA10CL901"], t, tnew)
    #u =  interpolate_data(df["LAB60CF001A"], t, tnew)
    #u_corr = interpolate_data(df["LAB60CF901"], t, tnew)

    x1 = df["LBA10CF901"]
    x2 = df["JEA10CL901"]
    u =  df["LAB60CF001A"]
    u_corr = df["LAB60CF901"]

    states = pd.DataFrame(columns = ["t", "x1","x2","u", "u_corr"])

    states["t"] = t
    states["x1"] = filter(x1)
    states["x2"] = filter(x2)
    states["u"] = filter(u)
    states["u_corr"] = filter(u_corr)
    
    return states

def states(tmin, tmax):
    df = load_data('data_gv10.csv')
    x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
    y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

    K_pressure =  np.polyfit(x,y,4)
    press_corr01 = np.polyval(K_pressure,  df['LBA10CP001'])*df['LBA10CF001A']
    press_corr02 = np.polyval(K_pressure,  df['LBA10CP951A'])*df['LBA10CF001B']

    df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

    X = df.loc[tmin:tmax, ["Data_Hora", "LBA10CF901", "JEA10CL901", "LAB60CF901", "LAB60CF001A"]]

    df_ = data_conditioning(X)

    return  np.round(df_,3)

def u_fit(df):
    def func(x, a, b, c, d):
        return a - b*np.exp(-c*(x + d))
        #return a*x**4 + b*x**3 + c*x**2 + d*x + e

    popt, _ = curve_fit(func, df["t"], df["u"])
    print(popt)
    plt.plot(df["t"], df["u"], 'b-', label='data')   
    plt.plot(df["t"], func(df["t"], *popt), 'g--')
    plt.show()    

def u_fun(t):
    return 0.37435684 / (0.0007211 + np.exp(-0.02606472 * t))

def identify_model(df):
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Entrada: Dados de Teste e Treinamento
    u_train = x_train.loc[:,["u"]].to_numpy()
    u_test = x_test.loc[:,["u"]].to_numpy()

    # Tempo: Treinamento e Teste
    t_train = x_train.loc[:, "t"].to_numpy().reshape((len(x_train),))
    t_test = x_test.loc[:, "t"].to_numpy().reshape((len(x_test),))

    # Estados: Dados de Treinamento e Teste
    x_train = x_train.loc[:,["x1", "x2"]].to_numpy()
    x_test = x_test.loc[:,["x1", "x2"]].to_numpy()

    optimizer = ps.SR3(threshold=0.1, thresholder="L1")
    poly_library = ps.PolynomialLibrary(degree=2)

    model = ps.SINDy(optimizer=optimizer, feature_library=poly_library, t_default=0.01)
    model.fit(x_train, u=u_train, t=t_train, quiet=True, unbias=True)    
    model.print()

    x0_test=x_test[0,:]
    x_model = model.simulate(x0=x0_test, t=t_test, u=u_test)
    x0_train=x_train[0,:]
    x_model_train = model.simulate(x0=x0_train, t=t_train, u=u_train)

    _, ax = plt.subplots(3, 1, figsize=(10,10))
    ax[0].plot(t_train[:-1], x_model_train[:, 0], label="trained Model", linestyle='dashed',linewidth=2.0)
    ax[0].plot(t_train, x_train[:, 0], label="train signal", linewidth=.5)
    ax[0].plot(t_test[:-1], x_model[:, 0], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[0].plot(t_test, x_test[:,0], label="test signal", linewidth=.5)
    ax[0].legend()

    ax[1].plot(t_train[:-1], x_model_train[:, 1], label="trained Model", linestyle='dashed',linewidth=2.0)
    ax[1].plot(t_train, x_train[:, 1], label="train signal", linewidth=.5)
    ax[1].plot(t_test[:-1], x_model[:, 1], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[1].plot(t_test, x_test[:,1], label="test signal", linewidth=.5)
    ax[1].legend()

    ax[2].plot(df.loc[:, "t"], df.loc[:, "u"], label="u", linestyle='dashed',linewidth=2.0)
    ax[2].legend()

    plt.show()

def graphics(states):
    _, ax = plt.subplots(4, 1, figsize=(10,10))
    
    ax[0].plot(states['x1'], label='LBA10CF901 - X1')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    
    ax[1].plot(states['x2'], label='JEA10CL901 - X2')
    #ax[1].plot(np.arange(2800, 3901) ,12.2*np.ones(len(states['x2'])), label='Setpoint')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    ax[2].plot(states['u'], label='LAB60CF001A - U Flux')
    ax[2].grid(True)
    ax[2].legend(loc='upper right')

    ax[3].plot(states['u_corr'], label='LAB60CF901 - U Corrected')
    ax[3].legend(loc='upper right')
    ax[3].grid(True)

    plt.show()   

X = states(2925, 3000)
#graphics(X)
#u_fit(X)
identify_model(X)




