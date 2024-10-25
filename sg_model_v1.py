from tkinter import E
import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import scipy.signal
import pysindy as ps

## Segunda tentativa de Modelao:
# Estados:
# Fluxo de Vapor - x1
# Nível medido - x2
# Saída y = x2 + E * (x1 - u)
# Entrada de Controle u = fluxo de água de alimentação - Aproximação por equação polinomial 
# u(t) = 9.33598958e-14*t**9 - 6.23916312e-11*t**8 + 1.72209054e-08*t**7 - 2.52453204e-06*t**6 + 2.09981872e-04*t**5 - 9.77749334e-03*t**4 + 
# + 2.35107113e-01*t**3 - 2.52255080*t**2 + 8.80827964*t + 5.20576633e+02

## Equação:
# (x1)' = -772994887.801 1 + 26925.437 x1 + 115387247.397 x2 + 1723795.432 u + 0.086 x1^2 - 2773.093 x1 x2 - 39.421 x1 u + -4124375.978 x2^2 - 245986.382 x2 u - 423.817 u^2 + 
# + 42.321 x1 x2^2 + 3.316 x1 x2 u - 9806.943 x2^3 + 8505.804 x2^2 u + 35.589 x2 u^2
# (x2)' = -63.324 x1 + 47.365 u + 10.396 x1 x2 + -0.003 x1 u + -7.006 x2 u + -0.016 u^2 + -0.417 x1 x2^2 + -2.618 x2^3 + 0.373 x2^2 u
# y =  x2 + E * (x1 - u)

def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.2)
    filtered = scipy.signal.filtfilt(b, a, x)

    return filtered

def data_conditioning(df, filtered):
    dt_ = 0.02
    df.drop(df[df["LBA10CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["JEA10CL901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF001A"] <= 0.].index, inplace=True)

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*(dt_), num=n)
    df["Data_Hora"] = t

    df.rename({"Data_Hora":"t", "LBA10CF901":"x1", "JEA10CL901":"x2", "LAB60CF001A":"u", "LAB60CF901":"u_corr"}, axis='columns', inplace=True)

    if filtered:
        df["x1"] = filter(df["x1"])
        df["x2"] = filter(df["x2"])
        df["u"] = filter(df["u"])
        df["u_corr"] = filter(df["u_corr"])
    
    return df

def states(tmin, tmax, filtered):
    df = load_data('data_gv10.csv')
    x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
    y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

    K_pressure =  np.polyfit(x,y,4)
    press_corr01 = np.polyval(K_pressure,  df['LBA10CP001'])*df['LBA10CF001A']
    press_corr02 = np.polyval(K_pressure,  df['LBA10CP951A'])*df['LBA10CF001B']

    df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

    X = df.loc[tmin:tmax, ["Data_Hora", "LBA10CF901", "JEA10CL901", "LAB60CF901", "LAB60CF001A"]]

    states = data_conditioning(X, filtered)

    return  np.round(states,3)

def u_fit(df):
    def func(x, a, b, c, d, e, f, g, h, i, j):
        return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j

    popt, _ = curve_fit(func, df["t"], df["u"])
    print(popt)
    plt.plot(df["t"], df["u"], 'b-', label='data')   
    plt.plot(df["t"], func(df["t"], *popt), 'g--')
    plt.show()    

def u_fun(t):
    return 9.33598958e-14*t**9 - 6.23916312e-11*t**8 + 1.72209054e-08*t**7 - 2.52453204e-06*t**6 + 2.09981872e-04*t**5 - 9.77749334e-03*t**4 + 2.35107113e-01*t**3 - 2.52255080*t**2 + 8.80827964*t + 5.20576633e+02

def identify_model(df):
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Tempo: Treinamento e Teste
    t_train = x_train.loc[:, "t"].to_numpy()
    u_train = u_fun(t_train)
    t_test = x_test.loc[:, "t"].to_numpy()

    # Estados: Dados de Treinamento e Teste
    x_train = x_train.loc[:,["x1", "x2"]].to_numpy()
    x_test = x_test.loc[:,["x1", "x2"]].to_numpy()

    optimizer = ps.SR3(threshold=0.1, thresholder="L1")
    poly_library = ps.PolynomialLibrary(degree=3)

    model = ps.SINDy(optimizer=optimizer, feature_library=poly_library, feature_names = ['x1', 'x2', 'u'],t_default=0.02)
    model.fit(x_train, u=u_train, t=t_train, quiet=True)    
    model.print()

    x0_test=x_test[0,:]
    x_model = model.simulate(x0=x0_test, t=t_test, u=u_fun)
    x0_train=x_train[0,:]
    x_model_train = model.simulate(x0=x0_train, t=t_train, u=u_fun)

    print("Predict x1: " + str(r2_score(x_test[:,0], x_model[:, 0])) + "\n")
    print("Predict x2: " + str(r2_score(x_test[:,1], x_model[:, 1])) + "\n")

    print("Trained x1: " + str(r2_score(x_train[:,0], x_model_train[:, 0])) + "\n")
    print("Trained x2: " + str(r2_score(x_train[:,1], x_model_train[:, 1])) + "\n")

    _, ax = plt.subplots(3, 1, figsize=(10,10))
    ax[0].plot(t_train, x_model_train[:, 0], label="trained Model", linestyle='dashed',linewidth=2.0)
    ax[0].plot(t_train, x_train[:, 0], label="train signal", linewidth=.5)
    ax[0].plot(t_test, x_model[:, 0], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[0].plot(t_test, x_test[:,0], label="test signal", linewidth=.5)
    ax[0].legend()

    ax[1].plot(t_train, x_model_train[:, 1], label="trained Model", linestyle='dashed',linewidth=2.0)
    ax[1].plot(t_train, x_train[:, 1], label="train signal", linewidth=.5)
    ax[1].plot(t_test, x_model[:, 1], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[1].plot(t_test, x_test[:,1], label="test signal", linewidth=.5)
    ax[1].legend()

    ax[2].plot(df.loc[:, "t"], df.loc[:, "u"], label="u", linestyle='dashed',linewidth=2.0)
    ax[2].legend()

    plt.show()

def graphics(states):
    _, ax = plt.subplots(4, 1, figsize=(10,10))
    
    ax[0].plot(states["x1"], label='LBA10CF901 - X1')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    
    ax[1].plot(states['x2'], label='JEA10CL901 - X2')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    ax[2].plot(states['u'], label='LAB60CF001A - U Flux')
    ax[2].grid(True)
    ax[2].legend(loc='upper right')

    ax[3].plot(states['u'] - states['x1'], label='In - out')
    ax[3].legend(loc='upper right')
    ax[3].grid(True)

    plt.show()   

X = states(2925, 3500, True)
#graphics(X)
#u_fit(X)
identify_model(X)




