import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import scipy.signal
import pysindy as ps

## Quarta tentativa de Modelao:
# Estados:
# Fluxo de Vapor - x1
# Nível medido - x2
# Saída y = x2 + E * (x1 - u)
# Entrada de Controle u = fluxo de água de alimentação - Vetor de entrada de medidas

## Equação:
# (x1)' = -7685165.182 1 + -140.808 x1 + 1263435.065 x2 + 47.484 u + 11.505 x1 x2 + -51925.145 x2^2 + -3.886 x2 u
# (x2)' = 0.000
# y = x2 + E * (x1 - u)

dt = 0.01

def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.08)
    filtered = scipy.signal.filtfilt(b, a, x)

    return filtered

def data_conditioning(df, filtered):    
    df.drop(df[df["LBA10CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["JEA10CL901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF001A"] <= 1.5].index, inplace=True)

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*dt, num=n)
    df["Data_Hora"] = t

    df.rename({"Data_Hora":"t", "LBA10CF901":"x1", "JEA10CL901":"x2", "LAB60CF001A":"u", "LAB60CF901":"u_corr"}, axis='columns', inplace=True)

    if filtered:
        df["x1"] = filter(df["x1"])
        df["x2"] = filter(df["x2"])
        df["u"] = filter(df["u"])
    
    return df

def states(tmin, tmax, filtered):
    df = load_data('data_gv10.csv')
    K_temp = -0.0006 # 1/ºC
    K_shift = 1.1381
    x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
    y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

    temp_corr01 = df['LAB60CF001A']*np.clip((K_temp*df['LAB60CT002'] + K_shift),0.998, 1.043) 
    temp_corr02 = df['LAB60CF001A']*np.clip((K_temp*df['LAB60CT003'] + K_shift),0.998, 1.043)

    df['LAB60CF901'] = (temp_corr01 + temp_corr02)/2.0
    df.drop(df[df["LAB60CF901"] == 0.].index, inplace=True)

    K_pressure =  np.polyfit(x,y,4)
    press_corr01 = np.polyval(K_pressure,  df['LBA10CP001'])*df['LBA10CF001A']
    press_corr02 = np.polyval(K_pressure,  df['LBA10CP951A'])*df['LBA10CF001B']

    df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

    X = data_conditioning(df, filtered)

    return  X.loc[tmin:tmax, ["t", "x1", "x2", "u", "u_corr"]]

def u_fit(df):
    def func(x, a, b, c, d, e, f, g, h, i, j):
        return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j

    popt, _ = curve_fit(func, df["t"], df["u"])
    print(popt)
    plt.plot(df["t"], df["u"], 'b-', label='data')   
    plt.plot(df["t"], func(df["t"], *popt), 'g--')
    plt.show()    

def u_fun(t):
    return 7.45409110e-07*t**9 - 1.55942700e-04*t**8 + 1.08221950e-02*t**7 - 1.82047991e-05*t**6 - 4.49095364e+01*t**5 + 3.06799073e+03*t**4 - 1.04204413e+05*t**3 + 2.01072271e+06*t**2 - 2.10917709e+07*t + 9.38253864e+07

def identify_model(df):
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Tempo: Treinamento e Teste
    t_train = x_train["t"].to_numpy()
    t_test = x_test["t"].to_numpy()

    # Estados: Dados de Treinamento e Teste
    x_train = x_train.loc[:,["x1", "x2"]].to_numpy()
    x_test = x_test.loc[:,["x1", "x2"]].to_numpy()

    u_train = u_fun(t_train)
    u_test = u_fun(t_test)  

    optimizer = ps.SR3(threshold=0.1, thresholder="L1")
    differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 23})

    model = ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, t_default=dt, feature_names = ['x1', 'x2', 'u'])
    model.fit(x_train, u=u_train, t=t_train, quiet=True)    
    model.print()

    x0_test=x_test[0,:]
    x_model = model.simulate(x0=x0_test, t=t_test, u=u_fun)
    x0_train=x_train[0,:]
    x_model_train = model.simulate(x0=x0_train, t=t_train, u=u_fun)

    print("Score : " + str(model.score(x_test, t=dt, u=u_test)))

    _, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(t_train, x_model_train[:, 0], label="trained Model", linestyle='dashed')
    ax[0].plot(t_train, x_train[:, 0], label="train signal")
    ax[0].plot(t_test, x_model[:, 0], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[0].plot(t_test, x_test[:,0], label="test signal")
    ax[0].legend()

    ax[1].plot(t_train, x_model_train[:, 1], label="trained Model", linestyle='dashed')
    ax[1].plot(t_train, x_train[:, 1], label="train signal")
    ax[1].plot(t_test, x_model[:, 1], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[1].plot(t_test, x_test[:,1], label="test signal")
    ax[1].legend()

    plt.show()

def graphics(states):
    _, ax = plt.subplots(4, 1, figsize=(10,10))
    
    ax[0].plot(states["x1"], label='X1')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    
    ax[1].plot(states['x2'], label='X2')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    ax[2].plot(states['u'], label='U Flux')
    ax[2].grid(True)
    ax[2].legend(loc='upper right')

    ax[3].plot(states['x1'] - states['u'], label='out - in')
    ax[3].legend(loc='upper right')
    ax[3].grid(True)

    plt.show()   

X = states(2500, 4300, True)
#graphics(X)
#u_fit(X)
identify_model(X)




