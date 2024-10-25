import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from pysindy.feature_library import CustomLibrary
import scipy.signal
import pysindy as ps

## Quarta tentativa de Modelo (USANDO CUSTOM LIB):
# Estados:
# Fluxo de Vapor - x1
# Nível medido - x2
# Saída y = x2 + E * (x1 - u)
# Entrada de Controle u = fluxo de água de alimentação - Vetor de entrada de medidas

## Equação:
#(x1)' = 4207755272.506 1 + 539.554 x1 + -1079175153.059 x2 + -546.931 u + -0.151 x1^2 + -43.356 x1 x2 + 0.293 x1 u + 90916427.935 x2^2 + 43.997 x2 u + -0.142 u^2 + 
# -2517477.568 f0(x2) + -3.115 f1(x1) + -13198648.474 f1(x2) + 3.594 f1(u) + 0.266 f2(x1) + -7702397.829 f2(x2) + -2.904 f2(u)
#(x2)' = 4.792 x1 + -50426.098 x2 + -4.701 u + -0.391 x1 x2 + 8319.331 x2^2 + 0.384 x2 u + -338.561 f0(x2) + 1667.459 f1(x2) + 0.066 f2(x1) + -8223.899 f2(x2)
# y = x2 + E * (x1 - u)

dt = 0.01

def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.1)
    filtered = scipy.signal.filtfilt(b, a, x)

    return filtered

def data_conditioning(df, filtered):    
    df.drop(df[df["LBA10CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["JEA10CL901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF001A"] <= 0.].index, inplace=True)

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*dt, num=n)
    df["Data_Hora"] = t

    df.rename({"Data_Hora":"t", "LBA10CF901":"x1", "JEA10CL901":"x2", "LAB60CF001A":"u", "LAB60CF901":"u_corr"}, axis='columns', inplace=True)

    flt = pd.DataFrame(columns=['x1','x2','u'])
    if filtered:        
        flt["t"] = t
        flt["x1"] = filter(df["x1"])
        flt["x2"] = filter(df["x2"])
        flt["u"] = filter(df["u"])
        flt["u_corr"] = filter(df["u_corr"])

        return df, flt
    
    return df, df

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

    X, flt = data_conditioning(df, filtered)

    return  X.loc[tmin:tmax, ["t", "x1", "x2", "u", "u_corr"]], flt.loc[tmin:tmax, ["t", "x1", "x2", "u", "u_corr"]]

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

def optimizer(name):
    if name == 'SSR':
        optimizer_ = ps.SSR(
            alpha=0.05,
            fit_intercept=True,
        )
    if name == 'SR3':
        optimizer_ = ps.SR3(
            threshold=0.01,
            thresholder="L1",
            trimming_fraction=0.1,
            max_iter=4000,
            tol=1e-14,
        )
    if name == 'STLSQ':
        optimizer_ = ps.STLSQ(
        threshold=0.01,
        fit_intercept=False,
    )
    
    return optimizer_

def identify_model(df):

    opts = ['SSR', 'SR3', 'STLSQ']
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Entrada: Dados de Teste e Treinamento
    u_train = x_train.loc[:,["u"]].to_numpy()
    u_test = x_test.loc[:,["u"]].to_numpy()

    # Tempo: Treinamento e Teste
    t_train = x_train["t"].to_numpy()
    t_test = x_test["t"].to_numpy()

    # Estados: Dados de Treinamento e Teste
    x_train = x_train.loc[:,["x1", "x2"]].to_numpy()
    x_test = x_test.loc[:,["x1", "x2"]].to_numpy()

    # Initialize custom SINDy library so that we can have x_dot inside it.
    library_functions = [
         lambda x : x**3,
         lambda x : np.sin(x),
         lambda x : np.cos(x),
    ]

    poly_lib = ps.PolynomialLibrary(degree=2)
    fourier_lib = ps.FourierLibrary(n_frequencies=1)
    identity_lib = ps.IdentityLibrary()
    custom_lib = CustomLibrary(library_functions=library_functions)

    combined_lib = poly_lib + custom_lib

    _, axs = plt.subplots(2*len(opts), 1, sharex=True, figsize=(10, 10))
    j = 0
    for opt in opts:
        print('Otimizador esparso : ' + opt)
        optimizer_ = optimizer(opt)

        model = ps.SINDy(
            feature_names = ['x1', 'x2', 'u'],
            optimizer=optimizer_,
            feature_library=combined_lib,
            differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={'window_length':5})
        )

        model.fit(x_train, u=u_train, t=t_train)
        model.print()

        # Compute derivatives with a finite difference method, for comparison
        x_dot_train_computed  = model.differentiate(x_train, dt)
        x_dot_test_computed  = model.differentiate(x_test, dt)

        # Predict derivatives using the learned model
        x_dot_train_predicted  = model.predict(x_train, u=u_train)
        x_dot_test_predicted  = model.predict(x_test, u=u_test)

        for i in range(2):
            axs[j].plot(t_train, x_dot_train_computed[:, i], 'g', label='trained numerical derivative - ' + opt)
            axs[j].plot(t_train, x_dot_train_predicted[:, i],'b--', label='trained model prediction - ' + opt)
            axs[j].plot(t_test, x_dot_test_computed[:, i], 'k', label='tested numerical derivative - ' + opt)            
            axs[j].plot(t_test, x_dot_test_predicted[:, i],'r--', label='tested model prediction - ' + opt)
            axs[j].legend(loc='center')
            axs[j].set(xlabel='t', ylabel='$\dot x_{}$'.format(i+1))
            j = j + 1
    plt.show()

def graphics(states, flt):
    _, ax = plt.subplots(4, 1, figsize=(10,10))
    
    ax[0].plot(states["x1"], label='X1')
    ax[0].plot(flt["x1"], label='X1 - Filtered')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    
    ax[1].plot(states['x2'], label='X2')
    ax[1].plot(flt['x2'], label='X2 - Filtered')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    ax[2].plot(states['u'], label='U Flux')
    ax[2].plot(flt['u'], label='U Flux - Filtered')
    ax[2].grid(True)
    ax[2].legend(loc='upper right')

    ax[3].plot(states['x1'] - states['u'], label='out - in')
    ax[3].plot(flt['x1'] - flt['u'], label='out - in - Filtered')
    ax[3].legend(loc='upper right')
    ax[3].grid(True)

    plt.show()   

X, flt = states(2920, 4000, True)
#graphics(X, flt)
#u_fit(X)
identify_model(flt)




