from tkinter import E
import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import scipy.signal
import pysindy as ps

## Terceira tentativa de Modelao:
# Estados:
# Fluxo de Vapor - x1
# Pressão de vapor - x2
# Temperatura (água de alimentação) - x3
# Saída y = y0 + E * (x1 - u) - Entrada - saída
# Entrada de Controle u = fluxo de água de alimentação - Aproximação por equação polinomial 
# u(t) = 9.33598958e-14*t**9 - 6.23916312e-11*t**8 + 1.72209054e-08*t**7 - 2.52453204e-06*t**6 + 2.09981872e-04*t**5 - 9.77749334e-03*t**4 + 
# + 2.35107113e-01*t**3 - 2.52255080*t**2 + 8.80827964*t + 5.20576633e+02

## Equação:

# y = y0 + E * (x1 - u)

dt = 0.01

def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.08)
    filtered = scipy.signal.filtfilt(b, a, x)

    return filtered

def data_conditioning(df, filtered):
    df.drop(df[df["x1"] <= 0.].index, inplace=True)
    df.drop(df[df["x2"] <= 0.].index, inplace=True)
    df.drop(df[df["x3"] <= 0.].index, inplace=True)
    df.drop(df[df["u"] <= 0.].index, inplace=True)
    df.drop(df[df["y"] <= 0.].index, inplace=True)

    n = len(df["t"])
    t = np.linspace(0, n*(dt), num=n)
    df["t"] = t

    if filtered:
        df["x1"] = filter(df["x1"])
        df["x2"] = filter(df["x2"])
        df["x3"] = filter(df["x3"])
        df["u"] = filter(df["u"])
        df["y"] = filter(df["y"])
    
    return df

def states(tmin, tmax, filtered):
    df = load_data('data_gv10.csv')

    states = pd.DataFrame(columns=['t', 'x1', 'x2', 'x3', 'u', 'y'])

    states['t'] = df.loc[:,"Data_Hora"]
    states['x1'] = (df.loc[:,"LBA10CF001A"] + df.loc[:,'LBA10CF001B'])/2.0
    states['x2'] = (df.loc[:,"LBA10CP001"] + df.loc[:,"LBA10CP951A"])/2.0
    states['x3'] = (df.loc[:,"LAB60CT002"] + df.loc[:,"LAB60CT003"])/2.0
    states['u'] = df.loc[:,"LAB60CF001A"]
    states['y'] = df.loc[:,"JEA10CL901"]

    X = data_conditioning(states.loc[tmin:tmax, ['t', 'x1', 'x2', 'x3', 'u', 'y']], filtered)

    return  X

def y(df, tmin_in_out, tmax_in_out):

    input = pd.DataFrame(columns=['u'])
    out_steam = pd.DataFrame(columns=['x1'])
    output = pd.DataFrame(columns=['y'])

    input['u'] = df.loc[tmin_in_out:tmax_in_out,"u"]
    out_steam['x1'] = df.loc[tmin_in_out:tmax_in_out,"x1"]
    output['y'] = df.loc[tmin_in_out:tmax_in_out,"y"]

    U = input['u']
    Yo = out_steam['x1']
    X = Yo - U
    Y = output['y']

    def func(x, a, b, c, d, e):
        return (a*x - b)**2 + (c*x - d)**2 - e

    popt, _ = curve_fit(func, X, Y)
    print(popt)
    plt.scatter(Yo-U, Y)   
    plt.plot(Yo-U, func(Yo-U, *popt), 'g--')
    plt.show() 

def func(x, a, b, c, d, e, f, g, h, i, j):
    return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j

def u_fit(df):
    popt, _ = curve_fit(func, df["t"], df["u"])
    #print(popt)
    #plt.plot(df["t"], df["u"], 'b-', label='data')   
    #plt.plot(df["t"], func(df["t"], *popt), 'g--')
    #plt.show()

def u_fun(t):
    return 9.33598958e-14*t**9 - 6.23916312e-11*t**8 + 1.72209054e-08*t**7 - 2.52453204e-06*t**6 + 2.09981872e-04*t**5 - 9.77749334e-03*t**4 + 2.35107113e-01*t**3 - 2.52255080*t**2 + 8.80827964*t + 5.20576633e+02

def identify_model(df):
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Tempo: Treinamento e Teste
    t_train = x_train.loc[:, "t"].to_numpy()
    t_test = x_test.loc[:, "t"].to_numpy()

    # Entrada: Dados de Treinamento Teste
    u_train = u_fun(t_train)
    u_test = u_fun(t_test)

    # Estados: Dados de Treinamento e Teste
    x_train = x_train.loc[:,["x1", "x2", "x3"]].to_numpy()
    x_test = x_test.loc[:,["x1", "x2", "x3"]].to_numpy()

    optimizer = ps.STLSQ(threshold=0.001)
    poly_library = ps.PolynomialLibrary(degree=2)

    model = ps.SINDy(optimizer=optimizer, feature_library=poly_library, feature_names = ['x1', 'x2', 'x3', 'u'],t_default=dt)
    model.fit(x_train, u=u_train, t=t_train, quiet=True)    
    model.print()

    print("Score : " + str(model.score(x_test, t=dt, u=u_test)))

    x0_test=x_test[0,:]
    x_model = model.simulate(x0=x0_test, t=t_test, u=u_fun)
    x0_train=x_train[0,:]
    x_model_train = model.simulate(x0=x0_train, t=t_train, u=u_fun)

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
    _, ax = plt.subplots(6, 1, figsize=(10,10))
    
    ax[0].plot(states["x1"], label='X1')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    
    ax[1].plot(states['x2'], label='X2')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    ax[2].plot(states['x3'], label='X3')
    ax[2].legend(loc='upper right')
    ax[2].grid(True)

    ax[3].plot(states['u'], label='U Flux')
    ax[3].grid(True)
    ax[3].legend(loc='upper right')

    ax[4].plot((states['y']), 'g', label='Level - Output')
    ax[4].grid(True)
    ax[4].legend(loc='upper right')

    ax[5].plot(states['x1'] - states['u'], label='out - in')
    ax[5].legend(loc='upper right')
    ax[5].grid(True)

    plt.show()   

X = states(2700, 3600, True)
#y(X, 0, 8000)
#graphics(X)
#u_fit(X)
identify_model(X)




