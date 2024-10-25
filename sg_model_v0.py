from tkinter import E
import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy.signal
import pysindy as ps

## Primeira tentativa de Modelao:
# Estados:
# Fluxo de Vapor - x1
# Nível medido - x2
# Saída y = x2
# Entrada de Controle u = fluxo de água de alimentação - Vetor de entrada de medidas

## Equação:
# (x1)' = -115093.218 1 + -42.860 x1 + 18671.568 x2 + 54.333 u + 3.391 x1 x2 + -757.046 x2^2 + -4.341 x2 u
# (x2)' = -351.479 1 + -0.008 x1 + 58.660 x2 + 0.006 u + -2.443 x2^2
# y = x2


def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.4)
    filtered = scipy.signal.filtfilt(b, a, x)

    return filtered

def data_conditioning(df, filtered):
    dt_ = 0.1
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

def identify_model(df):
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Entrada: Dados de Teste e Treinamento
    u_train = x_train.loc[:,["u"]].to_numpy()
    u_test = x_test.loc[:,["u"]].to_numpy()

    # Tempo: Treinamento e Teste
    t_train = x_train.loc[:, "t"].to_numpy()
    t_test = x_test.loc[:, "t"].to_numpy()

    # Estados: Dados de Treinamento e Teste
    x_train = x_train.loc[:,["x1", "x2"]].to_numpy()
    x_test = x_test.loc[:,["x1", "x2"]].to_numpy()

    optimizer = ps.SR3(threshold=0.1, thresholder="L1")
    poly_library = ps.PolynomialLibrary(degree=2)

    model = ps.SINDy(optimizer=optimizer, feature_library=poly_library, feature_names = ['x1', 'x2', 'u'],t_default=0.01)
    model.fit(x_train, u=u_train, t=t_train, quiet=True)    
    model.print()

    # # Compute derivatives with a finite difference method, for comparison
    # x_dot_train_computed  = model.differentiate(x_train, 0.01)
    # x_dot_test_computed  = model.differentiate(x_test, 0.01)

    # # Predict derivatives using the learned model
    # x_dot_train_predicted  = model.predict(x_train, u=u_train)
    # x_dot_test_predicted  = model.predict(x_test, u=u_test)

    # _, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    # for i in range(x_test.shape[1]):
    #     axs[i].plot(t_train, x_dot_train_computed[:, i], 'k', label='trained numerical derivative')
    #     axs[i].plot(t_test, x_dot_test_computed[:, i], 'k', label='tested numerical derivative')
    #     axs[i].plot(t_train, x_dot_train_predicted[:, i],'r--', label='trained model prediction')
    #     axs[i].plot(t_test, x_dot_test_predicted[:, i],'r--', label='tested model prediction')
    #     axs[i].legend()
    #     axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))
    # plt.show()

    x0_test=x_test[0,:]
    x_model = model.simulate(x0=x0_test, t=t_test, u=u_test, integrator='solve_ivp')
    x0_train=x_train[0,:]
    x_model_train = model.simulate(x0=x0_train, t=t_train, u=u_train, integrator='solve_ivp')

    print("Predict x1: " + str(r2_score(x_test[:-1,0], x_model[:, 0])) + "\n")
    print("Predict x2: " + str(r2_score(x_test[:-1,1], x_model[:, 1])) + "\n")

    print("Trained x1: " + str(r2_score(x_train[:-1,0], x_model_train[:, 0])) + "\n")
    print("Trained x2: " + str(r2_score(x_train[:-1,1], x_model_train[:, 1])) + "\n")

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

X = states(2500, 4300, True)
#graphics(X)
identify_model(X)




