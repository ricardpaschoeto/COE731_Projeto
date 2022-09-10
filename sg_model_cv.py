from copy import copy
from turtle import window_height
import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import curve_fit
import scipy.signal
import pysindy as ps

def load_data(path):
    df = pd.read_csv(path)

    return df

def filter(x):
    b, a = scipy.signal.butter(3, 0.4)
    filtered = scipy.signal.filtfilt(b, a, x)
    #filtered = scipy.signal.savgol_filter(x, 9, 3)

    return filtered

def data_conditioning(df):

    df.drop(df[df["LBA10CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["JEA10CL901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF001A"] <= 0.].index, inplace=True)

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*(0.1), num=n)
    df["Data_Hora"] = t

    df.columns = ["t", "x1","x2","u_corr", "u"]

    df["t"] = filter(df["t"])
    df["x1"] = filter(df["x1"])
    df["x2"] = filter(df["x2"])
    df["u"] = filter(df["u"])
    df["u_corr"] = filter(df["u_corr"])
    
    return df

def states():
    df = load_data('data_gv10.csv')
    x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
    y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

    K_pressure =  np.polyfit(x,y,4)
    press_corr01 = np.polyval(K_pressure,  df['LBA10CP001'])*df['LBA10CF001A']
    press_corr02 = np.polyval(K_pressure,  df['LBA10CP951A'])*df['LBA10CF001B']

    df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

    X = df.loc[2925:3000, ["Data_Hora", "LBA10CF901", "JEA10CL901", "LAB60CF901", "LAB60CF001A"]]

    df_ = data_conditioning(X)

    return  np.round(df_,3)

# TODO: Encontrar uma função (regressão) para a entrada U (Fluxo)
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

    ## CROSS-VLIDATION
    model = ps.SINDy(t_default=0.01, feature_names = ['x1', 'x2'])
    param_grid = {
        "optimizer":[ps.SR3()],
        "optimizer__threshold":[0.1],
        "optimizer__thresholder":["L1"],
        "feature_library":[ps.PolynomialLibrary()],
        "feature_library__degree":[2]
    }

    search = GridSearchCV(
        model,
        param_grid,
        cv=TimeSeriesSplit(n_splits=5)
    )

    fit_params = {}
    fit_params["u"]: u_train
    fit_params["unbias"]: False

    search.fit(x_train, **fit_params)
    print("Best parameters:", search.best_params_)
    search.best_estimator_.print() 

    x0_test=x_test[0,:]
    x_model = search.best_estimator_.simulate(x0=x0_test, t=t_test, u=u_test)
    x0_train=x_train[0,:]
    x_model_train = search.best_estimator_.simulate(x0=x0_train, t=t_train, u=u_train)

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

X = states()
#graphics(X)
#u_fit(X)
identify_model(X)




