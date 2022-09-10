import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import curve_fit
import pysindy as ps

def load_data(path):
    df = pd.read_csv(path)

    return df

def data_conditioning(df):    
    df.drop(df[df["LBA10CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["JEA10CL901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF901"] <= 0.].index, inplace=True)
    df.drop(df[df["LAB60CF001A"] <= 0.].index, inplace=True)

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*6, num=n)
    df["Data_Hora"] = t
    
    return np.round(df, 5)

def states():
    df = load_data('data.csv')
    K_temp = -0.0006 # 1/ºC
    K_shift = 1.1381
    x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
    y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

    temp_corr01 = df['LAB60CF001A']*np.clip((K_temp*df['LAB60CT002'] + K_shift),0.998, 1.043) 
    temp_corr02 = df['LAB60CF001B']*np.clip((K_temp*df['LAB60CT003'] + K_shift),0.998, 1.043)

    df['LAB60CF901'] = (temp_corr01 + temp_corr02)/2.0
    df.drop(df[df["LAB60CF901"] == 0.].index, inplace=True)

    K_pressure =  np.polyfit(x,y,4)
    press_corr01 = np.polyval(K_pressure,  df['LBA10CP001'])*df['LBA10CF001A']
    press_corr02 = np.polyval(K_pressure,  df['LBA10CP951A'])*df['LBA10CF001B']

    df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

    X = data_conditioning(df)

    return  X.loc[:, ["Data_Hora", "LBA10CF901", "JEA10CL901", "LAB60CF901", "LAB60CF001A"]]

def u_fit(df):
    def func(x, a, b, c):
        return a / (b + np.exp(-c*x))

    popt, _ = curve_fit(func, df["Data_Hora"], df["LAB60CF001A"])
    print(popt)
    plt.plot(df["Data_Hora"], df["LAB60CF001A"], 'b-', label='data')   
    plt.plot(df["Data_Hora"], func(df["Data_Hora"], *popt), 'g--')
    plt.show()    

def u_fun(t):
    return 0.37435684 / (0.0007211 + np.exp(-0.02606472 * t))

def identify_model(df):
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=False)

    t_train = x_train["Data_Hora"].to_numpy().reshape((len(x_train),))
    t_test = x_test["Data_Hora"].to_numpy().reshape((len(x_test),))
    x_train = x_train.loc[:,["LBA10CF901", "JEA10CL901", "LAB60CF901"]].to_numpy()
    x_test = x_test.loc[:,["LBA10CF901", "JEA10CL901", "LAB60CF901"]].to_numpy()

    u_train = u_fun(t_train)    
    u_test = u_fun(t_test)    

    #optimizer = ps.SR3(threshold=0.25, thresholder="l1")
    optimizer = ps.SR3()
    #differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 15})
    differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 17})
    #poly_library = ps.PolynomialLibrary(degree=2)

    ## TODO:CROSS-VLIDATION
    model = ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, t_default=6)
    param_grid = {
        "optimizer__threshold": [0.01, 0.02, 0.1, 0.25, 0.35, 0.4],
        "optimizer__thresholder":["l1"],
        "feature_library": [ps.PolynomialLibrary()],
    }

    search = GridSearchCV(
        model,
        param_grid,
        cv=TimeSeriesSplit(n_splits=5)
    )

    search.fit(x_train)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print() 
    #model = ps.SINDy(optimizer=optimizer, feature_library=poly_library, differentiation_method=differentiation_method)
    #model.fit(x_train, u=u_train, t=t_train, quiet=True)    
    #model.print()
    #print('Model score: %f' % model.score(x_test, u=u_test, t=6))

    x0=x_test[0,:]
    x_model = search.best_estimator_.simulate(x0=x0, t=t_test, u=u_fun)
    x0=x_train[0,:]
    x_model_train = search.best_estimator_.simulate(x0=x0, t=t_train, u=u_fun)

    _, ax = plt.subplots(3, 1, figsize=(10,10))
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

    ax[2].plot(t_train, x_model_train[:, 2], label="trained Model", linestyle='dashed')
    ax[2].plot(t_train, x_train[:, 2], label="train signal")
    ax[2].plot(t_test, x_model[:, 2], label="tested Model", color="black", linestyle='dashed',linewidth=2.0)
    ax[2].plot(t_test, x_test[:,2], label="test signal")
    ax[2].legend()

    plt.show()

def graphics(states):
    plt.figure()
    plt.subplot(411)    
    plt.plot(states['LBA10CF901'], label='LBA10CF901')
    plt.ylabel('X1')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(412)
    plt.plot(states['JEA10CL901'], label='JEA10CL901')
    plt.plot(12.2*np.ones(len(states['JEA10CL901'])), label='Setpoint')
    plt.ylabel('X2')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(413)
    plt.plot(states['LAB60CF901'], label='LAB60CF901')
    plt.ylabel('X3')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(414)
    plt.plot(states['LAB60CF001A'], label='LAB60CF001A')
    plt.ylabel('U- Control')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

X = states()
#graphics(df)
#u_fit(df)
identify_model(X)




