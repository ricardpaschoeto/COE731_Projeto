import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pysindy.feature_library import CustomLibrary
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import pysindy as ps
import sympy as smp
from sympy.abc import s,t
from sklearn.kernel_ridge import KernelRidge
import control as ctrl
from scipy.signal import lti, lsim

# Estados:
# Temperatura e pressão da água interna ao GV
# Fluxo de Vapor - x1
# Pressão de Vapor principal - x2
# Entrada de Controle u_hp = fluxo de água de alimentação
# Pertubação u_hp = fluxo de água de alimentação corrigido e realimentado no controle
# Saída y = Nível medido

dt = 0.01

def load_data(path):
    df = pd.read_csv(path)

    return df

def data_conditioning(tmin,tmax):
    # Carregamento dos dados
    df = load_data('data_gv10.csv')
    x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
    y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

    # # Operações sobre os estados so sistema (pré-processamneto e ajustes)
    K_pressure =  np.polyfit(x,y,4)
    press_corr01 = np.polyval(K_pressure,  df['LBA10CP001'])*df['LBA10CF001A']
    press_corr02 = np.polyval(K_pressure,  df['LBA10CP951A'])*df['LBA10CF001B']
    df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

    n = len(df["Data_Hora"])
    t = np.linspace(0, n*dt, num=n)
    df["Data_Hora"] = t

    data = {'t':df["Data_Hora"],
            'x1':df['LBA10CF001A'],
            'x2':df['LBA10CP951A'],
            'y':df['JEA10CL901'],
            'u_hp':df['LAB60CF001A'],
            'u_hp_corr':df['LAB60CF901'],
            'x1_corr':df['LBA10CF901']          
            }

    df_processed = pd.DataFrame(data)

    # scaler = MinMaxScaler()
    # df_normalized = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
    df_sliced = df_processed.loc[tmin:tmax, ['t', 'x1', 'x2', 'y', 'u_hp', 'u_hp_corr', 'x1_corr']]

    return df_sliced

def ss_sim(df):
    Tn = 5.71
    Fg = 4.00
    Th = 0.71
    tau = 4.29
    Tg = 1.429
    Tint = 20

    # Define system matrices
    A = np.array([[0, 0, 0, 1/Tn], [0, -1/Th, 0, -1/Tn],[0,  0, -1/Tg, 0],[0, 0, 0, -1/tau]])
    B = np.array([0, 0, 0, 1/tau]).reshape((4,1))
    F = np.array([-1/Tn, 0, (1+Fg)/Tn, 0]).reshape((4,1))
    C = np.array([[1, 1, 1, 0],[Tn/Tint, 0, 0, tau/Tint]])
    B_dist = np.hstack((B,F))
    D = np.zeros((C.shape[0],B_dist.shape[1]))

    # Create the state-space system
    sys = lti(A, B_dist, C, D)

    # Display the system
    print(sys)

    # Define the time vector and input signal
    u = df['u_hp'].values
    d = df['x1'].values
    t = df['t'].values
    U = np.vstack((u,d)).T

    # Simulate the system response
    t_out, y_out, x_out = lsim(sys, U=U, T=t)
    
    # Plot the results
    _, ax = plt.subplots(3, 1, figsize=(10,10))
    ax[0].plot(t_out, u)
    ax[1].plot(t_out, d)
    ax[2].plot(t_out, y_out)
    plt.show()

def internal_level_rates(w, st, tin):
    Tn = np.array([5.14, 8.00, 9.00, 6.29, 5.71, 5.71, 5.71])
    Fg = np.array([13.00, 18.00, 10.00, 4.00, 4.00, 4.00, 4.00])
    Th = np.array([24.29, 8.00, 4.29, 1.43, 1.14, 0.71, 0.71])
    tau = np.array([1.43, 1.43, 1.43, 4.29, 4.29, 4.29, 4.29])
    Tg = 1.429
    Tint = 10

    nge = []
    ngl = []
    qgv_list = []
    qef_list = []

    for ii in range(len(tau)):

        water, steam = w, st

        Qgv = (1/(Tn[ii]*s)) * (1 - Fg[ii]*Tg*s)/(1 + Tg*s)
        Qef = (1/(Tn[ii]*s)) * (1/(Th[ii]*tau[ii]*s**2 + (Th[ii] + tau[ii])*s + 1))

        qgv = smp.inverse_laplace_transform(Qgv, s, t)
        qgv_time  = [qgv.subs({t:tv}) for tv in tin.values]
        qef = smp.inverse_laplace_transform(Qef, s, t)
        qef_time  = [qef.subs({t:tv}) for tv in tin.values]

        f1 = np.convolve(qgv_time, steam, mode='same')        
        f2 = np.convolve(qef_time, water, mode='same')        

        qgv_list.append(qgv_time)
        qef_list.append(qef_time)

        nge_v = f2 - f1     

        Ngl = (1/(Tint*s))
        ngl_t = smp.inverse_laplace_transform(Ngl, s, t)
        ngl_v_conv = [ngl_t.subs({t:tv}) for tv in tin.values] 
        ngl_v = np.convolve(ngl_v_conv, (water - steam), mode='same')               

        nge.append(nge_v)
        ngl.append(ngl_v)

    return nge, ngl, qef_list, qgv_list

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

def identify_y(df):
    model = KernelRidge(alpha=0.1, kernel='laplacian', gamma=0.1)
    model.fit(df['t'].values.reshape(-1,1), df['y'].values)

    y_pred = model.predict(df['u_hp'].values.reshape(-1,1))

    # Evaluate the model
    mse = root_mean_squared_error(df['y'], y_pred)
    r2 = r2_score(df['y'], y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Plot the results
    plt.scatter(df['t'], df['y'], color='blue', label='Actual')
    plt.plot(df['t'], y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def identify_model(df):

    opts = ['SSR', 'SR3', 'STLSQ']
    x_train, x_test = train_test_split(df, train_size=0.8, shuffle=True)

    # Entrada: Dados de Teste e Treinamento
    u_train = x_train.loc[:,["u_hp"]].to_numpy()
    u_test = x_test.loc[:,["u_hp"]].to_numpy()

    # X: Dados de Treinamento e Teste
    xs_train =  x_train.loc[:,["x1", "x2"]].to_numpy()
    xs_test = x_test.loc[:,["x1", "x2"]].to_numpy()

    # Initialize custom SINDy library so that we can have x_dot inside it.
    library_functions = [
         lambda x : x,
         lambda x : x ** 2,
         lambda x : np.sin(x),
         lambda x : np.cos(x),
    ]

    feature_libs = [
        CustomLibrary(library_functions=library_functions),
        ps.PolynomialLibrary(degree=2), 
        ps.FourierLibrary(n_frequencies=1), 
        ps.IdentityLibrary()
    ]

    _, axs = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
    for opt in opts:
        optimizer_ = optimizer(opt)

        for lib in feature_libs:
            model = ps.SINDy(
                feature_names = ["x1", "x2", "u"],
                optimizer=optimizer_,
                feature_library=lib,
                differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={'window_length':11})
            )

            print(opt, lib)
            model.fit(x=xs_train, u=u_train)
            model.print()
            # Compare SINDy-predicted derivatives with finite difference derivatives
            print("Model score: %f" % model.score(x=xs_test, u=u_test))
            print('=========================')

            # # Compute derivatives with a finite difference method, for comparison
            # x_dot_train_computed  = model.differentiate(xs_train, dt)
            # x_dot_test_computed  = model.differentiate(xs_test, dt)

            # # Predict derivatives using the learned model
            # x_dot_train_predicted  = model.predict(xs_train, u=u_train)
            # x_dot_test_predicted  = model.predict(xs_test, u=u_test)

            # for i in range(2):
            #     axs.plot(x_dot_train_computed[:, i], 'g', label='trained numerical derivative - ' + opt)
            #     axs.plot(x_dot_train_predicted[:, i],'b--', label='trained model prediction - ' + opt)
            #     axs.plot(x_dot_test_computed[:, i], 'k', label='tested numerical derivative - ' + opt)            
            #     axs.plot(x_dot_test_predicted[:, i],'r--', label='tested model prediction - ' + opt)
            #     axs.legend(loc='center')
            #     axs.set(xlabel='t', ylabel='$\dot x_{}$'.format(i+1))
            # plt.show()

def graphics(df):
    _, ax = plt.subplots(5, 1, figsize=(10,10))
    
    ax[0].plot(df["x1"], label='Steam Flux')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)

    ax[1].plot(df["x2"], label='Steam Pressure')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    ax[2].plot(df['y'], label='Level')
    ax[2].legend(loc='upper right')
    ax[2].grid(True)

    ax[3].plot(df['u_hp'], label='Water flux')
    ax[3].grid(True)
    ax[3].legend(loc='upper right')

    ax[4].plot(df['u_hp_corr'], label='Water Flux - corrected')
    ax[4].legend(loc='upper right')
    ax[4].grid(True)

    plt.show()

def plot_levels(nge, ngl):
    _, ax = plt.subplots(7, 2, figsize=(10,10))
    ii = 0
    for lvl1, lvl2 in zip(nge, ngl):
        ax[ii, 0].plot(lvl1, label='nge')
        ax[ii, 1].plot(lvl2, label='ngl')
        ax[ii, 0].grid(True)
        ax[ii, 1].grid(True)
        ii = ii + 1

    plt.show()


df = data_conditioning(2500, 4200)
#identify_y(df)

#water_steam_regression(df['u_hp'], df['x1'], df['t'])
# nge, ngl, qef_list, qgv_list = internal_level_rates(df['u_hp'], df['x1'], df['t'])
# norm_nge_arr = (np.array(nge) - np.min(nge)) / (np.max(nge) - np.min(nge))
# norm_ngl_arr = (np.array(ngl) - np.min(ngl)) / (np.max(ngl) - np.min(ngl))
# norm_y_arr = (np.array(df['y']) - np.min(df['y'])) / (np.max(df['y']) - np.min(df['y']))
# norm_water_arr = (np.array(df['u_hp']) - np.min(df['u_hp'])) / (np.max(df['u_hp']) - np.min(df['u_hp']))
# norm_steam_arr = (np.array(df['x1']) - np.min(df['x1'])) / (np.max(df['x1']) - np.min(df['x1']))
# norm_qef_arr = (np.array(qef_list) - np.min(qef_list)) / (np.max(qef_list) - np.min(qef_list))
# norm_qgv_arr = (np.array(qgv_list) - np.min(qgv_list)) / (np.max(qgv_list) - np.min(qgv_list))
# _, ax = plt.subplots(4, 1, figsize=(10,10))
# ax[0].plot(norm_qef_arr[6,:], label='qge')
# ax[1].plot(norm_qgv_arr[6,:], label='qgl')
# ax[2].plot(norm_water_arr, label='water')
# ax[3].plot(norm_steam_arr, label='steam')
# plt.show()
#plot_levels(norm_qef_arr, norm_qgv_arr)
#graphics(df)
ss_sim(df)