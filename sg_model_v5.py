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
from scipy.signal import lti, lsim

##############################################
# Class for Extration and transform Data     #
##############################################
class ET():
    def __init__(self, dt, path, tmin, tmax, normalized=False):    
        self.dt = dt
        self.df = pd.read_csv(path)
        self.tmin = tmin
        self.tmax = tmax
        self.normalized = normalized 

    def data_conditioning(self):
        x = [0,	2,	4,	6,	8,	10,	13,	16,	22,	30,	41,	55,	70,	80,	90,	100]
        y = [0.129,	0.216,	0.267,	0.321,	0.362,	0.398,	0.447,	0.491,	0.569,	0.661,	0.771,	0.897,	1.0,	1.1,	1.16,	1.257]

        # # Operações sobre os estados so sistema (pré-processamneto e ajustes)
        K_pressure =  np.polyfit(x,y,4)
        press_corr01 = np.polyval(K_pressure,  self.df['LBA10CP001'])*self.df['LBA10CF001A']
        press_corr02 = np.polyval(K_pressure,  self.df['LBA10CP951A'])*self.df['LBA10CF001B']
        self.df['LBA10CF901'] = (press_corr01 + press_corr02)/2.0

        n = len(self.df["Data_Hora"])
        t = np.linspace(0, n*self.dt, num=n)
        self.df["Data_Hora"] = t

        data = {
                'x1':self.df['LBA10CF001A'],
                'x2':self.df['LBA10CP951A'],
                'u_hp':self.df['LAB60CF001A'],
                'u_hp_corr':self.df['LAB60CF901'],
                'x1_corr':self.df['LBA10CF901']          
                }

        df_processed = pd.DataFrame(data)
        df_time_col = pd.DataFrame({'t': t})
        df_y_col = pd.DataFrame({'y': self.df['JEA10CL901']})
        if self.normalized:
            scaler = MinMaxScaler()
            df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns= ['x1', 'x2', 'u_hp', 'u_hp_corr', 'x1_corr'])        
            
        df_processed = pd.concat([df_processed, df_time_col, df_y_col], axis=1)
        df_sliced = df_processed.loc[self.tmin:self.tmax, ['t', 'x1', 'x2', 'y', 'u_hp', 'u_hp_corr', 'x1_corr']]

        return df_sliced

######################################################
# Class for Simulate the Dynamics of Steam generator #
# used in the paper - "Level Control in the Steam    #
# Generator of a Nuclear Power Plant"                #
######################################################
class Steam(ET):
    def __init__(self, dt, path, tmin, tmax, normalized):
        super().__init__(dt, path, tmin, tmax, normalized)
        self.df = super().data_conditioning()
        self.Tn = np.array([5.14, 8.00, 9.00, 6.29, 5.71, 5.71, 5.71])
        self.Fg = np.array([13.00, 18.00, 10.00, 4.00, 4.00, 4.00, 4.00])
        self.Th = np.array([24.29, 8.00, 4.29, 1.43, 1.14, 0.71, 0.71])
        self.tau = np.array([1.43, 1.43, 1.43, 4.29, 4.29, 4.29, 4.29])
        self.Tg = 1.429
        self.Tint = 20

    def ss_sim(self):
        # Define system matrices
        A = np.array([[0, 0, 0, 1/self.Tn[6]], [0, -1/self.Th[6], 0, -1/self.Tn[6]],[0,  0, -1/self.Tg, 0],[0, 0, 0, -1/self.tau[6]]])
        B = np.array([0, 0, 0, 1/self.tau[6]]).reshape((4,1))
        F = np.array([-1/self.Tn[6], 0, (1 + self.Fg[6])/self.Tn[6], 0]).reshape((4,1))
        C = np.array([[1, 1, 1, 0],[self.Tn[6]/self.Tint, 0, 0, self.tau[6]/self.Tint]])
        B_dist = np.hstack((B,F))
        D = np.zeros((C.shape[0],B_dist.shape[1]))

        # Create the state-space system
        sys = lti(A, B_dist, C, D)

        # Display the system
        print(sys)

        # Define the time vector and input signal
        u = self.df['u_hp'].values
        d = self.df['x1'].values
        t = self.df['t'].values
        y_real = self.df['y'].values
        U = np.vstack((u,d)).T

        # Initial Condition
        x0 = [12, 0, 0 ,0]

        # Simulate the system response
        t_out, y_out, x_out = lsim(sys, U=U, T=t, X0=x0)
        yout_normalized = (y_out - np.min(y_out))/(np.max(y_out) - np.min(y_out))
        
        # Plot the results
        _, ax = plt.subplots(4, 1, figsize=(10,10))
        ax[0].plot(t_out, u, label="water flux")
        ax[0].legend(loc='upper right')
        ax[1].plot(t_out, d, label="steam flux")
        ax[1].legend(loc='upper right')
        ax[2].plot(t_out, y_out, label=["Nge", "Ngl"])
        ax[2].legend(loc='upper right')
        ax[3].plot(t_out, y_real, label="level measured")
        ax[3].legend(loc='upper right')
        plt.show()

    def internal_level_rates(self, w, st, tin):
        nge = []
        ngl = []
        qgv_list = []
        qef_list = []

        for ii in range(len(self.tau)):

            water, steam = w, st

            Qgv = (1/(self.Tn[ii]*s)) * (1 - self.Fg[ii]*self.Tg*s)/(1 + self.Tg*s)
            Qef = (1/(self.Tn[ii]*s)) * (1/(self.Th[ii]*self.tau[ii]*s**2 + (self.Th[ii] + self.tau[ii])*s + 1))

            qgv = smp.inverse_laplace_transform(Qgv, s, t)
            qgv_time  = [qgv.subs({t:tv}) for tv in tin.values]
            qef = smp.inverse_laplace_transform(Qef, s, t)
            qef_time  = [qef.subs({t:tv}) for tv in tin.values]

            f1 = np.convolve(qgv_time, steam, mode='same')        
            f2 = np.convolve(qef_time, water, mode='same')        

            qgv_list.append(qgv_time)
            qef_list.append(qef_time)

            nge_v = f2 - f1     

            Ngl = (1/(self.Tint*s))
            ngl_t = smp.inverse_laplace_transform(Ngl, s, t)
            ngl_v_conv = [ngl_t.subs({t:tv}) for tv in tin.values] 
            ngl_v = np.convolve(ngl_v_conv, (water - steam), mode='same')               

            nge.append(nge_v)
            ngl.append(ngl_v)

        return nge, ngl, qef_list, qgv_list

    def plot_levels(self, nge, ngl):
        _, ax = plt.subplots(7, 2, figsize=(10,10))
        ii = 0
        for lvl1, lvl2 in zip(nge, ngl):
            ax[ii, 0].plot(lvl1, label='nge')
            ax[ii, 1].plot(lvl2, label='ngl')
            ax[ii, 0].grid(True)
            ax[ii, 1].grid(True)
            ii = ii + 1

        plt.show()

############################################################
# Class for Simulate the Dynamics of Steam generator       #
# generated by data using SINDyc                           # 
############################################################  
class Model(ET):
    # Fluxo de Vapor - x1                                      
    # Pressão de Vapor principal - x2                          
    # Entrada de Controle u_hp = fluxo de água de alimentação  
    # Pertubação u_hp = fluxo de água de alimentação corrigido 
    # e realimentado no controle                               
    # Saída y = Nível medido
                                       
    def __init__(self, dt, path, tmin, tmax, normalized):
        super().__init__(dt, path, tmin, tmax, normalized)
        self.df = super().data_conditioning()

    def optimizer(self, name):
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

    def identify_y(self):
        model = KernelRidge(alpha=0.1, kernel='laplacian', gamma=0.1)
        model.fit(self.df['t'].values.reshape(-1,1), self.df['y'].values)

        y_pred = model.predict(self.df['u_hp'].values.reshape(-1,1))

        # Evaluate the model
        mse = root_mean_squared_error(self.df['y'], y_pred)
        r2 = r2_score(self.df['y'], y_pred)

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}")

        # Plot the results
        plt.scatter(self.df['t'], self.df['y'], color='blue', label='Actual')
        plt.plot(self.df['t'], y_pred, color='red', linewidth=2, label='Predicted')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def identify_model(self):

        opts = ['SSR', 'SR3', 'STLSQ']
        x_train, x_test = train_test_split(self.df, train_size=0.8, shuffle=True)

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
            optimizer_ = self.optimizer(opt)

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

    def graphics(self):
        _, ax = plt.subplots(5, 1, figsize=(10,10))
        
        ax[0].plot(self.df["x1"], label='Steam Flux')
        ax[0].legend(loc='upper right')
        ax[0].grid(True)

        ax[1].plot(self.df["x2"], label='Steam Pressure')
        ax[1].legend(loc='upper right')
        ax[1].grid(True)

        ax[2].plot(self.df['y'], label='Level')
        ax[2].legend(loc='upper right')
        ax[2].grid(True)

        ax[3].plot(self.df['u_hp'], label='Water flux')
        ax[3].grid(True)
        ax[3].legend(loc='upper right')

        ax[4].plot(self.df['u_hp_corr'], label='Water Flux - corrected')
        ax[4].legend(loc='upper right')
        ax[4].grid(True)

        plt.show()

def main():
    steam = Steam(0.01, 'data_gv10.csv', 2500, 4200, True)
    steam.ss_sim()

if __name__ == "__main__":
    main()