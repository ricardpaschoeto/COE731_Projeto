import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pysindy.feature_library import CustomLibrary
from sklearn.preprocessing import MinMaxScaler
import pysindy as ps
import sympy as smp
from sympy.abc import s,t
from scipy.signal import lti, lsim
from scipy.optimize import curve_fit
import scipy.signal

##############################################
# Class for Extration and transform Data     #
##############################################
class ET():
    def __init__(self, path, tmin, tmax, normalized=False):    
        self.df = pd.read_excel('data_gv10.xlsx')
        self.path = path
        self.tmin = tmin
        self.tmax = tmax
        self.normalized = normalized

    def data_conditioning(self):
        cols = self.df.columns.values
        t = self.df[cols[0]]

        data = {
                'd':self.df[cols[2]], # LBA10CF001A
                'u':self.df[cols[4]] # LBA60CF001A
                }

        df_processed = pd.DataFrame(data)
        df_time_col = pd.DataFrame({'t': t})
        df_x_col = pd.DataFrame({'x': self.df[cols[3]]}) # JEA10CL951A
        df_processed = pd.concat([df_processed, df_x_col], axis=1)
        if self.normalized:
            scaler = MinMaxScaler()
            df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns= ['d', 'u', 'x'])

        df_processed = pd.concat([df_processed, df_time_col], axis=1)
        df_processed['t'] = pd.to_datetime(df_processed['t'], format='%d/%m/%Y  %H:%M:%S')   
        if self.tmax != 0:
            df_processed = df_processed.loc[self.tmin:self.tmax, ['t', 'd', 'u', 'x']]

        # Sinal Filtrado
        df_processed['d'] = self.filtering(df_processed['d'], 3, 0.01)
        df_processed['u'] = self.filtering(df_processed['u'], 3, 0.01)
        df_processed['x'] = self.filtering(df_processed['x'], 3, 0.0015)

        # Média móvel
        df_processed['d'] = df_processed['d'].rolling(window=3).mean()
        df_processed['u'] = df_processed['u'].rolling(window=3).mean()
        df_processed['x'] = df_processed['x'].rolling(window=1).mean()

        df_processed = df_processed.dropna(axis=0)
        
        return df_processed

    def filtering(self, data, N, Wn):
        b, a = scipy.signal.butter(N, Wn)
        filtered = scipy.signal.filtfilt(b, a, data)

        return filtered

    def graphics(self, df):
        _, ax = plt.subplots(3, 1, figsize=(10,10))
        
        ax[0].plot(df['t'], df["d"], label='Steam Flux')
        ax[0].legend(loc='upper right')
        ax[0].grid(True)

        ax[1].plot(df['t'], df["u"], label='Main feedwater')
        ax[1].legend(loc='upper right')
        ax[1].grid(True)

        ax[2].plot(df['t'], df['x'], label='Level')
        ax[2].legend(loc='upper right')
        ax[2].grid(True)

        plt.show()

######################################################
# Class for Simulate the Dynamics of Steam generator #
# used in the paper - "Level Control in the Steam    #
# Generator of a Nuclear Power Plant"                #
######################################################
class SteamGenerator(ET):
    def __init__(self, path, tmin, tmax, normalized=False):
        super().__init__(path, tmin, tmax, normalized)
        self.df = super().data_conditioning()
        self.Tn = np.array([5.14, 8.00, 9.00, 6.29, 5.71, 5.71, 5.71])
        self.Fg = np.array([13.00, 18.00, 10.00, 4.00, 4.00, 4.00, 4.00])
        self.Th = np.array([24.29, 8.00, 4.29, 1.43, 1.14, 0.71, 0.71])
        self.tau = np.array([1.43, 1.43, 1.43, 4.29, 4.29, 4.29, 4.29])
        self.Tg = 1.429
        self.Tint = 20

    def ss_sim(self):
        # Define system matrices
        ii = 6
        A = np.array([[0, 0, 0, 1/self.Tn[ii]], [0, -1/self.Th[ii], 0, -1/self.Tn[ii]],[0,  0, -1/self.Tg, 0],[0, 0, 0, -1/self.tau[ii]]])
        B = np.array([0, 0, 0, 1/self.tau[ii]]).reshape((4,1))
        F = np.array([-1/self.Tn[ii], 0, (1 + self.Fg[ii])/self.Tn[ii], 0]).reshape((4,1))
        C = np.array([[1, 1, 1, 0],[self.Tn[ii]/self.Tint, 0, 0, self.tau[ii]/self.Tint]])
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
        y_real = self.df['x'].values
        U = np.vstack((u,d)).T

        # Initial Condition
        x0 = [12, 0, 0 ,0]

        # Simulate the system response
        t_out, y_out, x_out = lsim(sys, U=U, T=t, X0=x0)
        
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

    def internal_level_rates(self):
        nge = []
        ngl = []
        qgv = []
        qef = []

        ii = 6
        Hgv = (1/(self.Tn[ii]*s)) * (1 - self.Fg[ii]*self.Tg*s)/(1 + self.Tg*s)
        Hef = (1/(self.Tn[ii]*s)) * (1/(self.Th[ii]*self.tau[ii]*s**2 + (self.Th[ii] + self.tau[ii])*s + 1))

        hgv = smp.inverse_laplace_transform(Hgv, s, t)
        hgv_time  = [hgv.subs({t:tv}) for tv in self.df['t'].values]
        hef = smp.inverse_laplace_transform(Hef, s, t)
        hef_time  = [hef.subs({t:tv}) for tv in self.df['t'].values]

        f1 = np.convolve(hgv_time, self.df['x1'].values, mode='same')
        f2 = np.convolve(hef_time, self.df['u_hp'].values, mode='same')       

        qgv.append(f1)
        qef.append(f2)

        nge_v = f2 - f1     

        Hgl = (1/(self.Tint*s))
        hgl = smp.inverse_laplace_transform(Hgl, s, t)
        hgl_time = [hgl.subs({t:tv}) for tv in self.df['t'].values] 
        ngl_v = np.convolve(hgl_time, (self.df['u_hp'].values - self.df['x1'].values), mode='same')             

        nge.append(nge_v)
        ngl.append(ngl_v)

        return nge, ngl, qef, qgv

    def plot_levels(self, nge, f1, f2, df):
        _, ax = plt.subplots(6, 1, figsize=(10,10))

        ax[0].plot(nge[0], label='level')
        ax[0].legend()
        ax[1].plot(df['y'], label='measured level')
        ax[1].legend()
        ax[2].plot(df['u_hp'], label='water')
        ax[2].legend()
        ax[3].plot(df['x1'], label='steam')
        ax[3].legend()
        ax[4].plot(f1, label='qef')
        ax[4].legend()
        ax[5].plot(f2, label='qgv')
        ax[5].legend()

        plt.show()

############################################################
# Class for Simulate the Dynamics of Steam generator       #
# generated by data using SINDyc                           # 
############################################################  
class Model(ET):

    # Nível (narrow range) - x1                                      
    # Entrada de Controle u = fluxo de água de alimentação  
    # Pertubação d = fluxo de vapor principal 
    # e realimentado no controle                               
    # Saída y = Nível medido
                                       
    def __init__(self, path, tmin, tmax, normalized=False):
        super().__init__(path, tmin, tmax, normalized)
        self.df = super().data_conditioning()

    def optimizer(self, name):
        optimizer_ = None
        if name == 'SSR':ps.SSR(
            alpha=0.01,
            normalize_columns=True,
            max_iter=20,
        )
        if name == 'SR3':
            optimizer_ = ps.SR3(
                threshold=0.01,
                thresholder="L2",
                trimming_fraction=0,
                max_iter=30,
                tol=1e-8,
            )
        if name == 'STLSQ':
            optimizer_ = ps.STLSQ(
            threshold=0.01,
            fit_intercept=False,
            alpha=0.01,
        )
        
        return optimizer_

    def identify_y(self):
        def func(X, a, b, c):
            x, y, z = X
            return a * np.exp(-b * (x - y)) + c * z

        Xdata = np.vstack((range(len(self.df['x'])), self.df['d'], self.df['u']))
        popt, _ = curve_fit(func, Xdata, self.df['x'])
        print("Optimal parameters:", popt)

        # Generate fitted data
        zfit = func(Xdata, *popt)

        # Plot the original data and the fitted data
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['t'], self.df['x'], label='Data')
        plt.plot(self.df['t'], zfit, label='Fitted curve', color='red')
        plt.xlabel('t')
        plt.ylabel('z')
        plt.legend()
        plt.title('Multivariable Curve Fitting')
        plt.show()      

    def identify_model(self):

        opts = ['SSR', 'SR3', 'STLSQ']
        x_train, x_test = train_test_split(self.df, train_size=0.8, shuffle=True)

        # Entrada: Dados de Teste e Treinamento
        du_train = x_train.loc[:,["d", "u"]].to_numpy()
        du_test = x_test.loc[:,["d", "u"]].to_numpy()

        # X: Dados de Treinamento e Teste
        xs_train =  x_train.loc[:,["x"]].to_numpy()
        xs_test = x_test.loc[:,["x"]].to_numpy()

        # Initialize custom SINDy library so that we can have x_dot inside it.
        library_functions = [
            lambda x : x ** 3,
            lambda x : np.exp(x),
            lambda x : x ** 2,
            lambda x : x,
        ]

        feature_libs = [
            CustomLibrary(library_functions=library_functions),
            ps.PolynomialLibrary(degree=3), 
            ps.FourierLibrary(n_frequencies=1), 
            ps.IdentityLibrary()
        ]

        _, axs = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
        for opt in opts:
            optimizer_ = self.optimizer(opt)

            for lib in feature_libs:
                model = ps.SINDy(
                    feature_names = ["x", "d", "u"],
                    optimizer=optimizer_,
                    feature_library=lib,
                    differentiation_method=ps.FiniteDifference(order=1)
                )

                print(opt, lib)
                model.fit(x=xs_train, u=du_train)
                model.print()

                print("Model score: %f" % model.score(x=xs_test, u=du_test))
                print('=========================')

                # # Compute derivatives with a finite difference method, for comparison
                x_dot_train_computed  = model.differentiate(xs_train, 0.01)
                x_dot_test_computed  = model.differentiate(xs_test, 0.01)

                # Compare SINDy-predicted derivatives with finite difference derivatives
                # Predict derivatives using the learned model
                x_dot_train_predicted  = model.predict(xs_train, u=du_train)
                x_dot_test_predicted  = model.predict(xs_test, u=du_test)

                for i in range(1):
                    axs.plot(x_dot_train_computed[:, i], 'g', label='trained numerical derivative - ' + opt)
                    axs.plot(x_dot_train_predicted[:, i],'b--', label='trained model prediction - ' + opt)
                    axs.plot(x_dot_test_computed[:, i], 'k', label='tested numerical derivative - ' + opt)            
                    axs.plot(x_dot_test_predicted[:, i],'r--', label='tested model prediction - ' + opt)
                    axs.legend(loc='center')
                    axs.set(xlabel='t', ylabel='$\dot x_{}$'.format(i+1))
                plt.show()

def main():
    tmin = 11000
    tmax = 23000
    # et = ET('data_gv10.csv', tmin, tmax, True)
    # df = et.data_conditioning()
    # et.graphics(df)
    # nge, _, qef, qgv = sg.internal_level_rates()
    # sg.plot_levels(nge, qef[0], qgv[0], sg.df)
    model = Model('data_gv10.csv', tmin, tmax, True)
    model.identify_y()

if __name__ == "__main__":
    main()