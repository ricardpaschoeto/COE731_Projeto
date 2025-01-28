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

        # Input Training data
        du_train = self.df.loc[:,["d", "u"]].to_numpy()
        u_train = self.df.loc[:,["u"]].to_numpy()
        
        # Training time
        t_timestamp = self.df.loc[:,["t"]].to_numpy()
        t_train = np.arange(0, len(t_timestamp), 1)        

        # State training data
        xs_train =  self.df.loc[:,["x"]].to_numpy()

        # Scan over the number of integration points and the number of subdomains
        n = 10
        errs = np.zeros((n))
        K_scan = np.linspace(20, 3000, n, dtype=int)
        library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
        library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
        for i, K in enumerate(K_scan):

            ode_lib = ps.WeakPDELibrary(
                library_functions=library_functions,
                function_names=library_function_names,
                spatiotemporal_grid=t_train,
                include_bias=True,
                is_uniform=True,
                K=K,
            )
            opt = ps.SR3(
                threshold=100,
                thresholder="l0",
                max_iter=1000,
                tol=1e-1,
                normalize_columns=True,
            )
            xs_dot_train_integral = ode_lib.convert_u_dot_integral(xs_train)

            # Instantiate and fit the SINDy model with the integral of xs_dot
            model = ps.SINDy(feature_names = ["x", "d", "u"], 
                            feature_library=ode_lib, 
                            optimizer=opt)
                            
            model.fit(x=xs_train, u=du_train, t=t_train)
            errs[i] = np.sqrt(
                (
                    np.sum((xs_dot_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
                    / np.sum(xs_dot_train_integral**2)
                )
                / xs_dot_train_integral.shape[0]
            )

        print("weak model")
        model.print()

        plt.title("Convergence of weak SINDy, hyperparameter scan", fontsize=12)
        plt.plot(K_scan, errs)
        plt.xlabel("Number of subdomains", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.show()

        x0 = [12]
        level_sim = model.simulate(x0, t_train, u_train)
        plot_kws = dict(linewidth=2)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(t_train, xs_train[:, 0], "r", label="level", **plot_kws)
        axs[0].plot(t_train, level_sim[:, 0], "k--", label="model", **plot_kws)
        axs[0].legend()
        axs[0].set(xlabel="t", ylabel="levels")
        fig.show()

def main():
    tmin = 11000
    tmax = 23000
    # et = ET('data_gv10.csv', tmin, tmax, True)
    # df = et.data_conditioning()
    # et.graphics(df)
    model = Model('data_gv10.csv', tmin, tmax, True)
    model.identify_model()

if __name__ == "__main__":
    main()