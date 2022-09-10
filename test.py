import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso

from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
import pysindy as ps

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Control input
def u_fun(t):
    return np.column_stack([np.sin(2 * t), t ** 2])

# Generate measurement data
dt = .002

t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27]
x_train = solve_ivp(lorenz_control, t_train_span, x0_train,
                    t_eval=t_train, args=(u_fun,), **integrator_keywords).y.T
u_train = u_fun(t_train)

# Instantiate and fit the SINDYc model
model = ps.SINDy()
model.fit(x_train, u=u_train, t=dt)
model.print()

t_test = np.arange(0, 15, dt)
t_test_span = (t_test[0], t_test[-1])
u_test = u_fun(t_test)
x0_test = np.array([8, 7, 15])
x_test = solve_ivp(lorenz_control, t_test_span, x0_test,
                   t_eval=t_test, args=(u_fun,), **integrator_keywords).y.T
u_test = u_fun(t_test)

# Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, u=u_test, t=dt))

# Evolve the new initial condition in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test, u=u_fun)