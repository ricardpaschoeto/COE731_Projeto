import numpy as np
import do_mpc
from casadi import *

def model():
    # Obtain an instance of the do-mpc model class
    # and select time discretization:
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Introduce new states, inputs and other variables to the model, e.g.:
    x1 = model.set_variable(var_type='_x', var_name='steam_flux', shape=(1,1))
    x2 = model.set_variable(var_type='_x', var_name='level', shape=(1,1))

    dx = model.set_variable(var_type='_x', var_name='dx', shape=(2,1))

    u = model.set_variable(var_type='_u', var_name='water_flux')

    # Model parameters
    p1 = np.array([549.713, -887.318, -685.325, -0.131, -43.116, 0.132, -19416.948, 55.079, -0.034, 1191.063, -0.222, -68886.868, 2.106, -2.509, 11472.492, -2.795, 701601.729])
    p2 = np.array([3.251, -7.995, -3.998, -0.277, -167.465, 0.349, 10.237, 0.010, -573.396, 0.004, -0.023, 111.521, -0.012, 6115.374])

    # Set right-hand-side of ODE for all introduced states (_x).
    model.set_rhs('steam_flux', dx[0])
    model.set_rhs('level', dx[1])

    x_next = vertcat(
        p1[0]*x1 + p1[1]*x1 + p1[2]*u + p1[3]*(x1**2) + p1[4]*x1*x2   + p1[5]*x1*u + p1[6]*(x2**2) + p1[7]*x2*u    + p1[8]*(u**2)  + p1[9]*x2**3  + p1[10]*sin(x1) + p1[11]*sin(x2) + p1[12]*sin(u) + p1[13]*cos(x1) + p1[14]*cos(x2) + p1[15]*cos(x2) + p1[16],
        p2[0]*x1 + p2[1]*x2 + p2[2]*u + p2[3]*x1*x2   + p2[4]*(x2**2) + p2[5]*x2*u + p2[6]*(x2**3) + p2[7]*sin(x1) + p2[8]*sin(x2) + p2[9]*sin(u) + p2[10]*cos(x1) + p2[11]*cos(x2) + p2[12]*cos(u) + p2[13],
    )

    model.set_rhs('dx', x_next)

    # Setup model:
    model.setup()

    return model