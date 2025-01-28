import numpy as np
import do_mpc
from casadi import *

import matplotlib.pyplot as plt
import matplotlib as mpl

def model():
    # Obtain an instance of the do-mpc model class
    # and select time discretization:
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Introduce new states, inputs and other variables to the model, e.g.:
    x = model.set_variable(var_type='_x', var_name='level', shape=(1,1))
    # one states for the desired (set) water flux:
    x_set = model.set_variable(var_type='_u', var_name='level_set')
    # Three additional states for the true water flux and steam flux:
    u = model.set_variable(var_type='_u', var_name='water_flux')
    d = model.set_variable(var_type='_u', var_name='steam_flux')

    # Model parameters
    p = np.array([0.003, -0.012, -0.011, 0.028, 0.009, -0.018, -0.165, 0.111])
 
    # Set right-hand-side of ODE for all introduced states (_x).
    x_dot = p[0] + p[1]*x + p[2]*d + p[3]*u + p[4]*x**2 + p[5]*d**2 + p[6]*u**2 + p[7]*d*u
    model.set_rhs('level', x_dot)

    # Setup model:
    model.setup()

    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc_ = do_mpc.controller.MPC(model)

    # Set parameters:
    setup_mpc = {
        'n_horizon': 20,
        'n_robust': 1,
        't_step': 0.01,
        'store_full_solution': True,
    }
    mpc_.set_param(**setup_mpc)

    # Configure objective function:
    lterm = (x ** 2)
    mterm = (x ** 2)
 
    mpc_.set_objective(mterm=mterm, lterm=lterm)
    mpc_.set_rterm(level_set = 100)

    # State and input bounds:
    mpc_.bounds['lower', '_u', 'steam_flux'] = 0
    mpc_.bounds['upper', '_u', 'steam_flux'] = 555.55

    mpc_.bounds['lower', '_x', 'level'] = 8.0
    mpc_.bounds['upper', '_x', 'level'] = 14.55

    mpc_.bounds['lower', '_u', 'level_set'] = 8.0
    mpc_.bounds['upper', '_u', 'level_set'] = 14.55

    mpc_.bounds['lower', '_u', 'water_flux'] = 0
    mpc_.bounds['upper', '_u', 'water_flux'] = 555.55

    mpc_.scaling['_x', 'level'] = 2
    mpc_.scaling['_u', 'water_flux'] = 2
    mpc_.scaling['_u', 'steam_flux'] = 2

    mpc_.setup()

    # Obtain an instance of the do-mpc simulator class
    # and initiate it with the model:
    simulator = do_mpc.simulator.Simulator(model)

    # Set parameter(s):
    simulator.set_param(t_step = 0.01)

    # Setup simulator:
    simulator.setup()

    # Creating the control loop
    x0 = np.array([12])
    u0 = np.array([12, 100, 500])

    simulator.x0 = x0
    simulator.u0 = u0
    mpc_.x0 = x0
    mpc_.u0 = u0

    mpc_.set_initial_guess()

    # Setting up the Graphic
    # Customizing Matplotlib:
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True

    mpc_graphics = do_mpc.graphics.Graphics(mpc_.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
    fig.align_ylabels()
    for g in [sim_graphics, mpc_graphics]:
        # Plot the level (x) on the first axis:
        g.add_line(var_type='_x', var_name='level', axis=ax[0])

        # Plot the set level (u_set) and disturbance (d) on the second axis:
        g.add_line(var_type='_u', var_name='level_set', axis=ax[1])
        g.add_line(var_type='_u', var_name='steam_flux', axis=ax[1])

    ax[0].set_ylabel('level [m]')
    ax[1].set_ylabel('water and steam flux [kg/s]')
    ax[1].set_xlabel('time [s]')

    u0 = np.zeros((3,1))
    for i in range(500):
        simulator.make_step(u0)

    sim_graphics.plot_results()
    sim_graphics.reset_axes()
    #plt.show()

    # Running the optimizer
    u0 = mpc_.make_step(x0)

    sim_graphics.clear()
    mpc_graphics.plot_predictions()
    mpc_graphics.reset_axes()
    plt.show()

model()