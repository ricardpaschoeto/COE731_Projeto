import do_mpc
import numpy as np

def estimator(model):
    # Obtain an instance of the do-mpc MHE class
    # and initiate it with the model.
    # Optionally pass a list of parameters to be estimated.
    mhe = do_mpc.estimator.MHE(model)

    # Set parameters:
    setup_mhe = {
        'n_horizon': 20,
        't_step': 0.01,
        'meas_from_data': True,
    }
    mhe.set_param(**setup_mhe)

    # Set custom objective function
    # based on:
    v = mhe._v.cat
    stage_cost = v.T @ np.diag(np.array([1,1])) @ v

    # and (for the arrival cost):
    x_0 = mhe._x
    x_prev = mhe._x_prev

    dx = x_0.cat - x_prev.cat

    arrival_cost = dx.T@dx

    mhe.set_objective(stage_cost, arrival_cost)

    # Set bounds for states, parameters, etc.
    mhe.bounds['lower', '_x', 'steam_flux'] = 0
    mhe.bounds['upper', '_x', 'steam_flux'] = 555.55

    mhe.bounds['lower', '_x', 'level'] = 8.0
    mhe.bounds['upper', '_x', 'level'] = 14.55

    mhe.bounds['lower', '_u', 'water_flux'] = 0
    mhe.bounds['upper', '_u', 'water_flux'] = 555.55

    mhe.setup()

    return mhe