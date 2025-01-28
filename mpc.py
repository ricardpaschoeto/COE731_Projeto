import do_mpc
import numpy as np

def mpc(model, weights_cost, weights_final, R):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    setpoint = 12.2 # meters
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
    lterm = weights_cost * (model.x ** 2 + model.u ** 2 + model.d ** 2)
    mterm = weights_final*(model.x ** 2 + model.u ** 2 + model.d ** 2)
    rterm = R * 

    mpc_.set_objective(mterm=mterm, lterm=lterm)
    mpc_.set_rterm(water_flux = 100) # Scaling for quad. cost.

    # State and input bounds:
    mpc_.bounds['lower', '_x', 'steam_flux'] = 0
    mpc_.bounds['upper', '_x', 'steam_flux'] = 555.55

    mpc_.bounds['lower', '_x', 'level'] = 8.0
    mpc_.bounds['upper', '_x', 'level'] = 14.55

    mpc_.bounds['lower', '_u', 'water_flux'] = 0
    mpc_.bounds['upper', '_u', 'water_flux'] = 555.55

    mpc_.setup()

    return mpc_