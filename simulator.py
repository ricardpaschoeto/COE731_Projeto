import do_mpc

def simulator(model):
    # Obtain an instance of the do-mpc simulator class
    # and initiate it with the model:
    simulator_ = do_mpc.simulator.Simulator(model)

    # Set parameter(s):
    simulator_.set_param(t_step = 0.01)

    # Setup simulator:
    simulator_.setup()

    return simulator_