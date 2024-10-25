from model import model
from mpc import mpc
from simulator import simulator
import do_mpc
import numpy as np
import matplotlib.pyplot as plt

model = model()
mpc = mpc(model, 100, 100)
simulator = simulator(model)

x0 = np.array([500, 12.2, 0., 0.]).reshape(-1,1)

simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

# Initialize graphic:
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(3, sharex=True)
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot fluxo de vapor e nível:
    g.add_line(var_type='_x', var_name='steam_flux', axis=ax[0])
    g.add_line(var_type='_x', var_name='level', axis=ax[1])

    # Plot fluxo de água:
    g.add_line(var_type='_u', var_name='water_flux', axis=ax[2])


ax[0].set_ylabel('Steam Flux [kg/s]')
ax[0].grid(True)
ax[1].set_ylabel('level [m]')
ax[1].grid(True)
ax[2].set_ylabel('feedwater flux [kg/s]')
ax[2].grid(True)

## Running Simulator

# u0 = np.zeros((1,1))
# for k in range(120):
#     simulator.make_step(u0)

# sim_graphics.plot_results()
# sim_graphics.reset_axes()
# plt.show()

## Running MPC

# u0 = mpc.make_step(x0)

# sim_graphics.clear()
# mpc_graphics.plot_predictions()
# mpc_graphics.reset_axes()
# plt.show()

## ## Running loop: MPC + Simulator

simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

for i in range(20):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
plt.show()