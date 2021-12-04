import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

states = pd.read_pickle('Experimental-Data/testExp_v0_v0_TD3_500_500_5_states.csv')

"""
state_labels = ['Timestep',  # units in [m] and [rad]
                  'x_pos', 'z_pos', 'y_pos',  # torso body frame cartesian position
                  'bt_pos', 'bs_pos', 'bf_pos',  # back t(high), s(hin), f(oot) joint angular position
                  'ft_pos', 'fs_pos', 'ff_pos',  # front t(high), s(hin), f(oot) joint angular position
                  'x_vel', 'z_vel', 'y_vel',  # torso body frame cartesian position
                  'bt_vel', 'bs_vel', 'bf_vel',  # back t(high), s(hin), f(oot) joint angular velocity
                  'ft_vel', 'fs_vel', 'ff_vel']  # front t(high), s(hin), f(oot) joint angular velocity

action_labels =['Timestep',  # units in [N-m]
                  'bt_torq', 'bs_torq', 'bf_torq',  # back t(high), s(hin), f(oot) joint torque
                  'ft_torq', 'fs_torq', 'ff_torq']  # front t(high), s(hin), f(oot) joint torque
"""

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
foot_z = states['bf_pos']
shin_y = states['bs_pos']
thigh_x = states['bt_pos']

ax.plot3D(thigh_x, shin_y, foot_z)

# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');