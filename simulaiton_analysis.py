import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

states = pd.read_pickle('Experimental-Data/testExp_v0_v0_TD3_500_500_1_states.csv')

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

# POSITION 3D Phase Plot
fig_pos = plt.figure()
ax_pos = plt.axes(projection='3d')
# front leg
ffoot_z = states['ff_pos']
fshin_y = states['fs_pos']
fthigh_x = states['ft_pos']
# back leg
bfoot_z = states['bf_pos']
bshin_y = states['bs_pos']
bthigh_x = states['bt_pos']

# 3D plotting
ax_pos.plot3D(fthigh_x, fshin_y, ffoot_z, 'o-', linewidth=1, markersize=1, label='Front Leg')
ax_pos.plot3D(bthigh_x, bshin_y, bfoot_z, 'o-', linewidth=1, markersize=1, label='Back Leg')
# plot labeling and config
ax_pos.set_title('Leg Joint Position')
ax_pos.set_xlabel('Ankle Angle (rad)')
ax_pos.set_ylabel('Knee Angle (rad)')
ax_pos.set_zlabel('Hip Angle (rad)')
ax_pos.legend()
ax_pos.grid(True)

# VELOCITY 3D Phase Plot
fig_vel = plt.figure()
ax_vel = plt.axes(projection='3d')
# front leg
ffoot_z = states['ff_vel']
fshin_y = states['fs_vel']
fthigh_x = states['ft_vel']
# back leg
bfoot_z = states['bf_vel']
bshin_y = states['bs_vel']
bthigh_x = states['bt_vel']

# 3D plotting
ax_vel.plot3D(fthigh_x, fshin_y, ffoot_z, 'o-', linewidth=1, markersize=1,  label='Front Leg')
ax_vel.plot3D(bthigh_x, bshin_y, bfoot_z, 'o-', linewidth=1, markersize=1, label='Back Leg')
# plot labeling and config
ax_vel.set_title('Leg Joint Velocity')
ax_vel.set_xlabel('Ankle Velocity (rad/s)')
ax_vel.set_ylabel('Knee Velocity (rad/s)')
ax_vel.set_zlabel('Hip Velocity (rad/s)')
ax_vel.legend()
ax_vel.grid(True)

# POSITION vs Time 2D Line Plot
timestep = states['Timestep']

fig_tpos = plt.figure()
plt.plot(timestep, ffoot_z)
