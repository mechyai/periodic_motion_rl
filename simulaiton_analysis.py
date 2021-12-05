import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np

states = pd.read_pickle('Experimental-Data/smallSample_v0_v0_TD3_300_300_1_states.csv')

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


# drop first row
states = states.iloc[1:, :]
# select span
# start = 300
# stop = 700
# states = states.iloc[:,start:stop]

# POINCARE SECTION
period = 30 # indices, discovered empirically
period_start = 31  # indices
time_events = np.arange(period_start, 300, period)
state_events = states.iloc[time_events, :]

# TIMING
timestep = states['Timestep']


# POSITION
# front leg
ffoot_pos = states['ff_pos']
fshin_pos = states['fs_pos']
fthigh_pos = states['ft_pos']
# back leg
bfoot_pos = states['bf_pos']
bshin_pos = states['bs_pos']
bthigh_pos = states['bt_pos']

# VELOCITY
# front leg
ffoot_vel = states['ff_vel']
fshin_vel = states['fs_vel']
fthigh_vel = states['ft_vel']
# back leg
bfoot_vel = states['bf_vel']
bshin_vel = states['bs_vel']
bthigh_vel = states['bt_vel']


plt.close('all')

# POSITION 3D Phase Plot
fig_pos = plt.figure()
ax_pos = plt.axes(projection='3d')
# 3D plotting
ax_pos.plot3D(fthigh_pos, fshin_pos, ffoot_pos, 'o-', linewidth=1, markersize=1, label='Front Leg')
ax_pos.plot3D(bthigh_pos, bshin_pos, bfoot_pos, 'o-', linewidth=1, markersize=1, label='Back Leg')
ax_pos.plot3D(state_events['ft_pos'], state_events['fs_pos'], state_events['ff_pos'], 'o')
ax_pos.plot3D(state_events['bt_pos'], state_events['bs_pos'], state_events['bf_pos'], 'o')

# plot labeling and config
ax_pos.set_title('Leg Joint Position')
ax_pos.set_xlabel('Ankle Pos (rad)')
ax_pos.set_ylabel('Knee Pos (rad)')
ax_pos.set_zlabel('Hip Pos (rad)')
ax_pos.legend()
ax_pos.grid(True)

# VELOCITY 3D Phase Plot
fig_vel = plt.figure()
ax_vel = plt.axes(projection='3d')


# 3D plotting
ax_vel.plot3D(fthigh_vel, fshin_vel, ffoot_vel, 'o-', linewidth=1, markersize=1,  label='Front Leg')
ax_vel.plot3D(bthigh_vel, bshin_vel, bfoot_vel, 'o-', linewidth=1, markersize=1, label='Back Leg')
ax_vel.plot3D(state_events['ft_vel'], state_events['fs_vel'], state_events['ff_vel'], 'o')
ax_vel.plot3D(state_events['bt_vel'], state_events['bs_vel'], state_events['bf_vel'], 'o')
# plot labeling and config
ax_vel.set_title('Leg Joint Vel')
ax_vel.set_xlabel('Ankle Vel (rad/s)')
ax_vel.set_ylabel('Knee Vel (rad/s)')
ax_vel.set_zlabel('Hip Vel (rad/s)')
ax_vel.legend()
ax_vel.grid(True)

# POSITION vs Time 2D Line Plot
# front leg
fig_f_tpos = plt.figure()
# plt.plot(timestep, ffoot, label='Ankle')
plt.plot(timestep, fshin_pos, label='Knee')
# plt.plot(timestep, fthigh, label='Hip')
plt.legend()
plt.title('Front Leg')
plt.xlabel('Timestep')
plt.ylabel('Pos (rad)')
plt.grid(True)
# back leg
fig_b_tpos = plt.figure()
# plt.plot(timestep, bfoot, label='Ankle')
plt.plot(timestep, bshin_pos, label='Knee')
# plt.plot(timestep, bthigh, label='Hip')
plt.legend()
plt.title('Back Leg')
plt.xlabel('Timestep')
plt.ylabel('Pos (rad)')
plt.grid(True)



