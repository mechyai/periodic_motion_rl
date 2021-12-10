import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np

states = pd.read_pickle('Experimental-Data/touch_v0_v0_TD3_4700_300_1_states.csv')

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


def local_minimum_indices(data_list, ceiling, min_index, max_index):
    """Take in an array, and returns all indices of local mins under a given value ceiling."""
    local_min_index = []
    data_pre = data_list[0]
    data = data_list[1]
    for i, data_post in enumerate(data_list):
        if i > 1:
            if data_pre > data < data_post:  # unique local min
                if data < ceiling and min_index < i < max_index:
                    local_min_index.append(i-1)
            data_pre = data
            data = data_post
    return local_min_index


def rising_edge_indices(data_list):
    """Takes in array, and returns all indices of rising signal from 0 to greater than 0."""
    rising_index = []
    for i, data in enumerate(data_list):
        if data_list[i-1] == 0 and data > 0:
            rising_index.append(i)
    return rising_index


def average_trajectory(trajectory_df, column_names, event_indices):
    pass

# drop first row
states = states.iloc[1:, :]
# select desirable periodic span
start = 0
stop = 500
states = states.iloc[start:stop]

# POINCARE SECTION
# Time-Based Event
# period = 30  # indices, discovered empirically
# period_start = 31  # indices
# time_events = np.arange(period_start, 300, period)
# state_events = states.iloc[time_events, :]
# State-Based Event
btoe_pos = states['btoe_height']  # toe contact with ground
# ground_contact_indices = local_minimum_indices(list(btoe_pos), 0.025, start, stop)  # find contacts
ground_contact_indices = rising_edge_indices(list(btoe_pos))  # find contacts

state_events = states.iloc[ground_contact_indices]
first_event = states.iloc[ground_contact_indices[0]].to_numpy()  # first contact, use as phase event -> Poincare section
states_np = states.to_numpy()
first_event = states_np[ground_contact_indices[0]]
states['Norm'] = np.linalg.norm(states_np[:,2:]-first_event[2:], axis=1)  # ! ignore timestep & x_pos column for norm
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
# ax_pos.plot3D(bthigh_pos, bshin_pos, bfoot_pos, 'o-', linewidth=1, markersize=1, label='Back Leg')
ax_pos.plot3D(state_events['ft_pos'], state_events['fs_pos'], state_events['ff_pos'], 'o', color='k', markersize=3, label='Back Leg Contact')
# ax_pos.plot3D(state_events['bt_pos'], state_events['bs_pos'], state_events['bf_pos'], 'o', color='k', markersize=3)
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
ax_vel.plot3D(state_events['ft_vel'], state_events['fs_vel'], state_events['ff_vel'], 'o', color='k', markersize=3, label='Back Leg Contact')
ax_vel.plot3D(state_events['bt_vel'], state_events['bs_vel'], state_events['bf_vel'], 'o', color='k', markersize=3)
# plot labeling and config
ax_vel.set_title('Leg Joint Vel')
ax_vel.set_xlabel('Ankle Vel (rad/s)')
ax_vel.set_ylabel('Knee Vel (rad/s)')
ax_vel.set_zlabel('Hip Vel (rad/s)')
ax_vel.legend()
ax_vel.grid(True)

# STATE NORM vs Time 2D Line Plot
# norm of distance from first toe contact with ground
fig_norm = plt.figure()
plt.plot(timestep, states['Norm'])
plt.title('State Norm from 1st Toe Contact')
plt.xlabel('Timestep')
plt.ylabel('Vector 2-Norm')
plt.grid(True)

# POSITION vs Time 2D Line Plot
# TOE - for Poincare section (and motion period)
fig_b_toepos = plt.figure()
plt.plot(timestep, btoe_pos, label='Toe Height')
plt.legend()
plt.title('Back Toe Position')
plt.xlabel('Timestep')
plt.ylabel('Height (m)')
plt.grid(True)

# POSITION vs Time 2D Line Plot
# front leg
fig_f_tpos = plt.figure()
plt.plot(timestep, ffoot_pos, label='Ankle')
plt.plot(timestep, fshin_pos, label='Knee')
plt.plot(timestep, fthigh_pos, label='Hip')
plt.legend()
plt.title('Front Leg Joint Positions')
plt.xlabel('Timestep')
plt.ylabel('Pos (rad)')
plt.grid(True)
# back leg
fig_b_tpos = plt.figure()
plt.plot(timestep, bfoot_pos, label='Ankle')
plt.plot(timestep, bshin_pos, label='Knee')
plt.plot(timestep, bthigh_pos, label='Hip')
# plt.plot(timestep, btoe_pos, label='Toe Height')
plt.legend()
plt.title('Back Leg Joint Positions')
plt.xlabel('Timestep')
plt.ylabel('Pos (rad)')
plt.grid(True)



