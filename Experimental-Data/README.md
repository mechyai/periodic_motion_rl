This directory contains experimental .csv data from the HalfCheetah MuJoCo simulation. Each experiment produces two Pandas Dataframes,
one for the state space and action space. 

The file labeling is described as `name_env-ver_xml-ver_algo_sim-len_sim-start_frame-skip_actions/states.csv`

- name: arbitrary experiment name, if desired
- env-ver: which custom HalfCheetah-Gait OpenAI gym environment is being used. Ex: v0 from half_cheetah_gait_v0.py
- xml_ver: which MuJoCo XML model is being used Ex: v0 from half_cheetah_gait_v0.xml
- algo: abbreviated name of deep RL algorithm used
- sim-len: length of simulation. Total lengh of simulation ran is sim-len + sim_start. Rows of data is (sim-len * frame-skip) 
- sim-start: the simulation timestep when data collection begins. This is used to skip transients, if desired.
- frame-skip: The amount of simulation timesteps between each RL-MDP timestep/transaction. The renderings and agent act/observe happens at the rate of the frame-skip.
- actions/states: determines whether this is state or action data.

EX: exp1_v0_v0_ppo_950_50_1_action.csv