import os, sys

# Runs non-NBRA calculations
for istate_traj in [0, 2]:
    os.system(F"python main-namd.py --istate_traj={istate_traj}")


# Runs DATA gen
for istate_traj in [0, 2]:
    os.system(F"python main-data-gen.py --istate_traj={istate_traj}")


# Runs NBRA-FSSH calculations
for istate_traj in [0, 2]:
    for method_indx in [0]:
        for params_option in [-1]: 
            os.system(F"python nbra-standard.py --istate_traj={istate_traj} --method_indx={method_indx} --params_option={params_option}")


# Runs TC-NBRA-FSSH calculations
for istate_traj in [0, 2]:
    for method_indx in [1]:
        for params_option in range(10): 
            os.system(F"python nbra-standard.py --istate_traj={istate_traj} --method_indx={method_indx} --params_option={params_option}")
