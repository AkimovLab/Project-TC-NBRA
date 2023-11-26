#!/usr/bin/env python
# coding: utf-8

########################################################
#
#   This code is for data generation - only 1 trajectory
#   but with a lot of data saved - to be used in the
#   NBRA-based workflows
#
########################################################

# ## 1. Generic setups
import sys
import cmath
import math
import os
import h5py
import matplotlib.pyplot as plt   # plots
import numpy as np
import time
import warnings

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv
import libra_py.models.Holstein as Holstein
import libra_py.models.Tully as Tully
from libra_py import dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
import libra_py.data_savers as data_savers

from recipes import ehrenfest_adi_nac, ehrenfest_adi_ld, ehrenfest_dia
from recipes import fssh, fssh_ssy, gfsh, gfsh_ssy, mssh, mssh_ssy, bcsh, bcsh_ssy
from recipes import ida, ida_ssy, sdm, sdm_ssy
from recipes import dish, mfsd
from recipes import dish_nbra, fssh_nbra, fssh2_nbra, gfsh_nbra, ida_nbra, mash_nbra, msdm_nbra


import argparse
parser = argparse.ArgumentParser(description='Data generation...')
parser.add_argument('--istate_traj', type=int) 
args = parser.parse_args() 



#from matplotlib.mlab import griddata
#%matplotlib inline 
warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})  # red       
colors.update({"12": "#FF4500"})  # orangered 
colors.update({"13": "#B22222"})  # firebrick 
colors.update({"14": "#DC143C"})  # crimson   
colors.update({"21": "#5e9c36"})  # green
colors.update({"22": "#006400"})  # darkgreen  
colors.update({"23": "#228B22"})  # forestgreen
colors.update({"24": "#808000"})  # olive      
colors.update({"31": "#8A2BE2"})  # blueviolet
colors.update({"32": "#00008B"})  # darkblue  
colors.update({"41": "#2F4F4F"})  # darkslategray

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13","23", "14", "24"]


# ## 1. Model Hamiltonians and parameters sets

def compute_model(q, params, full_id):
    model = params["model"]
    res = None

    if model==1:        
        res = Holstein.Holstein2(q, params, full_id) 
    elif model==2:
        res = Tully.chain_potential(q, params, full_id)
    else:
        pass #res = compute_model_nbra_files(q, params, full_id)            

    return res


# * Sets 1 to 4 - for the 2-level Holstein Hamiltonians. These are just parabolas with constant coupling.
# * Set 5 - for the 3-level chain potential from Tully-Parandekar 
model_params1 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.000}
model_params2 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.001}
model_params3 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.01}
model_params4 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0, -0.01], "x_n":[0.0,  0.5],"k_n":[0.002, 0.008],"V":0.001}

couplings = [  [ 0.0000, 0.002, 0.002 ],
               [ 0.002, 0.0000, 0.002 ],
               [ 0.002, 0.002, 0.0000 ] 
            ]
model_params5 = {"model":2, "model0":2, "nstates":3,
                 "timestep":0, "icond":0,
                 "E_n":[0.0, 0.001, 0.002], "x_n":[0.0, 1.0, 2.0], "k_n":[0.001, 0.002, 0.003],
                 "V":couplings,  "U0":0.01      }

all_model_params = [model_params1, model_params2, model_params3, model_params4, model_params5  ]


# Let's visualize these 5 models. Also refer to another tutorial for a more detailed description of the `plot_surfaces` function.
# Common setups
plot_params = {"figsize":[24, 6], "titlesize":24, "labelsize":28, "fontsize": 28, "xticksize":26, "yticksize":26,
               "colors": colors, "clrs_index": clrs_index,
               "prefix":F"case", "save_figures":1, "do_show":0,
               "plotting_option":1, "nac_idof":0, "show_nac_abs":1 }

# Model 1
plot_params.update( { "xlim":[-4, 5], "ylim":[-0.01, 0.06], "ylim2":[-2, 2] })
dynamics_plotting.plot_surfaces(compute_model, [ model_params1 ], [0, 1], -4.0, 5.0, 0.05, plot_params)

# Model 2
plot_params.update( { "xlim":[-1, 5], "ylim":[-0.01, 0.03], "ylim2":[-2, 2] })
dynamics_plotting.plot_surfaces(compute_model, [ model_params2 ], [0, 1], -4.0, 5.0, 0.05, plot_params)

# Model 3
plot_params.update( { "xlim":[-4, 5], "ylim":[-0.01, 0.06], "ylim2":[-0.3, 0.3] })
dynamics_plotting.plot_surfaces(compute_model, [ model_params3 ], [0, 1], -4.0, 5.0, 0.05, plot_params)

# Model 4
plot_params.update( { "xlim":[-4, 5], "ylim":[-0.01, 0.06], "ylim2":[-3, 3] })
dynamics_plotting.plot_surfaces(compute_model, [ model_params4 ], [0, 1], -4.0, 5.0, 0.05, plot_params)

# Model 5
plt_params = dict(plot_params)
plt_params.update( {  "ylim":[-0.025, 0.05], "xlim":[-5.0, 10.0], "ylim2":[-3, 3]} )
                
all_coords = [0.0, 0.0 ] # , 0.0, 0.0, 0.0, 0.0]
ndof = len(all_coords)
scan_coord = 0
dynamics_plotting.plot_surfaces(compute_model, [ model_params5 ], [0,1,2], -5, 10.0, 0.05, plt_params, ndof, scan_coord, all_coords)
scan_coord = 1
dynamics_plotting.plot_surfaces(compute_model, [ model_params5 ], [0,1,2], -5, 10.0, 0.05, plot_params, ndof, scan_coord, all_coords)


# Select the model
# 0 - Holstein, trivial crossing, 2 level
# 1 - Holstein, strong nonadiabatic, 2 level
# 2 - Holstein, adiabatic, 2 level
# 3 - Holstein, double crossing, strong nonadiabatic, 2 level
# 4 - 3-level N-dimensional chain potential model

#################################
model_indx = 4
################################

model_params = all_model_params[model_indx]

# ## 2. Choosing the Nonadiabatic Dynamics Methodology 

NSTATES = model_params["nstates"]
print(F"NSTATES = {NSTATES}")

dyn_general = { "nsteps":15000, "ntraj":1, "nstates":NSTATES,
                "dt":5.0, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),      
                "prefix":"adiabatic_md", "prefix2":"adiabatic_md",               
                "ensemble":0, 
                "quantum_dofs":[0], 
                "thermostat_dofs":[], 
                "constrained_dofs":[],
                "mem_output_level":4,
                "properties_to_save":[ "timestep", "time", "q", "p", "f", "Cadi", "Cdia", "Epot_ave", "Ekin_ave", "Etot_ave",
                                        "se_pop_adi", "se_pop_dia", "sh_pop_adi", "hvib_adi", "hvib_dia", "St", 
                                        "basis_transform", "D_adi"],
                "icond":0
              }

#################################
# Give the recipe above an index
method_indx = 4
#################################

if method_indx == 0:
    ehrenfest_dia.load(dyn_general); prf = "EHR_DIA"  # Ehrenfest, dia
elif method_indx == 1:
    ehrenfest_adi_nac.load(dyn_general);  prf = "EHR_ADI_NAC"  # Ehrenfest, adi with NACs    
elif method_indx == 2:
    ehrenfest_adi_ld.load(dyn_general);  prf = "EHR_DIA_LD"  # Ehrenfest, adi with LD
elif method_indx == 3:
    mfsd.load(dyn_general);  prf = "MFSD"  # MFSD
elif method_indx == 4:
    fssh.load(dyn_general); prf = "FSSH"  # FSSH
elif method_indx == 5:
    fssh_ssy.load(dyn_general); prf = "FSSH_SSY"  # FSSH + SSY
elif method_indx == 6:
    gfsh.load(dyn_general); prf = "GFSH"  # GFSH
elif method_indx == 7:
    gfsh_ssy.load(dyn_general); prf = "GFSH_SSY"  # GFSH + SSY
elif method_indx == 8:
    mssh.load(dyn_general); prf = "MSSH"  # MSSH
elif method_indx == 9:
    mssh_ssy.load(dyn_general); prf = "MSSH_SSY"  # MSSH + SSY    
elif method_indx == 10:
    bcsh.load(dyn_general); prf = "BCSH"  # BCSH    
elif method_indx == 11:
    bcsh_ssy.load(dyn_general); prf = "BCSH_SSY"  # BCSH + SSY
elif method_indx == 12:
    sdm.load(dyn_general); prf = "SDM_EDC"  # SDM with default EDC parameters
elif method_indx == 13:
    sdm_ssy.load(dyn_general); prf = "SDM_EDC_SSY"  # SDM with default EDC parameters + SSY
elif method_indx == 14:
    ida.load(dyn_general); prf = "IDA"  # IDA
elif method_indx == 15:
    ida_ssy.load(dyn_general); prf = "IDA_SSY"  # IDA + SSY
elif method_indx == 16:
    dish.load(dyn_general); prf = "DISH_EDC"  # DISH with default EDC parameters 
elif method_indx == 17:
    dish.load(dyn_general)  # DISH with IPSD decoherence rates
    dyn_general.update({"decoherence_times_type":3}); prf = "DISH_SCHWARTZ" # Schwartz 2 decoherence times
elif method_indx == 18:
    sdm.load(dyn_general)  # SDM with IPSD decoherence rates
    dyn_general.update({"decoherence_times_type":3}); prf = "SDM_SCHWARTZ" # Schwartz 2 decoherence times
elif method_indx == 19:
    sdm_ssy.load(dyn_general)  # SDM with IPSD decoherence rates + SSY
    dyn_general.update({"decoherence_times_type":3}); prf = "SDM_SCHWARTZ_SSY" # Schwartz 2 decoherence times

    
# ## 3. Choosing initial conditions: Nuclear and Electronic
##############
#icond_indx = 0
icond_indx = args.istate_traj
##############

#============== How nuclear DOFs are initialized =================
#icond_nucl = 0  # Coords and momenta are set exactly to the given value
#icond_nucl = 1  # Coords are set, momenta are sampled
#icond_nucl = 2  # Coords are sampled, momenta are set
icond_nucl = 3  # Both coords and momenta are sampled

m = 2000.0
k = 0.01
wq, wp = 1.0, 1.0

nucl_params = { "ndof":1, "q":[-4.0], "p":[0.0], 
                "mass":[2000.0], "force_constant":[0.01], 
                "q_width":[wq],
                "p_width":[wp],
                "init_type":icond_nucl }
if model_indx==4: 
    nucl_params = { "ndof":2, 
                    "q":[-4.0, 1.0], 
                    "p":[0.0, 0.0], 
                    "q_width":[wq, wq],
                    "p_width":[wp, wp],
                    "mass":[m,m], "force_constant":[k,k], 
                    "init_type":icond_nucl }

#============= How electronic DOFs are initialized ==================
# Select a specific initial condition
istate = icond_indx

istates = list(np.zeros(NSTATES))
istates[istate] = 1.0

elec_params = {"verbosity":-1, "init_dm_type":0, "ndia":NSTATES, "nadi":NSTATES, "rep":1, "init_type":1, "istate":istate, "istates":istates }


# ## 4. Running the calculations
dyn_params = dict(dyn_general)
dyn_params.update({ "prefix":F"DATA-model{model_indx}-method{method_indx}-icond{icond_indx}", 
                    "prefix2":F"DATA-model{model_indx}-method{method_indx}-icond{icond_indx}" })
print(F"Computing DATA-model{model_indx}-method{method_indx}-icond{icond_indx}")    

rnd = Random()
res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)


# ## 5. Plotting the results
pref = F"DATA-model{model_indx}-method{method_indx}-icond{icond_indx}"

plot_params = { "prefix":pref, "filename":"mem_data.hdf", "output_level":3,
                "which_trajectories":[0], "which_dofs":[0], "which_adi_states":list(range(NSTATES)), 
                "which_dia_states":list(range(NSTATES)), 
                "frameon":True, "linewidth":3, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":0,
                "what_to_plot":["coordinates", "momenta",  "forces", "energies", "phase_space", "se_pop_adi",
                                "se_pop_dia", "sh_pop_adi", "traj_resolved_adiabatic_ham", "traj_resolved_diabatic_ham", 
                                "time_overlaps" ] 
              }

tsh_dynamics_plot.plot_dynamics(plot_params)

