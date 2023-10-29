#!/usr/bin/env python
# coding: utf-8

import os, glob, time, h5py, warnings, sys
import multiprocessing as mp
import matplotlib.pyplot as plt   # plots
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit

from liblibra_core import *
import util.libutil as comn

import libra_py
from libra_py import units, data_conv #, dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
#import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
#import libra_py.data_savers as data_savers
import libra_py.workflows.nbra.decoherence_times as decoherence_times
import libra_py.data_visualize

from recipes import dish_nbra, fssh_nbra, fssh_nbra_tc, fssh2_nbra, gfsh_nbra, ida_nbra, mash_nbra, msdm_nbra

#from matplotlib.mlab import griddata
#%matplotlib inline 
warnings.filterwarnings('ignore')




###########################################
istep = 0    # the first timestep to read
fstep = 15000 # the last timestep to read
itraj = 0
istate_traj = 2

hdf_filename = F"DATA-model4-method4-icond{istate_traj}/mem_data.hdf"
###########################################

nsteps = fstep - istep
NSTEPS = nsteps
print(F"Number of steps = {nsteps}")

nstates = 1
with h5py.File(F"{hdf_filename}", 'r') as f:    
    nadi = int(f["hvib_adi/data"].shape[2] )
    nstates = nadi
NSTATES = nstates                                                            
print(F"Number of states = {nstates}")

#================== Read energies =====================
E, St, NAC, Hvib, Ekin = [], [], [], [], []
with h5py.File(F"{hdf_filename}", 'r') as f:
    for timestep in range(istep,fstep):
        x = np.array( np.diag( f["hvib_adi/data"][timestep, itraj, :, :])  )
        y = np.array( f["hvib_adi/data"][timestep, itraj, :, :] )
        ekin = f["Ekin_ave/data"][timestep]
        Ekin.append(ekin)
        E.append( x.real )
        NAC.append( -y.imag)
        Hvib.append( y )
        St_mat = np.array( f["St/data"][timestep, itraj, :, :].real  )
        St.append( St_mat)
                  
E = np.array(E)
St = np.array(St)
NAC = np.array(NAC)
Hvib = np.array(Hvib)


# ### 3.2. Define the Hamiltonian computation function
# As mentioned above, this function only mimics computing the required properties, by simply grabbing needed matrices from the global variables created above

class abstr_class:
    pass

def compute_model(q, params, full_id):
    timestep = params["timestep"]
    nst = params["nstates"]
    obj = abstr_class()

    obj.ham_adi = data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) )
    obj.nac_adi = data_conv.nparray2CMATRIX( NAC[timestep, :, :] )
    obj.hvib_adi = data_conv.nparray2CMATRIX( Hvib[timestep, :, :] )
    obj.basis_transform = CMATRIX(nst,nst); obj.basis_transform.identity()  #basis_transform
    obj.time_overlap_adi = data_conv.nparray2CMATRIX( St[timestep, :, :] )
    obj.gs_kinetic_energy = Ekin[timestep]
    #print(Ekin[timestep])
    
    return obj


# ## 4. Precompute and visualize key properties
# ### 4.1. Dephasing times and rates

# ================= Computing the energy gaps and decoherence times ===============
# Prepare the energies vs time arrays
HAM_RE = []
for step in range(E.shape[0]):
    HAM_RE.append( data_conv.nparray2CMATRIX( np.diag(E[step, : ]) ) )

# Average decoherence times and rates
tau, rates = decoherence_times.decoherence_times_ave([HAM_RE], [0], NSTEPS, 0)

# Computes the energy gaps between all states for all steps
dE = decoherence_times.energy_gaps_ave([HAM_RE], [0], NSTEPS)

# Decoherence times in fs
deco_times = data_conv.MATRIX2nparray(tau) * units.au2fs

# Zero all the diagonal elements of the decoherence matrix
np.fill_diagonal(deco_times, 0)

# Saving the average decoherence times [fs]
np.savetxt('decoherence_times.txt',deco_times.real)

# Computing the average energy gaps
gaps = MATRIX(NSTATES, NSTATES)
for step in range(NSTEPS):
    gaps += dE[step]
gaps /= NSTEPS

rates.show_matrix("decoherence_rates.txt")
gaps.show_matrix("average_gaps.txt")


# Let's visualize the map of decoherence times:
plt.figure()
avg_deco = np.loadtxt('decoherence_times.txt')
nstates = avg_deco.shape[0]
plt.imshow(np.flipud(avg_deco), cmap='hot', extent=(0,nstates,0,nstates))#, vmin=0, vmax=100)
plt.xlabel('State index')
plt.ylabel('State index')
colorbar = plt.colorbar()
colorbar.ax.set_title('fs')
plt.clim(vmin=0, vmax=30)
plt.title(F'Decoherence times')
plt.tight_layout()
plt.savefig('Decoherence_times.png')
#plt.show()


# ### 4.2. Nonadiabatic couplings map
# Compute averaged NACs
nac = np.zeros((NSTATES,NSTATES))
for naci in NAC:
    nac += (np.abs(naci)*1000*27.211385)
nac /= len(NAC)
nstates = avg_deco.shape[0]

plt.figure()
plt.imshow(np.flipud(nac), cmap='hot', extent=(0,NSTATES,0,NSTATES))#, vmin=0, vmax=100)
plt.xlabel('State index')
plt.ylabel('State index')
colorbar = plt.colorbar()
colorbar.ax.set_title('meV')
plt.clim(vmin=0, vmax=40)
plt.title(F'Nonadiabatic couplings')
plt.tight_layout()
plt.savefig('Nonadiabatic_couplings.png', dpi=600)

# ## 5. Dynamics

#================== Model parameters ====================
model_params = { "timestep":0, "icond":0,  "model0":0, "nstates":NSTATES }

#=============== Some automatic variables, related to the settings above ===================
#############
NSTEPS = 15000
#############

dyn_general = { "nsteps":NSTEPS, "ntraj":250, "nstates":NSTATES, "dt":5.0,
                "decoherence_rates":rates, "ave_gaps":gaps,                
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),
                "mem_output_level":2,
                "properties_to_save":[ "timestep", "time","se_pop_adi", "sh_pop_adi", "Epot_ave"],
                "prefix":F"NBRA", "prefix2":F"NBRA", "isNBRA":0, "nfiles": nsteps - 1,
                "thermally_corrected_nbra":0, "total_energy":0.086, "tcnbra_nu_therm":0.01, "tcnbra_nhc_size":10,
                "tcnbra_do_nac_scaling":1
              }


prop1 = [ "timestep", "time","se_pop_adi", "sh_pop_adi", "Epot_ave", "tcnbra_ekin","tcnbra_thermostat_energy" ]


if istate_traj==0:
    dyn_general.update( { "total_energy":0.061 } )
elif istate_traj==2:
    dyn_general.update( { "total_energy":0.14 } )

#####################
params_option = 9
#####################

if params_option==0:
    # High freq, 1 NHC, with NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.01, "tcnbra_nhc_size":1,  "tcnbra_do_nac_scaling":1} )
elif params_option==1:
    # High freq, 1 NHC, no NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.01, "tcnbra_nhc_size":1,  "tcnbra_do_nac_scaling":0} )
elif params_option==2:
    # High freq, 10 NHC, with NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.01, "tcnbra_nhc_size":10,  "tcnbra_do_nac_scaling":1} )
elif params_option==3:
    # High freq, 10 NHC, no NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.01, "tcnbra_nhc_size":10,  "tcnbra_do_nac_scaling":0} )
elif params_option==4:
    # Low freq, 1 NHC, with NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.0001, "tcnbra_nhc_size":1,  "tcnbra_do_nac_scaling":1} )
elif params_option==5:
    # Low freq, 1 NHC, no NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.0001, "tcnbra_nhc_size":1,  "tcnbra_do_nac_scaling":0} )
elif params_option==6:
    # Low freq, 10 NHC, with NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.0001, "tcnbra_nhc_size":10,  "tcnbra_do_nac_scaling":1} )
elif params_option==7:
    # Low freq, 10 NHC, no NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.0001, "tcnbra_nhc_size":10,  "tcnbra_do_nac_scaling":0} )
elif params_option==8:
    # Very low freq, 1 NHC, with NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.00000001, "tcnbra_nhc_size":10,  "tcnbra_do_nac_scaling":1} )
elif params_option==9:
    # Very low freq, 1 NHC, no NAC scaling
    dyn_general.update( { "tcnbra_nu_therm":0.00000001, "tcnbra_nhc_size":10,  "tcnbra_do_nac_scaling":0} )



##########################################################
#============== Select the method =====================
#dish_nbra.load(dyn_general); prf = "DISH"  # DISH
#fssh_nbra.load(dyn_general); prf = F"FSSH-{istate_traj}"  # FSSH
fssh_nbra_tc.load(dyn_general); prf = F"FSSH_TC-{istate_traj}-{params_option}"; dyn_general.update({ "properties_to_save":prop1})  # FSSH_TC
#fssh2_nbra.load(dyn_general); prf = "FSSH2"  # FSSH2
#gfsh_nbra.load(dyn_general); prf = "GFSH"  # GFSH
#ida_nbra.load(dyn_general); prf = "IDA"  # IDA
#mash_nbra.load(dyn_general); prf = "MASH"  # MASH
#msdm_nbra.load(dyn_general); prf = "MSDM"  # MSDM
##########################################################


#=================== Initial conditions =======================
#============== Nuclear DOF: these parameters don't matter much in the NBRA calculations ===============
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1 }

#============== Electronic DOF: Amplitudes are sampled ========
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}

###########
istate = 2
###########
elec_params.update( {"init_type":1,  "rep":1,  "istate":istate } )  # how to initialize: random phase, adiabatic representation

if prf=="MASH":
    istates = list(np.zeros(NSTATES))
    istates[istate] = 1.0
    elec_params.update( {"init_type":4,  "rep":1,  "istate":2, "istates":istates } )  # different initialization for MASH


# ### 5.2. Running the dynamics
def function1(icond):
    print('Running the calculations for icond:', icond)
    time.sleep( icond * 0.01 )
    rnd=Random()
    mdl = dict(model_params)
    #mdl.update({"icond": icond})  #create separate copy
    dyn_gen = dict(dyn_general)
    dyn_gen.update({"prefix":F"{prf}_icond_{icond}", "prefix2":F"{prf}_icond_{icond}", "icond": icond })
    res = tsh_dynamics.generic_recipe(dyn_gen, compute_model, mdl, elec_params, nucl_params, rnd)


# Here, we finally start the calculations themselves.
# We use 4 threads and 4 initial conditions (starting at files 1, 301, 601, and 901).

################################
nthreads = 4
ICONDS = list(range(1,1000,300))
#ICONDS = [1]
################################

pool = mp.Pool(nthreads)
pool.map(function1, ICONDS)
pool.close()                            
pool.join()


