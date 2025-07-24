import numpy as np
import pickle
import FHN.definitions as fhn
from joblib import Parallel, delayed

n=1000
simulations_params=[(0.35,0.1),  #(r,sigma)
             (0.33,0.1),
             (0.33,0.19),
             (0.33,0.23),
             (0.33,0.28),
             (0.32,0.04),
             (0.29,0.13),
             (0.266,0.202),
             (0.25,0.25)]

np.random.seed(0)
random_phases=np.random.uniform(0,2*np.pi,size=(n))
initial_state=np.empty(2*n)
initial_state[0::2]=2*np.cos(random_phases)
initial_state[1::2]=2*np.sin(random_phases)

simulations_list=Parallel(n_jobs=-1)(delayed(fhn.FHN_ring_solver)(n=n,r=params[0],sigma=params[1],initial_state=initial_state,dt=0.1) for params in simulations_params)


simulations_dict={}
for i in range(len(simulations_params)):
    simulations_dict[f"array_{i}.npy"]=simulations_list[i]

np.savez_compressed("runFHN.npz",**simulations_dict)
