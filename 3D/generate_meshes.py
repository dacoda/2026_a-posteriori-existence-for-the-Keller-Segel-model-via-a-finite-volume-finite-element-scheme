# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes the primal mesh and the dual mesh for the unit cube [0,1)^3 usable in Algorithm 4.3.
# Further, this script computes the intersected mesh used to compute the a posteriori error estimator.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import myfun as my
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
import os



# Get current path
current_path = os.getcwd()

# Create folder if it doesn't exist
folder_name = "data"
folder_path = os.path.join(current_path, folder_name)
os.makedirs(folder_path, exist_ok=True)



tic = time.time()
toc = time.time()

for index in range(2,7) :

    fineness = index # number of refinements of mesh before calculating numerical solution
    print('------------------- fineness '+str(fineness)+' --------------------')

    # GET PRIMAL MESH for the finite volume scheme to approximate the bacterial density
    K = my.primal_mesh(fineness)
    K.init_edges()
    F = my.faces(K)

    elapsed = time.time() - toc
    print('Primal mesh initialized in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()


    #GET DUAL MESH for FE method to approximate chemical density:
    K_dual = my.dual_mesh(K, F)

    elapsed = time.time() - toc
    print('Dual mesh initialized in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()


    # GET INTERSECTED MESH to compute a posteriori error estimator:
    K_inter = my.intersect_mesh(K,F)
    my.post_process(K_inter,K)
    my.post_process_TinK(K_inter,K)

    elapsed = time.time() - toc
    print('Intersected mesh initialized in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()


    if K.quality > 0 or K.F.quality > 0 or K_dual.quality > 0 or K_inter.quality > 0 : # quality check, always positive for meshes from [46]
        print('Mesh denied.')
    else : 
        data = [K,F,K_dual,K_inter]
        pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        pickle.dump(data,open(file_path,'wb')) # store data

