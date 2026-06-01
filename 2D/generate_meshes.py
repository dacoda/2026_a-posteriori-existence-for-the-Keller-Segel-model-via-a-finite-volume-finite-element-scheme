# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes the primal mesh and the dual mesh for the unit square [0,1)^2 usable in Algorithm 4.3.
# Further, this script computes the intersected mesh used to compute the a posteriori error estimator.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import myfun as my
import numpy as np
import time
from matplotlib import pyplot as plt
import pickle
import os



# Get current path
current_path = '/local/scratch/hoffmann' # os.getcwd()

# Create folder if it doesn't exist
folder_name = "data"
folder_path = os.path.join(current_path, folder_name)
os.makedirs(folder_path, exist_ok=True)


tic = time.time()
toc = time.time()

for index in range(1,8) :

    fineness = index # number of refinements of mesh before calculating numerical solution
    print('------------------- fineness '+str(fineness)+' --------------------')

    # GET PRIMAL MESH for the finite volume scheme to approximate the bacterial density:
    K = my.primal_mesh(fineness)
    K.init_edges()
    E = my.edges(K)

    elapsed = time.time() - toc
    print('Primal mesh initialized in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()


    #GET DUAL MESH for FE method to approximate chemical density
    K_dual = my.dual_mesh(K,E)

    elapsed = time.time() - toc
    print('Dual mesh initialized in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()


    #GET INTERSECTED MESH to compute a posteriori error estimator
    K_inter = my.intersect_mesh(K,E)
    my.post_process(K_inter,K)
    my.post_process_TinK(K_inter,K)
    elapsed = time.time() - toc
    print('Intersected mesh initialized in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()

    elapsed = time.time() - tic
    print('Total mesh generation time was ',"%.2f" % round(elapsed/60, 2), 'minutes.') 


    data = [K,E,K_dual,K_inter]
    pickle_name = 'MESH_2D_UNITSQUARE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(data,open(file_path,'wb')) # store data

    # # PLOT THE PRIMAL MESH
    # plt.triplot(np.array(K.points)[:,0], np.array(K.points)[:,1], K.tri.simplices,color='steelblue')
    # plt.show()

    # # PLOT THE DUAL MESH
    # plt.triplot(np.array(K_dual.points)[:,0], np.array(K_dual.points)[:,1], K_dual.simplices,color='green')
    # plt.show()

    # # PLOT THE INTERSECTED MESH 
    # plt.triplot(K_inter.points[:,0], K_inter.points[:,1], K_inter.simplices,color='orange')
    # plt.show()