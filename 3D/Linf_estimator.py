# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes the pointwise a posteriori error estimator derived in Appendix A.1 and all required quantities.

# Make sure you generated all necessary meshes and numerical approximations beforehand (see generate_meshes.py and FVFEscheme.py).

# The quadrature points and weights, stored in triangle4.csv, are taken from

# Xiao, Hong and Gimbutas, Zydrunas. 
# A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions, 
# Computers & Mathematics with Applications 59(2), 663–676, 2010. 
# [DOI: 10.1016/j.camwa.2009.10.027]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import myfun as my
import numpy as np
import scipy as sp
import time
import pickle
import os


# Get current path
current_path = os.getcwd()

# Create folder if it doesn't exist
folder_name = "data"
folder_path = os.path.join(current_path, folder_name)
os.makedirs(folder_path, exist_ok=True)

# Configuration used to generate values in Table 2:
test = 'diff' #set test case
method = 'expl' 
spatial = [3,4,5,6,7]
temporal = [10,20,40,60,100]
TT = [0.00001,0.000016,0.000028,0.00004,0.00006]

tic = time.time()

for index in range(len(spatial)) :

    fineness = spatial[index] # number of refinements of mesh before calculating numerical solution
    Nt = temporal[index] # number of time steps, f5T0.02Nt400, f6T0.02Nt800, f7T0.03Nt2400
    maxiter = Nt

    print('fineness: ',fineness)
    print('Nt: ',Nt)

    pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    [K,F,K_dual,K_inter] = pickle.load(open(file_path,'rb')) # load mesh

    K_el = K.points[K.simplices]
    for i in range(K.num) :
        _, K.simplices[i] = my.orientation(K_el[i], K.simplices[i])
    K_el = K.points[K.simplices]

    toc = time.time()

    hht = []
    rrho = []
    vertex_val = []
    betaKE = []
    for n in range(maxiter) :

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [ht,aux_rho,rhs] = pickle.load(open(file_path,'rb'))
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [aux_q0,aux_beta] = pickle.load(open(file_path,'rb'))

        hht.append(ht)
        rrho.append(aux_rho)
        vertex_val.append(aux_q0)
        betaKE.append(aux_beta)

    vertex_val = np.asarray(vertex_val)
    betaKE = np.asarray(betaKE)

    elapsed = time.time() - toc
    print('numsol data loaded in ',"%.2f" % round(elapsed/60, 2), 'minutes.')

    data = np.loadtxt("tetrahedron8.csv", delimiter=",", skiprows=1)
    weights_tri = data[:,-1]
    xi_ref = np.column_stack((data[:,1],data[:,2],data[:,3])) # physical coordinates

    pt_primal, J_primal, v0_primal, grads_primal = my.tritrafo_quad_tet(K.points[K.simplices], xi_ref) # shape (K.num,len(xi),3)

    indices = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]

    L_primal = my.barycentric_coords(pt_primal, J_primal, v0_primal)
    bF = my.bubble_F(L_primal, indices)
    bK  = my.bubble_K(L_primal)
    gK  = my.grad_bubble_K(L_primal, grads_primal)
    gF = my.grad_bubble_F(L_primal, grads_primal, indices)

    [A,grads,M] = my.assemble_FE_matrix_q(K,K_el)

    qq_time = []
    grad_qq_time = []

    for n in range(maxiter) : 
        toc = time.time()

        # if n % 10 == 0:
        print('time step: ',n)

        [grad_q0,val] = my.get_grad_morley_val_primal(K,bK,gK,bF,gF,vertex_val[n],betaKE[n])
        aux_grad_morley = grad_q0 + val

        qq = []
        bb = []
        grad_qq = []
        q_old = np.zeros([len(K.pt_reduced),3])
        for d in range(3) : # go through components (2D)

            grad_morley = np.einsum('tij,i->tj',aux_grad_morley,weights_tri)[:,d]
            q,b = my.getq_FE(K,A,grad_morley,q_old[:,d])
            grad_q = my.get_gradq(K,grads,q)
            q_old[:,d] = q

            qq.append(q)
            bb.append(b)
            grad_qq.append(grad_q)


        data = []
        data.append(qq)
        data.append(bb)
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        pickle.dump(data,open(file_path,'wb')) # store data

        qq_time.append(np.transpose(qq))
        grad_qq_time.append(np.array(grad_qq)) # shape: d,Knum,2

        elapsed = time.time() - toc
        print('vector-valued FE computed in ',"%.2f" % round(elapsed/60, 2), 'minutes.')

    elapsed = time.time() - tic
    print('total FE sol computed in ',"%.2f" % round(elapsed/60, 2), 'minutes.')


    # -------------------------------- Linf estimator -------------------------------------

    cTr = 4.373213925301394 # L1

    cSZ01 = 1.1481812160876688
    cSZ11 = 8
    cSZ12 = 9.771236166739577
    cSZ02 = 1.3843005945553915

    C_ol = 12

    hmin = np.min(K.diam)
    C4h = (np.log(1/(np.sqrt(2)*hmin)) + hmin*sp.special.kv(1,hmin) - 1/np.sqrt(2)*sp.special.kv(1,1/np.sqrt(2))) + 11

    eta_inf_time = []
    for n in range(Nt):

        if n % 10 == 0:
            print('esti time step: ',n)

        eta_comp = 0
        for d in range(3):

            eta_K = []
            for i in range(K.num) : 

                hK = np.max(K.diam[i])
                C3h = 3*hK+16*(3*hK)**2 # compute C_3(3h)

                alpha = C_ol*cSZ02*C4h*hK**2 + cSZ01*C3h*hK
                beta = C_ol*cTr*(cSZ12+cSZ02)*C4h*hK + (cSZ11+cSZ01)*C3h

                gradq0 = 0
                for j in range(4) :
                    pt_i = K.simplices[i][j]
                    gradq0 += vertex_val[n][pt_i]*np.array([K.hat[i][j][0],K.hat[i][j][1],K.hat[i][j][2]],K.hat[i][j][3])

                # || q_h - f ||_Linf
                res = np.max(np.abs([qq_time[n][K.pt_ident[K.simplices[i][0]]][d]-gradq0[d],qq_time[n][K.pt_ident[K.simplices[i][1]]][d]-gradq0[d],qq_time[n][K.pt_ident[K.simplices[i][2]]][d]-gradq0[d]]))
                bubble_max = np.max(np.abs(betaKE[n][i]))/K.diam[i]*135/2 # TO DO
                neighs = K.neighbors[i]

                old = -1
                for k in range(3) :
                    j = neighs[k]
                    jump_q = np.max([old,np.abs(np.dot(grad_qq_time[n][d][i] - grad_qq_time[n][d][j],K.F.n[i][k]))])
                    old = jump_q
            
                eta_K.append(alpha*(bubble_max+res) + beta*jump_q)

            eta_comp += np.max(eta_K)**2

        eta_inf_time.append(np.sqrt(eta_comp))

    eta_inf_time = np.asarray(eta_inf_time)

    qmax = np.sqrt(np.max(np.asarray(qq_time),axis=1)[:,0]**2 + np.max(np.asarray(qq_time),axis=1)[:,1]**2 + np.max(np.asarray(qq_time),axis=1)[:,2]**2)

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(eta_inf_time+qmax,open(file_path,'wb')) # store data

    elapsed = time.time() - tic
    print('This took ',"%.2f" % round(elapsed/60, 2), 'minutes.')

