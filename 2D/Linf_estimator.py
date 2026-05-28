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


# Configuration used to generate values in Table 1:
test = 'diff' #set test case
method = 'expl'
spatial = [4,5,6,7]
temporal = [25,50,200,800]
TT = [0.0005,0.0005,0.001,0.001]


for index in range(len(spatial)) :

    fineness = spatial[index] # number of refinements of mesh before calculating numerical solution
    Nt = temporal[index] # number of time steps, f5T0.02Nt400, f6T0.02Nt800, f7T0.03Nt2400
    maxiter = Nt

    pickle_name = 'MESH_2D_UNITSQUARE_fineness'+str(fineness)+'.p'
    [K,F,K_dual,K_inter] = pickle.load(open('put_some_path_here'+pickle_name,'rb')) # load mesh

    for i in range(K.num) :
        _, K.simplices[i] = my.orientation(K.el[i], K.simplices[i])
    K.el = K.points[K.simplices]

    toc = time.time()

    hht = []
    rrho = []
    vertex_val = []
    betaKE = []
    for n in range(maxiter) :

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
        [ht,aux_rho,rhs] = pickle.load(open('put_some_path_here'+pickle_name,'rb'))
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
        [aux_q0,aux_beta] = pickle.load(open('put_some_path_here'+pickle_name,'rb'))

        hht.append(ht)
        rrho.append(aux_rho)
        vertex_val.append(aux_q0)
        betaKE.append(aux_beta)

    vertex_val = np.asarray(vertex_val)
    betaKE = np.asarray(betaKE)

    elapsed = time.time() - toc
    print('numsol data loaded in ',"%.2f" % round(elapsed/60, 2), 'minutes.')
    toc = time.time()

    data = np.loadtxt("2D/triangle4.csv", delimiter=",", skiprows=1)
    weights_tri = data[:,-1]
    xi_ref = np.column_stack((data[:,1],data[:,2])) # physical coordinates

    pt_primal, J_primal, v0_primal, grads_primal = my.tritrafo_quad_tri(K.points[K.simplices], xi_ref) # shape (K.num,len(xi),3)

    indices = [[1,2],[0,2],[0,1]]
    L_primal = my.barycentric_coords(pt_primal, J_primal, v0_primal)
    bF = my.bubble_E(L_primal, indices)
    bK  = my.bubble_K(L_primal)
    gK  = my.grad_bubble_K(L_primal, grads_primal)
    gF = my.grad_bubble_E(L_primal, grads_primal, indices)

    [A,grads,M] = my.assemble_FE_matrix_q(K)
    
    tic = time.time()
    toc = time.time()

    qq_time = []
    grad_qq_time = []

    for n in range(maxiter) : 

        if n % 10 == 0:
            print('FE time step: ',n)

        qq = []
        bb = []
        grad_qq = []
        for d in range(2) : # go through components (2D)
            print('component: ',d)

            aux_grad_morley = my.get_grad_morley_val_primal(K,bK,gK,bF,gF,vertex_val[n],betaKE[n])
            grad_morley = np.einsum('tij,i->tj',aux_grad_morley,weights_tri)[:,d]

            print('morley values done.')

            q,b = my.getq_FE(K,A,grad_morley)
            grad_q = my.get_gradq(K,grads,q)

            print('FE and grad done')

            qq.append(q)
            bb.append(b)
            grad_qq.append(grad_q)

        data = []
        data.append(qq)
        data.append(bb)
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty'+str(n)+'.p'
        pickle.dump(data,open('put_some_path_here'+pickle_name,'wb')) # store data

        qq_time.append(np.transpose(qq))
        grad_qq_time.append(np.array(grad_qq)) # shape: d,Knum,2

    elapsed = time.time() - tic


    # -------------------------------- Linf estimator -------------------------------------

    # set constants
    K_el = K.points[K.simplices]
    cUSR = 0
    for j in range(K.num) :
        hK = my.diam(K_el[j])
        a = np.linalg.norm(K.E.el[j][0]) 
        b = np.linalg.norm(K.E.el[j][1]) 
        c = np.linalg.norm(K.E.el[j][2]) 
        inradK = 1/2*np.sqrt(((b+c-a)*(c+a-b)*(a+b-c))/(a+b+c)) # 2D formula
        cUSR = np.max([cUSR,hK/inradK])

    cTr = cUSR # L1
    cBH = lambda k,j :cUSR**k
    def Cstab(k) :
        if k == 0:
            val = (5*np.sqrt(3))/12*cUSR
        elif k==1 :
            val = 5/4*cUSR
        return val

    cSZ = lambda k,j : cBH(k,j)*(1+2**(j-k)*Cstab(k))

    hmin = np.min(K.diam)
    C4h = 2*np.pi*(np.log(1/(np.sqrt(2)*hmin)) + hmin*sp.special.kv(1,hmin) - 1/np.sqrt(2)*sp.special.kv(1,1/np.sqrt(2))) + 66

    eta_inf_time = []
    for n in range(Nt):

        if n % 10 == 0:
            print('esti time step: ',n)

        eta_comp = 0
        for d in range(2):

            eta_K = []
            for i in range(K.num) : 

                hK = np.max(K.diam[i])
                C3h = 2*hK+16*(hK)**2

                alpha = 4*cSZ(0,2)*C4h*hK**2 + cSZ(0,1)*C3h*hK
                beta = 4*cTr*(cSZ(1,2)+cSZ(0,2))*C4h*hK + (cSZ(1,1)+cSZ(0,1))*C3h

                gradq0 = 0
                for j in range(3) :
                    pt_i = K.simplices[i][j]
                    gradq0 += vertex_val[n][pt_i]*np.array([K.hat[i][j][0],K.hat[i][j][1]])

                # || q_h - f ||_Linf
                res = np.max(np.abs([qq_time[n][K.pt_ident[K.simplices[i][0]]][d]-gradq0[d],qq_time[n][K.pt_ident[K.simplices[i][1]]][d]-gradq0[d],qq_time[n][K.pt_ident[K.simplices[i][2]]][d]-gradq0[d]]))
                bubble_max = np.max(np.abs(betaKE[n][i]))/K.diam[i]*135/2
                neighs = K.neighbors[i]

                old = -1
                for k in range(3) :
                    j = neighs[k]
                    jump_q = np.max([old,np.abs(np.dot(grad_qq_time[n][d][i] - grad_qq_time[n][d][j],K.E.n[i][k]))])
                    old = jump_q
            
                eta_K.append(alpha*(bubble_max+res) + beta*jump_q)

            eta_comp += np.max(eta_K)**2

        eta_inf_time.append(np.sqrt(eta_comp))

    eta_inf_time = np.asarray(eta_inf_time)

    qmax = np.sqrt(np.max(np.asarray(qq_time),axis=1)[:,0]**2 + np.max(np.asarray(qq_time),axis=1)[:,1]**2)
    print(np.sum(ht*(qmax)**2))
    int0T_a = ht*np.sum((np.asarray(eta_inf_time)+qmax)**2)

    print(int0T_a)

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty.p'
    pickle.dump(eta_inf_time+qmax,open('put_some_path_here'+pickle_name,'wb')) # store data

