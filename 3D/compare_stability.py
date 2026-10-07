# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script checks the availability of the stabililty framework based on the Genralized Gronwall Lemma, see Section 3.1 
# and based on a local-in-time continuation argument, see Section 3.2.

# Make sure you computed the a posteriori residual estimator and the pointwise a posteriori error estimator beforehand (see residual_estimates.py and Linf_estimator.py).

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import scipy as sp
import myfun as my
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

times_gengron = []
AA_gengron = []
EE_gengron = []
delta_gengron = []
times_loc = []
AA_loc = []
EE_loc = []
delta_loc = []
for index in range(len(spatial)) :
    
    fineness = spatial[index] # number of refinements of mesh before calculating numerical solution
    Nt = temporal[index] # number of time steps
    maxiter = Nt -1

    print('fineness: ',fineness)
    print('Nt: ',Nt)

    # DIFFUSION DOMINATED REGIME
    def rho0(x) :
        val = np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1])*np.cos(2*np.pi*x[2])+1 
        return val


    pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    [K,F,K_dual,K_inter] = pickle.load(open(file_path,'rb')) # load mesh

    # set upper bounds for required constants
    cP = 1/np.pi # Payne-Weinberger-type constant for convex domains of the Poincare-Wirtinger inequality
    C_ell = 1
    C_S = np.sqrt(7/3)
    C_ol = 12

    K_el = K.points[K.simplices]
    cUSR = 0
    for i in range(K.num) :
        hK = my.diam(K_el[i])
        S = 0
        for j in range(4):
            S += K.F.area[i][j]
        inradK = 3*K.area[i]/S
        cUSR = np.max([cUSR,hK/inradK])

    Csz = 2*np.sqrt(2*cUSR)/5

    B1 = 2*C_S**3*C_ell**2
    B2 = 4*C_S**6*C_ell**4

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_initL2.p'
    file_path = os.path.join(folder_path, pickle_name)
    initialL2 = pickle.load(open(file_path,'rb')) # load data

    # load data
    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_theta.p'
    file_path = os.path.join(folder_path, pickle_name)
    theta_R = pickle.load(open(file_path,'rb')) # array in time, one value per time step

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_L3.p'
    file_path = os.path.join(folder_path, pickle_name)
    morleyL3 = pickle.load(open(file_path,'rb')) # array in time, one value per time step

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty.p'
    file_path = os.path.join(folder_path, pickle_name)
    qhinfty = pickle.load(open(file_path,'rb')) # array in time, one value per time step


    eps = np.finfo(float).eps # set machine epsilon

    C1 = 7*K.num + 8*K_dual.num + 21
    C2 = 36*K.num
    C3 = 0 # see below, depends on num. sol.
    C4 = 37
    C5 = 25*K_dual.num
    C6 = 20 + 214*K_dual.num
    C7 = (12*(214*12+5)+2140 +10)*K.num + F.num*77 + K_inter.num*3/2*(37*6+2) + 2*K_dual.num*3/2 + 20
    Bh = K.num*(3*214 + 3*(44 + 36+2*np.size(K.pt_reduced)+3) + 214) + 20
    C8 = 0 # see below, depends on time step
    C9 = Bh + 5
    C10 = 37
    C11 = 75*K.num
    C12 = 20 + 214*K.num
    
    n = 0
    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    [ht,rhoFV,rhs] = pickle.load(open(file_path,'rb')) # load data

    FV_matrix = my.assemble_FV_matrix(ht,K,1)
    eigvals, eigvecs = sp.sparse.linalg.eigsh(FV_matrix, k=1, which='LM')
    largestEV_FV = eigvals[0]

    [A_FE,M_FE] = my.assemble_FE_matrix(K_dual)
    eigvals, eigvecs = sp.sparse.linalg.eigsh(M_FE, k=1, which='SM')
    smallestEV_M = eigvals[0]
    eigvals, eigvecs = sp.sparse.linalg.eigsh(M_FE, k=1, which='LM')
    largestEV_M = eigvals[0]
    eigvals, eigvecs = sp.sparse.linalg.eigsh(A_FE, k=1, which='LM')
    largestEV_FE = eigvals[0]

    K_el = K.points[K.simplices]
    for i in range(K.num) :
        _, K.simplices[i] = my.orientation(K_el[i], K.simplices[i])
    K_el = K.points[K.simplices]

    [A_FEq, grads, M_FEq] = my.assemble_FE_matrix_q(K,K_el)
    eigvals, eigvecs = sp.sparse.linalg.eigsh(M_FEq, k=1, which='SM')
    smallestEV_Mq = eigvals[0]
    eigvals, eigvecs = sp.sparse.linalg.eigsh(M_FEq, k=1, which='LM')
    largestEV_Mq = eigvals[0]
    eigvals, eigvecs = sp.sparse.linalg.eigsh(A_FEq, k=1, which='LM')
    largestEV_FEq = eigvals[0]

    int0T_Linfsq = 0
    int0T_theta = 0
    morleyL2L3 = 0
    Psi = initialL2**2
    delta = 1.01
    A, E = 0, 0


    for stability in ['genGronwall','loc-in-time']:
        tm = 0

        print(stability)

        for n in range(maxiter) : # time steps

            print(n)

            C8 = (936*K.num + 4)*(n+1)+6 # TO DO
            
            # load data
            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [ht,rhoFV_0,rhs_0] = pickle.load(open(file_path,'rb')) # load data

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n+1)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [ht,rhoFV_p,rhs_p] = pickle.load(open(file_path,'rb')) # load data
            
            if n == 0 :
                pass
            else:
                cc_m = cc_0
                b_m = b_0

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_c at time step'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [cc_0,aux_v,b_0] = pickle.load(open(file_path,'rb')) # load data

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [vertex_val,beta] = pickle.load(open(file_path,'rb')) # load data

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [qq_0,bb_0] = pickle.load(open(file_path,'rb')) # store data

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty'+str(n+1)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [qq_p,bb_p] = pickle.load(open(file_path,'rb')) # store data

            FKED = my.getinterpolationRHS(K,rhoFV) 

            hat_mat = np.array(K.hat)[:, :, :3]      # Hat matrices
            vv = vertex_val[K.simplices]     # Vertex values
            grad_q0 = np.einsum('tij,ti->tj', hat_mat, vv)  # grad_q0 = hat_mat @ vv 
            np.max(np.linalg.norm(grad_q0,axis=1))
            maxFq0 = np.max(K.F.area)*np.max(np.linalg.norm(grad_q0,axis=1))

            # TO DO
            C3a = 12*K.num*np.max(FKED)
            C3b = 8*K.num*np.max(maxFq0)
            C3 = C3a + C3b

            res0 = theta_R[0][n][0]
            res1 = theta_R[0][n][1]
            dj_0 = theta_R[1][n]
            dj_p = theta_R[1][n+1]
            time1 = theta_R[2][n][0]
            time2 = theta_R[2][n][1]
            CellLinf = theta_R[3][n]
            cj_0 = theta_R[4][n]
            cj_p = theta_R[4][n+1]

            print('res0: ',res0)
            print('res1: ',res1)
            print('dj0: ',dj_0)
            print('cj0: ',cj_0)
            print('time1: ',time1)
            print('time2 :',time2)
            print('CellLinf: ',CellLinf)

            if n==0:
                pass
            else :
                pre_FE_m = pre_FE_0
                FE_m = FE_0
                thetaFE_m = thetaFE_0

            pre_FE_0 = theta_R[5][n][0]
            FE_0 = theta_R[5][n][1]

            ell0n_t1 = 1/2 - 1/(2*np.sqrt(3))
            ell0n_t2 = 1/2 + 1/(2*np.sqrt(3)) 
            ell1n_t1 = ell0n_t2
            ell1n_t2 = ell0n_t1

            # need thetaFEa for time steps n and n-1 and thetaFVa for n and n+1
            thetaFE_0 = Csz*np.sqrt(((1-2*C4*eps)/smallestEV_M)/(1-(2+6*largestEV_M/smallestEV_M)*C4*eps))**(1/2)*((C5*eps)/(1-2*C5*eps)*np.linalg.norm(b_0) + np.linalg.norm(b_0 - A_FE @ cc_0) + ((C6*eps)/(1-2*C6*eps))*largestEV_FE*np.linalg.norm(cc_0))

            thetaFV_0 = 1/ht*np.max(K.area)**(1/2)*((C1*eps)/(1-2*C1*eps)*np.linalg.norm(rhs_0)+np.linalg.norm(rhs_0 - FV_matrix @ rhoFV_0)+((C2*eps)/(1-2*C2*eps))*largestEV_FV*np.linalg.norm(rhoFV_0))
            thetaFV_p = 1/ht*np.max(K.area)**(1/2)*((C1*eps)/(1-2*C1*eps)*np.linalg.norm(rhs_p)+np.linalg.norm(rhs_p - FV_matrix @ rhoFV_p)+((C2*eps)/(1-2*C2*eps))*largestEV_FV*np.linalg.norm(rhoFV_p))

            algebraic = 12*C3*C_S*eps*(np.sum(1/K.area))**(1/6) 
            
            thetaFEq_0 = C_ol*Csz*1.002*np.sqrt(((1-2*C10*eps)/smallestEV_M)/(1-(2+6*largestEV_Mq/smallestEV_Mq)*C10*eps))*((C11*eps)/(1-2*C11*eps)*(np.linalg.norm(bb_0[0])+np.linalg.norm(bb_0[1])) + np.linalg.norm(bb_0[0] - A_FEq @ qq_0[0]) + np.linalg.norm(bb_0[1] - A_FEq @ qq_0[1]) + ((C12*eps)/(1-2*C12*eps))*largestEV_FEq*(np.linalg.norm(qq_0[0])+np.linalg.norm(qq_0[1])))
            thetaFEq_p = C_ol*Csz*1.002*np.sqrt(((1-2*C10*eps)/smallestEV_M)/(1-(2+6*largestEV_Mq/smallestEV_Mq)*C10*eps))*((C11*eps)/(1-2*C11*eps)*(np.linalg.norm(bb_p[0])+np.linalg.norm(bb_p[1])) + np.linalg.norm(bb_p[0] - A_FEq @ qq_p[0]) + np.linalg.norm(bb_p[1] - A_FEq @ qq_p[1]) + ((C12*eps)/(1-2*C12*eps))*largestEV_FEq*(np.linalg.norm(qq_p[0])+np.linalg.norm(qq_p[1])))

            if n == 0 :
                theta_n1 = res0 + dj_0 + time1 + CellLinf + cj_0 + ell0n_t1*(pre_FE_0*(FE_0 + thetaFE_0) + res1 + thetaFV_p) + algebraic + ell1n_t1*thetaFV_0 
                theta_n2 = res0 + dj_0 + time1 + CellLinf + cj_0 + ell0n_t2*(pre_FE_0*(FE_0 + thetaFE_0) + res1 +thetaFV_p)+ algebraic + ell1n_t2*thetaFV_0 
            else : 
                theta_n1 = ell0n_t1*(res1+dj_p + cj_p + pre_FE_0*(FE_0+thetaFE_0) +thetaFV_p) + ell1n_t1*(res0 +dj_0 + cj_0 + pre_FE_m*(FE_m+thetaFE_m) + thetaFV_0 + time2) + time1 + CellLinf + algebraic
                theta_n2 = ell0n_t2*(res1+dj_p + cj_p + pre_FE_0*(FE_0+thetaFE_0) +thetaFV_p) + ell1n_t2*(res0 +dj_0 + cj_0 + pre_FE_m*(FE_m+thetaFE_m) + thetaFV_0 + time2) + time1 + CellLinf + algebraic

            if stability == 'genGronwall' :

                int0T_theta += ht/2*(theta_n1**2+theta_n2**2)*(1 + C7*eps/(1 - C7*eps))
                morleyL2L3 += ht/2 * ((ell0n_t1*(morleyL3[n+1]) + ell1n_t1*(morleyL3[n]))**2 + (ell0n_t2*(morleyL3[n+1]) + ell1n_t2*(morleyL3[n]))**2)
                
                int0T_Linfsq += ht/2 * ((ell0n_t1*(qhinfty[n+1] + thetaFEq_p) + ell1n_t1*(qhinfty[n] + thetaFEq_0))**2
                                        + (ell0n_t2*(qhinfty[n+1] + thetaFEq_p) + ell1n_t2*(qhinfty[n] + thetaFEq_0))**2)

                tm += ht

                A_old = A
                A = initialL2**2 + 12*int0T_theta
                print('A',A)

                a = (4*C_S**2*C_ell**2*morleyL2L3**(2/3) + 4*int0T_Linfsq + tm/8)*(1 + C8*eps/(1 - C8*eps))

                E_old = E
                E = np.exp(a)
                print('E ',E)

                func = lambda x : B1*x*A*E + B2*(x*A*E)**2 - (x-1)/(x*tm*E) # =!= 0
                dx_func = lambda x : B1*A*E + 2*x*B2*(A*E)**2 - 1/(tm*E*x**2)
                
                delta_old = delta

                delta_boundary, delta = my.first_admissible_delta(func, dx_func)

                print('condition: ', func(delta)<0)
                print('delta ',delta)
                print('time ',tm)

                if delta <= 1 :
                    print('delta smaller one.')
                    print('FINAL TIME ',tm-ht)
                    times_gengron.append(tm-ht)
                    AA_gengron.append(A_old)
                    EE_gengron.append(E_old)
                    delta_gengron.append(delta_old)
                    print('TIME STEP ',n-1)
                    break
                elif func(delta)>=0 :
                    print('condition not satisfied.')
                    print('FINAL TIME ',tm-ht)
                    times_gengron.append(tm-ht)
                    AA_gengron.append(A_old)
                    EE_gengron.append(E_old)
                    delta_gengron.append(delta_old)
                    print('TIME STEP ',n-1)
                    break

            elif stability == 'loc-in-time' :

                tm += ht
                print('time ',tm)

                inttn_theta = ht/2*(theta_n1**2+theta_n2**2)*(1 + C7*eps/(1 - C7*eps))
                morleyL2L3 = ht/2 * ((ell0n_t1*(morleyL3[n+1]) + ell1n_t1*(morleyL3[n]))**2 + (ell0n_t2*(morleyL3[n+1]) + ell1n_t2*(morleyL3[n]))**2)
                # inttn_Linfsq = ht/2*((ell0n_t1*(qhinfty[n+1] + thetaFEq_p))**2 + (ell1n_t1*(qhinfty[n] + thetaFEq_0))**2)
                inttn_Linfsq = ht/2 * ((ell0n_t1*(qhinfty[n+1] + thetaFEq_p) + ell1n_t1*(qhinfty[n] + thetaFEq_0))**2 
                                        + (ell0n_t2*(qhinfty[n+1] + thetaFEq_p) + ell1n_t2*(qhinfty[n] + thetaFEq_0))**2)

                A_old = A
                A = Psi + 12*inttn_theta
                print('A ', A)

                a = (4*C_S**2*C_ell**2*morleyL2L3**(2/3) + 4*inttn_Linfsq + ht/8)*(1 + C9*eps/(1 - C9*eps))

                E_old = E
                E = np.exp(a)
                print('E ',E)

                func = lambda x : ht*(B1*x*A*E + B2*(x*A*E)**2) - np.log(x) # =!= 0
                dx_func = lambda x : ht*(B1*A*E + 2*x*B2*(A*E)**2) - 1/x
                
                delta_old = delta

                delta_boundary, delta = my.first_admissible_delta(func, dx_func)

                print('condition: ', func(delta)<0)
                print('delta ',delta)

                
                if delta <= 1 :
                    print('delta smaller one.')
                    print('FINAL TIME ',tm-ht)
                    times_loc.append(tm-ht)
                    AA_loc.append(A_old)
                    EE_loc.append(E_old)
                    delta_loc.append(delta_old)
                    print('TIME STEP ',n-1)
                    break
                elif func(delta)>=0 :
                    print('condition not satisfied.')
                    print('FINAL TIME ',tm-ht)
                    times_loc.append(tm-ht)
                    AA_loc.append(A_old)
                    EE_loc.append(E_old)
                    delta_loc.append(delta_old)
                    print('TIME STEP ',n-1)
                    break
                else : 
                    Psi = delta*A*E

        print('Generalized Gronwall: ',times_gengron)
        print('Local-in-time continuation: ',times_loc)

print(spatial)
print(temporal)
print('Generalized Gronwall: ',times_gengron)
print('AA: ',AA_gengron)
print('EE: ',EE_gengron)
print('delta: ',delta_gengron)
print('Local-in-time continuation: ',times_loc)
print('AA: ',AA_loc)
print('EE: ',EE_loc)
print('delta: ',delta_loc)