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


# Configuration used to generate values in Table 1:
test = 'diff' #set test case
method = 'expl' 
spatial = [4,5,6,7]
temporal = [25,50,200,800]
TT = [0.0005,0.0005,0.001,0.001]

times_gengron = []
times_loc = []
for stability in ['genGronwall','loc-in-time']:

    print(stability)
    for index in range(len(spatial)) :
        
        fineness = spatial[index] # number of refinements of mesh before calculating numerical solution
        Nt = temporal[index] # number of time steps
        maxiter = Nt-2 

        print('fineness: ',fineness)
        print('Nt: ',Nt)

        # DIFFUSION DOMINATED REGIME
        def rho0(x) :
            val = np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1])+1 
            return val


        pickle_name = 'MESH_2D_UNITSQUARE_fineness'+str(fineness)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [K,F,K_dual,K_inter] = pickle.load(open(file_path,'rb')) # load mesh

        K_el = K.points[K.simplices]

        # set upper bounds for required constants
        cP = 1/(2*np.pi)**2 # Poincare-Wirtinger constant on the flat torus
        C_ell = 1
        C_S = 2.1357917 # see [43]

        cUSR = 0
        for j in range(K.num) :
            hK = my.diam(K_el[j])
            a = np.linalg.norm(K.E.el[j][0]) 
            b = np.linalg.norm(K.E.el[j][1]) 
            c = np.linalg.norm(K.E.el[j][2]) 
            inradK = 1/2*np.sqrt(((b+c-a)*(c+a-b)*(a+b-c))/(a+b+c)) # 2D formula
            cUSR = np.max([cUSR,hK/inradK])
        Csz = np.sqrt(cUSR/2)+1

        B1 = 8/5*C_S**3*C_ell**2
        B2 = 864/125*C_S**6*C_ell**4

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_initL2.p'
        file_path = os.path.join(folder_path, pickle_name)
        initialL2 = pickle.load(open(file_path,'rb')) # load data

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

        C1 = 42*K.num
        C2 = 15*(K.num + K_dual.num)
        C3 = 0 # see below, depends on num. sol.
        C4 = 21
        C5 = 33*K_dual.num
        C6 = 91*K_dual.num
        C7 = 2*K.num + 3543
        C8 = 0 # see below, depends on time step
        C9 = (936*K.num + 10)
        C10 = 21
        C11 = 11*K.num
        C12 = 91*K.num

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

        for i in range(K.num) :
            _, K.simplices[i] = my.orientation(K.el[i], K.simplices[i])
        K.el = K.points[K.simplices]

        [A_FEq, grads, M_FEq] = my.assemble_FE_matrix_q(K)
        eigvals, eigvecs = sp.sparse.linalg.eigsh(M_FEq, k=1, which='SM')
        smallestEV_Mq = eigvals[0]
        eigvals, eigvecs = sp.sparse.linalg.eigsh(M_FEq, k=1, which='LM')
        largestEV_Mq = eigvals[0]
        eigvals, eigvecs = sp.sparse.linalg.eigsh(A_FEq, k=1, which='LM')
        largestEV_FEq = eigvals[0]

        int0T_thetaR = 0
        int0T_theta = 0
        morleyL2L3 = 0
        Psi = initialL2**2
        delta = 1.01

        tm = 0

        for n in range(maxiter) : # time steps

            C8 = (936*K.num + 4)*(n+1)+6
            
            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [ht,rhoFV,rhs] = pickle.load(open(file_path,'rb')) # load data
            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_c at time step'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [cc,aux_v,b] = pickle.load(open(file_path,'rb')) # load data
            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [vertex_val,beta] = pickle.load(open(file_path,'rb')) # load data
            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'qhinfty'+str(n)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [qq,bb] = pickle.load(open(file_path,'rb')) # store data

            FKED = my.getinterpolationRHS(K,rhoFV) 

            hat_mat = np.array(K.hat)[:, :, :2]      # Hat matrices: (N_tri, 2, 3)
            vv = vertex_val[K.simplices]     # Vertex values: (N_tri, 3)
            grad_q0 = np.einsum('tij,ti->tj', hat_mat, vv)  # grad_q0 = hat_mat @ vv -> (N_tri, 2)
            np.max(np.linalg.norm(grad_q0,axis=1))
            maxFq0 = np.max(K.E.area)*np.max(np.linalg.norm(grad_q0,axis=1))

            C3a = 9*K.num*np.max(FKED)
            C3b = 6*K.num*np.max(maxFq0)
            C3 = C3a + C3b

            thetaFE = Csz*np.sqrt(((1-2*C4*eps)/smallestEV_M)/(1-(2+6*largestEV_M/smallestEV_M)*C4*eps))*((C5*eps)/(1-2*C5*eps)*np.linalg.norm(b) + np.linalg.norm(b - A_FE @ cc) + ((C6*eps)/(1-2*C6*eps))*largestEV_FE*np.linalg.norm(cc))
            thetaFV = 1/ht*np.max(K.area)**(1/2)*((C1*eps)/(1-2*C1*eps)*np.linalg.norm(rhs)+np.linalg.norm(rhs - FV_matrix @ rhoFV)+((C2*eps)/(1-2*C2*eps))*largestEV_FV*np.linalg.norm(rhoFV))
            algebraic = 12*C3*C_S*eps*(np.sum(1/K.area))**(1/6) + thetaFE + thetaFV 
            thetaFEq = Csz/(2*np.sqrt(np.pi))*np.sqrt(((1-2*C10*eps)/smallestEV_M)/(1-(2+6*largestEV_Mq/smallestEV_Mq)*C10*eps))*((C11*eps)/(1-2*C11*eps)*(np.linalg.norm(bb[0])+np.linalg.norm(bb[1])) + np.linalg.norm(bb[0] - A_FEq @ qq[0]) + np.linalg.norm(bb[1] - A_FEq @ qq[1]) + ((C12*eps)/(1-2*C12*eps))*largestEV_FEq*(np.linalg.norm(qq[0])+np.linalg.norm(qq[1])))

            if stability == 'genGronwall' :

                int0T_theta += ht/2*(theta_R[n]+theta_R[n+1] + 2*algebraic)*(1 + C7*eps/(1 - C7*eps))
                morleyL2L3 += ht/2*(morleyL3[n]**2 + morleyL3[n+1]**2)

                int0T_thetaR += ht/2*(qhinfty[n] + qhinfty[n+1] + 2*thetaFEq)

                tm += ht
                
                A = initialL2**2 + 12*int0T_theta
                print('A',A)

                a = (4*C_S**2*C_ell**2*morleyL2L3 + 4*int0T_thetaR + tm/8)*(1 + C8*eps/(1 - C8*eps))

                E = np.exp(a)
                print('E ',E)

                delta = 8/5 

                print('value cond ', (B1*delta*A*E + B2*(delta*A*E)**2)/((delta-1)/((delta*tm*E))))
                print('condition ', B1*delta*A*E + B2*(delta*A*E)**2 < (delta-1)/((delta*tm*E)))

                if not (B1*delta*A*E + B2*(delta*A*E)**2 < (delta-1)/((delta*tm*E))) :
                    print('FINAL TIME ',tm-ht)
                    times_gengron.append(tm-ht)
                    print('TIME STEP ',n-1)
                    break

            elif stability == 'loc-in-time' :

                tm += ht
                print('time ',tm)

                inttn_theta = ht/2*(theta_R[n]+theta_R[n+1]+ 2*algebraic)*(1 + C7*eps/(1 - C7*eps))
                morleyL2L3 = ht/2*(morleyL3[n]**2 + morleyL3[n+1]**2)
                int0T_thetaR = ht/2*(qhinfty[n] + qhinfty[n+1] + 2*thetaFEq)

                A = Psi + 12*inttn_theta
                print('A ', A)

                a = (4*C_S**2*C_ell**2*morleyL2L3 + 4*int0T_thetaR + tm/8)*(1 + C9*eps/(1 - C9*eps))
            
                E = np.exp(a)
                print('E ',E)

                func = lambda x : ht*(B1*x*A*E + B2*(x*A*E)**2) - np.log(x) # =!= 0
                dx_func = lambda x : ht*(B1*A*E + 2*x*B2*(A*E)**2) - 1/x
                
                delta_old = delta
                delta = my.newton_method1D(func,dx_func, x0 = delta_old ,tol = 1.5*10**-8, maxiter = 20) # tol as in fsolve
                print('delta ',delta)

                print(delta > 1)
                if delta <= 1 :
                    print('FINAL TIME ',tm-ht)
                    times_loc.append(tm-ht)
                    print('TIME STEP ',n-1)
                    break
                else : 
                    Psi = delta*A*E

print(spatial)
print(temporal)
print('Generalized Gronwall: ',times_gengron)
print('Local-in-time continuation: ',times_loc)
