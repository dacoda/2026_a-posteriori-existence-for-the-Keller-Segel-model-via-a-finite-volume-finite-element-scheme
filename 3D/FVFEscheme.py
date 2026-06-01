# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes the numerical approximation to the weak solution of the (parbolic-elliptic) Keller-Segel system using Algorithm 4.3.
# Make sure you generated all necessary meshes beforehand (see generate_meshes.py).

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import myfun as my
import numpy as np
import time
import pickle
import os


# Get current path
current_path = os.getcwd()

# Create folder if it doesn't exist
folder_name = "data"
folder_path = os.path.join(current_path, folder_name)
os.makedirs(folder_path, exist_ok=True)


tic = time.time()

# SETTINGS FOR SCHEME
method = 'expl' # expl

# Configuration used to generate values in Table 2:
test = 'manuf'
method = 'expl'
spatial = [2,3,4,5,6]
temporal = [32,64,128,256,512]

for index in range(len(spatial)) :

    fineness = spatial[index] # number of refinements of mesh before calculating numerical solution
    Nt = temporal[index] # number of time steps
    maxiter = Nt

    pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    [K,F,K_dual,K_inter] = pickle.load(open(file_path,'rb')) # load mesh

    if test == 'manuf':
    # MANUFACTURED SOLUTION WITH T = 1
        pi = np.pi
        def rho0(x) :
            val = np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2])+1
            return val

        def exact_rho(x,t) :
            val = 1/(1+t)*np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2])+1
            return val
        
        def exact_c(x) :
            val = np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2])+1
            return val
        
        def delt_exact_rho(x,t) : 
            val = -1/(1+t)**2*np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2])
            return val
        
        def grad_exact_rho(x,t) :
            val = -pi/(1+t)*np.array([2*np.sin(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2]),2*np.cos(2*pi*x[0])*np.sin(2*pi*x[1])*np.cos(2*pi*x[2]),2*np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.sin(2*pi*x[2])])
            return val
        
        def grad_exact_c(x) :
            val = -pi*np.array([2*np.sin(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2]),2*np.cos(2*pi*x[0])*np.sin(2*pi*x[1])*np.cos(2*pi*x[2]),2*np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.sin(2*pi*x[2])])
            return val
        
        def Laplacian_exact_rho(x,t) :
            val = -12*pi**2/(1+t)*np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2])
            return val
        
        def Laplacian_exact_c(x) :
            val = -12*pi**2*np.cos(2*pi*x[0])*np.cos(2*pi*x[1])*np.cos(2*pi*x[2])
            return val

        def f(x,t) :
            val = delt_exact_rho(x,t) + np.dot(grad_exact_c(x),grad_exact_rho(x,t)) + exact_rho(x,t)*Laplacian_exact_c(x) - Laplacian_exact_rho(x,t)
            return val
        
        def g(x,t) :
            val = exact_c(x) - Laplacian_exact_c(x) - exact_rho(x,t)
            return val
        
        T = 1 # time interval [0,T]

    else :
        print('Warning: Wrong string input for test.')
 
    # SCHEME/PROBLEM PARAMETERS
    eps = 1 # change if you are interested in different diffusion coefficients
    ht = (T)/(Nt)

    print('-------------------------------  FV-FE Algorithm  ---------------------------------')
    print('test '+test)
    print('fineness ',fineness)
    print('number primal elements ', K.num)
    print('number primal points ', len(K.points))
    print('number dual elements ', K_dual.num)
    print('number dual points ', len(K_dual.points))

    toc = time.time()

    # print some important information regarding the discretization
    print('Nt',Nt)
    print('hx',np.max(K.diam))
    print('ht',ht)

    # pre-allocate variables
    rho = np.zeros([maxiter+1,K.num])
    cc = np.zeros([maxiter+1,len(K_dual.pt_dual_reduced)])

    toc = time.time()
    A = my.assemble_FE_matrix(K_dual)

    data = [K,F,K_dual,K_inter]
    pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(data,open(file_path,'wb')) # store data

    elapsed = time.time() - toc
    print('FE matrix assembled in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 
    toc = time.time()

    FV_matrix = my.assemble_FV_matrix(ht,K,eps)

    elapsed = time.time() - toc
    print('FV matrix assembled in ',"%.2f" % round(elapsed/60, 2), 'minutes.') 

    # get discretization of rho0 
    for i in range(K.num) :
        rho[0][i] = rho0(K.CC[i])

    # SAVE RHO
    data = []
    data.append(ht) # time step size Delta t^n := t^{n+1} - t^n
    data.append(rho[0][:])
    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(0)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(data,open(file_path,'wb')) # store data

    # GET COEFFICIENTS FOR MORLEY/LINEAR INTERPOLATION AT TIME "n = -1":
    FKED = my.getinterpolationRHS(K,rho[0][:]) 
    vertex_val = my.getq0(K,rho[0][:]) # linear interpolation
    [betaKF,numerKF,denomKF] = my.getbetaE(K,vertex_val,FKED) # Jacobi_cdotn is first derivative of componentwise linear interpolation of grad c

    elapsed = time.time() - toc
    print('Morley reconstruction computed in ', "%.2f" % round(elapsed/60, 2), 'minutes.')
    toc = time.time()

    # SAVE MORLEY RECONSTRUCTION OF RHO 
    data = []
    data.append(vertex_val)
    data.append(betaKF)
    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(0)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(data,open(file_path,'wb')) # store data

    # calculate c0 with rho0 as right-hand side 
    c0 = my.getc_FE(rho[0][:],lambda x : g(x,0),K_dual,A) #,M) 
    cc[0][:] = c0
    elapsed = time.time() - toc
    print('initial chemical concentration computed via FE scheme in ', "%.2f" % round(elapsed/60, 2), 'minutes.')
    toc = time.time()

    ht = T/Nt

    for n in range(maxiter) : # range(Nt) :

        print('time step :',n)

        toc = time.time()

        # SAVE DATA CHEMICAL CONCENTRATION
        vv = my.get_gradc(K_dual,cc[n][:]) # set v = gradient_c
        data = []
        data.append(cc[n][:])
        data.append(vv)

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_c at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        pickle.dump(data,open(file_path,'wb')) # store data

        if np.min(rho[n][:]) < 0 :
            print('Error: Negative densities detected.')
            break

        toc = time.time()

        rho[n+1][:] = my.finitevolumescheme_rho_expl(rho[n][:],ht,K,vv,lambda x : f(x,(n+1)*ht),FV_matrix)
        
        elapsed = time.time() - toc
        print('Bacterial denisty computed via FV scheme in ', "%.2f" % round(elapsed/60, 2), 'minutes.')
        toc = time.time()

        # SAVE BACTERIAL DENSITY RHO 
        data = []
        data.append(ht) # time step size Delta t^n := t^{n+1} - t^n
        data.append(rho[n+1][:])
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n+1)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        pickle.dump(data,open(file_path,'wb')) # store data

        # GET COEFFICIENTS FOR MORLEY/LINEAR INTERPOLATION :
        FKED = my.getinterpolationRHS(K,rho[n+1][:]) 
        vertex_val = my.getq0(K,rho[n+1][:]) # linear interpolation
        [betaKF,numerKF,denomKF] = my.getbetaE(K,vertex_val,FKED) # Jacobi_cdotn is first derivative of componentwise linear interpolation of grad c
        elapsed = time.time() - toc
        print('Morley reconstruction computed in ', "%.2f" % round(elapsed/60, 2), 'minutes.')
        toc = time.time()

        # SAVE MORLEY RECONSTRUCTION OF RHO 
        data = []
        data.append(vertex_val)
        data.append(betaKF)
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n+1)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        pickle.dump(data,open(file_path,'wb')) # store data
        
        toc = time.time()
        # GET CHEMICAL DENISTY
        cc[n+1][:] = my.getc_FE(rho[n+1][:],lambda x : g(x,(n+1)*ht),K_dual,A) # ,M) 
        elapsed = time.time() - toc
        print('Chemical concentration approximated via FE scheme in ', "%.2f" % round(elapsed/60, 2), 'minutes.')
        
        if n == maxiter-1 :
            print('final c dumped.')
            # SAVE DATA CHEMICAL CONCENTRATION
            vv = my.get_gradc(K_dual,cc[n+1][:])
            data = []
            data.append(cc[n+1][:])
            data.append(vv)
            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_c at time step'+str(n+1)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            pickle.dump(data,open(file_path,'wb')) # store data

    progress = 100
    print("%.2f" % progress, 'procent of progress made.')
    elapsed = time.time() - tic
    print('This took',"%.2f" % round(elapsed/60, 2), 'minutes.')
