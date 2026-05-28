# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes the 'exact' L^2(H^1)-L^inf(L^2)-error, up to quadrature errors for integrating a nonlinear function numerically, 
# for the manufactured solution (29) and (30).
# Make sure you generated all necessary meshes and numerical approximations beforehand (see generate_meshes.py and FVFEscheme.py).

# The quadrature points and weights, stored in triangle10.csv, are taken from

# Xiao, Hong and Gimbutas, Zydrunas. 
# A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions, 
# Computers & Mathematics with Applications 59(2), 663–676, 2010. 
# [DOI: 10.1016/j.camwa.2009.10.027]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import myfun as my
import pickle


pi = np.pi
def exact_rho(x,y,z,t) :
    val = (1/(1+t))*np.cos(2*pi*x)*np.cos(2*pi*y)*np.cos(2*pi*z)+1
    return val

def grad_exact_rho(x,y,z,t) :
        val = -pi/(1+t)*np.array([2*np.sin(2*pi*x)*np.cos(2*pi*y)*np.cos(2*pi*z),2*np.cos(2*pi*x)*np.sin(2*pi*y)*np.cos(2*pi*z),2*np.cos(2*pi*x)*np.cos(2*pi*y)*np.sin(2*pi*z)])
        return val

T = 1 # time interval [0,T]

# Configuration used to generate values in Table 2:
test = 'manuf'
method = 'expl'
spatial = [2,3,4,5,6]
temporal = [32,64,128,256,512]

array_LinfL2 = []
array_L2H1 = []
hx = []
for index in range(len(spatial)) :

    fineness = spatial[index]
    Nt = temporal[index]

    pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
    [K,F,K_dual,K_inter] = pickle.load(open('put_some_path_here'+pickle_name,'rb')) # load mesh

    print('----- fineness '+str(fineness)+' -----')
    print('Number primal elements : ',K.num)

    data = np.loadtxt("3D/tetrahedron14.csv", delimiter=",", skiprows=1)
    weights_tet = data[:,-1]
    xi_ref = np.column_stack((data[:,1],data[:,2],data[:,3]))

    # transform points from unit tet to physical tet
    pt_primal, J_primal, v0_primal, grads_primal = my.tritrafo_quad_tet(K.points[K.simplices], xi_ref) # shape (K.num,len(xi),3)
    
    print('quadrature points initialized.')

    # -------------------------------- GET BUBBLE FUNCTION VALUES -------------------------------

    indices = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]

    # primal mesh
    L_primal = my.barycentric_coords(pt_primal, J_primal, v0_primal)

    bK_primal  = my.bubble_K(L_primal)
    bF_primal = my.bubble_F(L_primal, indices) # shape (K.num, four value per point on for each of the 4 face bubbles per K_primal)

    gK_primal  = my.grad_bubble_K(L_primal, grads_primal)

    print('bubble function values computed.')

    ht = T/Nt

    L2 = []
    H1 = []
    L2H1 = []
    
    vertex_val = 0
    betaKF = 0

    L2H1 = 0
    LinfL2 = 0
    for n in range(Nt) : 

        if n % 50 == 0 :
            print('time: ',n*ht)

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
        [ht,rho_0] = pickle.load(open('put_some_path_here'+pickle_name,'rb')) # load data


        if n > 0:
            rho_0 = rho_p
            vertex_val_0 = vertex_val_p
            betaKF_m1 = betaKF_p
        else :

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
            [ht_0,rho_0] = pickle.load(open('put_some_path_here'+pickle_name,'rb'))

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
            [vertex_val_0,betaKF_0] = pickle.load(open('put_some_path_here'+pickle_name,'rb'))

            vertex_val_0 = np.asarray(vertex_val_0)
            betaKF_0 = np.asarray(betaKF_0)

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n+1)+'.p'
        [ht_p,rho_p] = pickle.load(open('put_some_path_here'+pickle_name,'rb'))
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n+1)+'.p'
        [vertex_val_p,betaKF_p] = pickle.load(open('put_some_path_here'+pickle_name,'rb'))
 
        t_mid = 0.5*(n*ht+(n+1)*ht)

        exact_rho_array = exact_rho(pt_primal[...,0],pt_primal[...,1],pt_primal[...,2],t_mid)
        grad_exact_rho_array = grad_exact_rho(pt_primal[...,0],pt_primal[...,1],pt_primal[...,2],t_mid).transpose(1, 2, 0)

        morley_primal, q0_primal = my.get_morley_val(K,pt_primal,bF_primal,bK_primal,0.5*(vertex_val_0+vertex_val_p),0.5*(betaKF_p + betaKF_0))
        grad_q0,val= my.get_grad_morley_val_primal(K, bK_primal, gK_primal, bF_primal, 0.5*(vertex_val_0+vertex_val_p), 0.5*(betaKF_p + betaKF_0))
        grad_morley_primal = grad_q0
        
        diff_morleys = (exact_rho_array - morley_primal)**2
        weighted_sum = np.matmul(diff_morleys,weights_tet)
        L2 = (np.dot(K.area,weighted_sum))**(1/2)

        LinfL2 = np.max([LinfL2,L2])        

        diff_grad_morleys = np.linalg.norm(grad_exact_rho_array - grad_morley_primal,axis=2)**2
        weighted_sum = np.matmul(diff_grad_morleys,weights_tet)
        
        H1_sq = np.dot(K.area,weighted_sum)
        L2H1 += ht*H1_sq

    L2H1 = np.sqrt(L2H1)

    array_LinfL2.append(LinfL2) #LinfL2 
    array_L2H1.append(L2H1) #L2H1 Morley

    hx.append(np.max(K.diam))

    print('LinfL2   ',array_LinfL2)
    print('L2H1 ',array_L2H1)

    if fineness > 0 :
        eoc1 = []
        eoc2 = []
        for i in range(fineness) :
            eoc1.append((np.log((array_LinfL2[i])/(array_LinfL2[i+1])))/(np.log((hx[i])/(hx[i+1]))))
            eoc2.append((np.log((array_L2H1[i])/(array_L2H1[i+1])))/(np.log((hx[i])/(hx[i+1]))))

        print('eocLinfL2: ',np.array(eoc1)) # vgl. Bartels Table 3.2, should be 2
        print('eocL2H1: ',np.array(eoc2)) # vgl. Bartels Table 3.2, should be 1