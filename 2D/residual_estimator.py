# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes and stores the a posteriori residual esitmator (29) and other quantities needed to compute the full error estimators in Theorem 6.8.
# The full estimator, including round-off errors and algebraic errors, will be assembled in compare_stability.py.
# We compute all integrals exactly via a sufficiently exact quadrature rule.

# Make sure you generated all necessary meshes and numerical approximations beforehand (see generate_meshes.py and FVFEscheme.py).

# The quadrature points and weights, stored in triangle10.csv, are taken from

# Xiao, Hong and Gimbutas, Zydrunas. 
# A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions, 
# Computers & Mathematics with Applications 59(2), 663–676, 2010. 
# [DOI: 10.1016/j.camwa.2009.10.027]

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
test = 'diff' # blowup, idff or manuf


# Configuration used to generate values in Table 1:
test = 'diff' 
method = 'expl'
spatial = [4,5,6,7]
temporal = [25,50,200,800]
TT = [0.0005,0.0005,0.001,0.001]


def initial_rho(x,y) :
    val = np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+1
    return val


for index in range(len(spatial)):  

    fineness = spatial[index] # number of refinements of mesh before calculating numerical solution
    Nt = temporal[index] # number of time steps

    print('fineness: ',fineness)

    pickle_name = 'MESH_2D_UNITSQUARE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    [K,E,K_dual,K_inter] = pickle.load(open(file_path,'rb')) # load mesh

    hats = []
    for i in range(K_dual.num) : 
        hats.append(my.hat(K_dual,i))
    hats = np.asarray(hats)

    data = np.loadtxt("triangle10.csv", delimiter=",", skiprows=1)
    weights_tri = data[:,-1]
    xi_ref = np.column_stack((data[:,1],data[:,2])) # physical coordinates

    # transform points from unit tet to physical tet
    pt_primal, J_primal, v0_primal, grads_primal = my.tritrafo_quad_tri(K.points[K.simplices], xi_ref) # shape (K.num,len(xi),3)
    pt_dual, J_dual, v0_dual, grads_dual = my.tritrafo_quad_tri(K_dual.el, xi_ref)
    pt_inter, J_inter, v0_inter, grads_inter = my.tritrafo_quad_tri(K_inter.el, xi_ref)

    # Standard Gauss-Legendre on [-1,1]
    nodes, weights = np.polynomial.legendre.leggauss(5) # -> exactness 2n-1 = 9 > 8
    # Affine map [-1,1] -> [0,1]
    points_edge = 0.5 * (nodes + 1.0)
    weights1D = 0.5 * weights

    face_to_tet = np.array(E.indKs[:,:,0],dtype=int) # face to tet
    loc_face_ind = np.array(E.indKs[:,:,1],dtype=int) 

    areaEs = K.E.area[face_to_tet[:,0],loc_face_ind[:,0]]
    normals = K.E.n[face_to_tet[:,0],loc_face_ind[:,0],:]
    diamEs = areaEs


    # transform points from unit tri to physical triangular face
    # Canonical face ordering (orientation-free)
    E_el = np.asarray(K.points)[E.simplices] # (N_faces,3,3)
    pt_edge_primal, L_edge_primal = my.tritrafo_quad_edge_and_bary(E_el, points_edge) # shape (F.num,2,len(points),3)

    E_el = np.asarray(K_inter.edge_el) # (N_faces,3,3)
    pt_edge_inter, L_edge_inter = my.tritrafo_quad_edge_and_bary(E_el, points_edge) # shape (F.num,2,len(points),3)

    filter = np.array(K_inter.K_primal)
    order = np.argsort(filter)

    sorted = pt_inter[order]
    grads_sorted = grads_inter[order]
    area_sorted = K_inter.area[order]
    pt_inter = sorted.reshape(K.num,6,len(xi_ref),2) # shape (K.num, 6 K_inter per K_primal, number points, 2)
    grads_inter =  grads_sorted.reshape(K.num,6,3,2)
    area_inter = area_sorted.reshape(K.num,6)

    print('quadrature points initialized.')


    # -------------------------------- GET BUBBLE FUNCTION VALUES -------------------------------

    indices = [[1,2],[0,2],[0,1]]

    # primal mesh
    L_primal = my.barycentric_coords(pt_primal, J_primal, v0_primal) # compute barycentric coordinates of quadrature points to efficiently evaluate bubble functions

    bK_primal  = my.bubble_K(L_primal) # compute bubble function values at each quadrature point
    bF_primal = my.bubble_E(L_primal, indices) # shape (K.num, four value per point on for each of the 4 face bubbles per K_primal)

    bF_edge_primal = my.bubble_E_edge(L_edge_primal)
    gK_edge_primal  = my.grad_bubble_K_edge(K, E, L_edge_primal)

    # intersected mesh
    L_inter = my.barycentric_coords(  
                    pt_inter.reshape(K.num*6, len(xi_ref), 2),
                    J_primal.repeat(6, axis=0),
                    v0_primal.repeat(6, axis=0)
                    ).reshape(K.num, 6, len(xi_ref), 3)

    filter_primal = K_inter.edge_ident_primal[:,0]

    # Build inverse map
    M = filter_primal.max()
    counts = np.bincount(filter_primal, minlength=M+1) # count how many n map to each 
    sorted_indices = np.argsort(filter_primal) # sort n per m
    inverse_map_all = np.split(sorted_indices, np.cumsum(counts)[:-1]) #  split by counts
    inverse_map = [arr for arr in inverse_map_all if len(arr) > 0] # remove empty arrays

    L_face_inter = my.barycentric_coords_groups(  
                    pt_edge_inter,
                    J_primal,
                    v0_primal,
                    inverse_map
                    )

    L_face_inter[np.abs(L_face_inter) < 1e-13] = 0


    bK_inter  = my.bubble_K(L_inter)
    bF_inter = my.bubble_E(L_inter, indices)

    bK_face_inter = my.bubble_K(L_face_inter)
    bF_face_inter = my.bubble_E(L_face_inter, indices)

    gK_inter  = my.grad_bubble_K(L_inter, grads_inter)
    lapK_inter = my.laplace_bubble_K(L_inter, grads_inter)

    gF_inter = my.grad_bubble_E(L_inter, grads_inter, indices)
    lapF_inter = my.laplace_bubble_E(L_inter, grads_inter, indices)

    print('bubble function values computed.')


    # ------------------------------------- COMPUTE MORLEY VALUE ------------------------------

    # set upper bounds for required constants
    cP = 1/(2*np.pi)**2 # Poincare-Wirtinger constant on the flat torus
    C_ell = 1

    # set upper bounds for required constants
    C_S = 2.1357917 # see [43]

    B1 = 8/5*C_S**3*C_ell**2
    B2 = 864/125*C_S**6*C_ell**4

    # set further constants
    K_el = K.points[K.simplices]
    cUSR = 0
    for j in range(K.num) :
        hK = my.diam(K_el[j])
        a = np.linalg.norm(K.E.el[j][0]) 
        b = np.linalg.norm(K.E.el[j][1]) 
        c = np.linalg.norm(K.E.el[j][2]) 
        inradK = 1/2*np.sqrt(((b+c-a)*(c+a-b)*(a+b-c))/(a+b+c)) # 2D formula
        cUSR = np.max([cUSR,hK/(2*inradK)])

    cTr = np.sqrt(2*cUSR) # L2
    cR = 0.5*cTr*np.sqrt(3*(1+cUSR**2*cP**2))
    Csz = np.sqrt(cUSR/2)+1

    cUSR = 0
    for j in range(K_inter.num) :
        hK = my.diam(K_inter.el[j])
        a = np.linalg.norm(K_inter.E.el[j][0]) 
        b = np.linalg.norm(K_inter.E.el[j][1]) 
        c = np.linalg.norm(K_inter.E.el[j][2]) 
        inradK = 1/2*np.sqrt(((b+c-a)*(c+a-b)*(a+b-c))/(a+b+c))  # 2D formula
        cUSR = np.max([cUSR,hK/(2*inradK)])
    cRprime = cTr*np.sqrt(6*(1+cUSR**2*cP**2))

    total_time_space = 0
    morleyL3 = []
    list_time_space =[]
    for n in range(Nt) :

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [ht,aux_rho_0,rhs] = pickle.load(open(file_path,'rb')) # load data
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n+1)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [ht,aux_rho_p,rhs] = pickle.load(open(file_path,'rb')) # load data
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [vertex_val,betaKF] =  pickle.load(open(file_path,'rb')) # load data
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n+1)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [vertex_val_p,betaKF_p] =  pickle.load(open(file_path,'rb')) # load data
       
        if n % 10 == 0 :
            print('n ',n)
        
        morley_primal_0, q0_primal_0 = my.get_morley_val(K,pt_primal,bF_primal,bK_primal,vertex_val,betaKF)
        morley_primal_p, q0_primal_p = my.get_morley_val(K,pt_primal,bF_primal,bK_primal,vertex_val_p,betaKF_p)
        morley_inter_0, q0_inter_0 = my.get_morley_val(K,pt_inter,bF_inter,bK_inter,vertex_val,betaKF)
        morley_inter_p, q0_inter_p = my.get_morley_val(K,pt_inter,bF_inter,bK_inter,vertex_val_p,betaKF_p)

        morley_inter_faces = my.get_morley_val_edge(K,pt_edge_inter,vertex_val,bF_face_inter,bK_face_inter,betaKF,inverse_map)

        grad_morley_primal_faces = my.get_grad_morley_val_edge(K,bF_edge_primal,gK_edge_primal,vertex_val,betaKF,face_to_tet,loc_face_ind)
        grad_morley_inter = my.get_grad_morley_val(K,bF_inter,bK_inter,gF_inter,gK_inter,vertex_val_p,betaKF_p)
        lap_morley_inter = my.get_lap_morley_val(bF_inter,bK_inter,gF_inter,gK_inter,lapF_inter,lapK_inter,betaKF_p)


        # ------------------------------------ COMPUTE C FE VALUE --------------------------------------
        
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_c at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [aux_c,aux_v,b] =  pickle.load(open(file_path,'rb')) # load data

        filter1 = np.array(K_inter.K_primal)
        filter2 = np.array(K_inter.K_dual, dtype = int)
        dual_index = np.asarray(K.pt_ident)[np.asarray(K.simplices)[filter1]]  # shape (N_inter,4)

        # Sort filter_primal to group by primal index
        order = np.argsort(filter1)
        filter1_sorted = filter1[order]
        filter2_sorted = filter2[order]
        dual_index_sorted = dual_index[order]

        hats_loc = hats[filter2]  # (N_inter,4,4)

        N_primal = filter1.max() + 1
        N_inter = len(filter1)
        N_sub = N_inter // N_primal

        order = np.argsort(filter1, kind="stable") # sort by primal index
        fp_sorted = filter1[order]
        fd_sorted = filter2[order]
        sub_index_sorted = np.zeros_like(fp_sorted) # Compute sub-index within each primal group
        group_start = np.concatenate(([True], fp_sorted[1:] != fp_sorted[:-1])) # find where primal index changes
        sub_index_sorted = np.arange(N_inter) - np.maximum.accumulate(np.where(group_start, np.arange(N_inter), 0))

        c_val = my.get_c_val(aux_c, hats_loc, xi_ref, dual_index, fp_sorted, order, sub_index_sorted, N_primal, N_sub)

        grad_c = np.zeros((N_primal, N_sub, 2), dtype=aux_v.dtype)
        grad_c[fp_sorted, sub_index_sorted, :] = aux_v[fd_sorted]

        # get FE jump term
        plus  = aux_v[K_dual.face_to_tet[:,0]]
        minus = aux_v[K_dual.face_to_tet[:,1]]

        jump_grad_c = np.sum((plus - minus) * K_dual.face_normals, axis=1)


        # --------------------------------- computable terms from stability estimats ------------------------------------

        if n==0:
            exact_rho_array = initial_rho(pt_primal[...,0],pt_primal[...,1])
            morley_primal, q0_primal = my.get_morley_val(K,pt_primal,bF_primal,bK_primal,vertex_val,betaKF)
            diff_morleys = (exact_rho_array - morley_primal)**2
            weighted_sum = np.matmul(diff_morleys,weights_tri)
            initialL2 = (np.dot(K.area,weighted_sum))**(1/2)

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_initL2.p'
            file_path = os.path.join(folder_path, pickle_name)
            pickle.dump(initialL2,open(file_path,'wb')) # store data

        weighted_sum = np.matmul(morley_primal_0**3,weights_tri) # L3 norm
        morleyL3.append((np.dot(K.area,weighted_sum))**(1/3))


        # --------------------------------------- A POSTERIORI ESTIMATOR ------------------------------------------------
        total = 0

        # Lemma 6.1 ...
        res_inter = ((morley_inter_p - morley_inter_0)/ht + np.einsum('tkni,tki->tkn', grad_morley_inter, grad_c) - lap_morley_inter)**2
        weighted_sum = np.einsum('tki,i->tk', res_inter, weights_tri)
        quadrature = np.sum(K.diam**2 @ (area_inter*weighted_sum))

        total += quadrature

        jumps_FV = np.einsum('tki,ti->tk', grad_morley_primal_faces[:,0,...] - grad_morley_primal_faces[:,1,...], normals)**2
        weighted_sum = np.matmul(jumps_FV,weights1D)
        quadrature = cR*(diamEs @ (areaEs*weighted_sum))

        total += quadrature
        #... Lemma 6.1


        if n==0 :

            grad_morley_inter_diff = my.get_grad_morley_val(K,bF_inter,bK_inter,gF_inter,gK_inter,np.asarray(vertex_val_p) - np.asarray(vertex_val),np.asarray(betaKF_p) - np.asarray(betaKF))

            # Lemma 6.2 ...
            diff_time2 = ((morley_primal_p - morley_primal_0)/ht - (aux_rho_p[:,None] - aux_rho_0[:,None])/ht)**2
            weighted_sum = np.matmul(diff_time2,weights_tri)
            quadrature = np.dot(K.area,weighted_sum)

            total += quadrature
            # ... Lemma 6.2

            # Lemma 6.3 ...
            diff_morleys = (morley_primal_p - morley_primal_0)**2
            weighted_sum = np.matmul(diff_morleys,weights_tri)
            quadrature_diff = np.dot(K.area,weighted_sum)
                
            max_morley_p = np.max(vertex_val_p) + np.max(np.sum(betaKF_p,axis=1))
            max_diff_morley_p = np.max(np.abs(vertex_val_p - vertex_val)) + np.max(np.sum(np.abs(np.asarray(betaKF_p) - np.asarray(betaKF)),axis=1)) # betas can be estimated tighter if necessary

            quadrature = C_ell*(max_morley_p + max_diff_morley_p)*quadrature_diff 
            
            total += quadrature

            val = my.get_conv_terms(K_inter,aux_v,aux_rho_0,morley_inter_faces)
            weighted_sum = np.matmul(val**2,weights1D)
            quadrature = cRprime*(K_inter.edge_length @ (K_inter.edge_length*weighted_sum))

            total += quadrature

            # FE scheme ...
            res_FE = (c_val - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', res_FE, weights_tri)
            quadrature_res = np.sum(K.diam**2 @ (area_inter*weighted_sum))

            quadrature_jump = K_dual.edge_length @ (K_dual.edge_length*jump_grad_c**2)

            diff_q0_morley = (q0_inter_0 - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', diff_q0_morley, weights_tri)
            quadrature3 = (np.sum(K.diam**2 @ (area_inter*weighted_sum)))**(1/2)

            diff_FV_morley = (aux_rho_0[:,None] - morley_primal_0)**2
            weighted_sum = np.matmul(diff_FV_morley,weights_tri)
            quadrature4 = (np.dot(K.area,weighted_sum))**(1/2)

            quadrature = Csz*(quadrature_res + 1/2*quadrature_jump + quadrature3 + 4/3 * quadrature4) 

            total += quadrature *max_diff_morley_p
            # ... FE scheme

            # ... Lemma 6.3


            # first time step ...
            early_inter = np.linalg.norm(np.einsum('tki,tkj->tkij',morley_inter_p - morley_inter_0,grad_c) + grad_morley_inter_diff,axis=3)**2
            weighted_sum = np.einsum('tki,i->tk', early_inter, weights_tri)
            quadrature = np.sum(K.diam**2 @ (area_inter*weighted_sum))

            total += quadrature
            # ... first time step


        else:

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n-1)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [ht,aux_rho_m,rhs] = pickle.load(open(file_path,'rb')) # load data

            # Lemma 6.2 ...
            diff_time1 = ((aux_rho_p[:,None] - aux_rho_0[:,None])/ht - (aux_rho_0[:,None] - aux_rho_m[:,None])/ht)**2
            diff_time2 = ((morley_primal_p - morley_primal_0)/ht - (aux_rho_p[:,None] - aux_rho_0[:,None])/ht)**2
            weighted_sum = np.matmul(diff_time2,weights_tri)
            quadrature = (np.dot(K.area,weighted_sum+diff_time1[:,0]))**(1/2)

            total += quadrature
            # ... Lemma 6.2
                

            # Lemma 6.3 ...
            diff_morleys_prev = quadrature_diff
            diff_morleys = (morley_primal_p - morley_primal_0)**2
            weighted_sum = np.matmul(diff_morleys,weights_tri)
            quadrature_diff = (np.dot(K.area,weighted_sum))**(1/2)

            max_morley_0 = max_morley_p
            max_morley_p = np.max(vertex_val) + np.max(np.sum(betaKF,axis=1))
            max_diff_morley_p = np.max(np.abs(vertex_val_p - vertex_val)) + np.max(np.sum(np.abs(np.asarray(betaKF_p) - np.asarray(betaKF)),axis=1)) # betas can be estimated tighter if necessary

            quadrature = C_ell*(max_morley_p + max_diff_morley_p)*quadrature_diff  + max_morley_0*diff_morleys_prev
            
            total += quadrature

            val = my.get_conv_terms(K_inter,aux_v,aux_rho_0,morley_inter_faces)
            weighted_sum = np.matmul(val**2,weights1D)
            quadrature = cRprime*(K_inter.edge_length @ (K_inter.edge_length*weighted_sum))

            total += quadrature

            # FE scheme ...
            res_FE = (c_val - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', res_FE, weights_tri)
            quadrature_res = np.sum(K.diam**2 @ (area_inter*weighted_sum))

            quadrature_jump = K_dual.edge_length @ (K_dual.edge_length*jump_grad_c**2)

            diff_q0_morley = (q0_inter_0 - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', diff_q0_morley, weights_tri)
            quadrature3 = np.sum(K.diam**2 @ (area_inter*weighted_sum))

            diff_FV_morley = (aux_rho_0[:,None] - morley_primal_0)**2
            weighted_sum = np.matmul(diff_FV_morley,weights_tri)
            quadrature4 = np.dot(K.area,weighted_sum)

            quadrature = Csz*(quadrature_res + 1/2*quadrature_jump + quadrature3 + 4/3 * quadrature4) 
            
            total += quadrature *max_morley_0
            # ... FE scheme

            # ... Lemma 6.3

        total_time_space += total*ht 
        list_time_space.append(total)

    total_time_space = np.sqrt(total_time_space) # = theta_Omega
    morleyL3 = np.asarray(morleyL3)

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_theta.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(list_time_space,open(file_path,'wb')) # store data

    pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_L3.p'
    file_path = os.path.join(folder_path, pickle_name)
    pickle.dump(morleyL3,open(file_path,'wb')) # store data
