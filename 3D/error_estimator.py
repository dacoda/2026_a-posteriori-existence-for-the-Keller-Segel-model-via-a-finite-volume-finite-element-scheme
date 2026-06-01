# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# This script computes the a posteriori residual esitmator (27), computing all integrals exactly via a sufficiently exact quadrature rule.
# Here, we ignore the round-off and algebraic errors, as we are only interested in the asymptotics of the residual estimator.
# Make sure you generated all necessary meshes and numerical approximations beforehand (see generate_meshes.py and FVFEscheme.py).

# The quadrature points and weights, stored in triangle12.csv and tetrahedron14.csv, are taken from

# Xiao, Hong and Gimbutas, Zydrunas. 
# A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions, 
# Computers & Mathematics with Applications 59(2), 663–676, 2010. 
# [DOI: 10.1016/j.camwa.2009.10.027]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



import numpy as np
import pickle
import myfun as my
import time
import os


# Get current path
current_path = os.getcwd()

# Create folder if it doesn't exist
folder_name = "data"
folder_path = os.path.join(current_path, folder_name)
os.makedirs(folder_path, exist_ok=True)


tic = time.time()

test = 'manuf' # blowup or manuf
method = 'expl' # expl or impl

spatial = [2,3,4,5,6]
temporal = [32,64,128,256,512]

estimator = []
hx = []

for index in range(len(spatial)) :

    fineness = spatial[index]
    Nt = temporal[index]

    print('fineness: ',fineness)

    pickle_name = 'MESH_3D_UNITCUBE_fineness'+str(fineness)+'.p'
    file_path = os.path.join(folder_path, pickle_name)
    [K,F,K_dual,K_inter] = pickle.load(open(file_path,'rb')) # load mesh

    print('hx: ',np.max(K.diam))
    hx.append(np.max(K.diam))

    hats = []
    for i in range(K_dual.num) : 
        hats.append(my.hat(K_dual,i))

    hats = np.asarray(hats)

    data = np.loadtxt("tetrahedron14.csv", delimiter=",", skiprows=1)  # for quadrature on tetrahedral elements
    weights_tet = data[:,-1]
    xi_ref = np.column_stack((data[:,1],data[:,2],data[:,3])) # physical coordinates

    # transform points from unit tet to physical tet
    pt_primal, J_primal, v0_primal, grads_primal = my.tritrafo_quad_tet(K.points[K.simplices], xi_ref) # shape (K.num,len(xi),3)
    pt_dual, J_dual, v0_dual, grads_dual = my.tritrafo_quad_tet(K_dual.el, xi_ref)
    pt_inter, J_inter, v0_inter, grads_inter = my.tritrafo_quad_tet(K_inter.el, xi_ref)

    data = np.loadtxt("triangle12.csv", delimiter=",", skiprows=1) # for quadrature on triangular faces
    weights2D = data[:,-1]
    points_tri = np.column_stack((data[:,1],data[:,2]))

    face_to_tet = np.array(F.indKs[:,:,0],dtype=int) # face to tet
    loc_face_ind = np.array(F.indKs[:,:,1],dtype=int) 

    areaFs = K.F.area[face_to_tet[:,0],loc_face_ind[:,0]]
    normals = K.F.n[face_to_tet[:,0],loc_face_ind[:,0],:]
    diamFs =  K.F.diam[face_to_tet[:,0],loc_face_ind[:,0]]


    # transform points from unit tri to physical triangular face
    # Canonical face ordering (orientation-free)
    F_el = np.asarray(K.points)[F.simplices] # (N_faces,3,3)
    pt_face_primal, L_face_primal = my.tritrafo_quad_face_and_bary(F_el, points_tri) # shape (F.num,2,len(points),3)

    F_el = np.asarray(K_inter.face_el) # (N_faces,3,3)
    pt_face_inter, L_face_inter = my.tritrafo_quad_face_and_bary(F_el, points_tri) # shape (F.num,2,len(points),3)


    filter = np.array(K_inter.K_primal)
    order = np.argsort(filter)

    sorted = pt_inter[order]
    grads_sorted = grads_inter[order]
    area_sorted = K_inter.area[order]
    pt_inter = sorted.reshape(K.num,12,len(xi_ref),3) # shape (K.num, 12 K_inter per K_primal, number points, 3)
    grads_inter =  grads_sorted.reshape(K.num,12,4,3)
    area_inter = area_sorted.reshape(K.num,12)

    print('quadrature points initialized.')


    # -------------------------------- GET BUBBLE FUNCTION VALUES -------------------------------

    indices = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]

    # primal mesh
    L_primal = my.barycentric_coords(pt_primal, J_primal, v0_primal) # compute barycentric coordinates of quadrature points to efficiently evaluate bubble functions

    bK_primal  = my.bubble_K(L_primal)
    bF_primal = my.bubble_F(L_primal, indices) # shape (K.num, four value per point on for each of the 4 face bubbles per K_primal)

    bF_face_primal = my.bubble_F_face(L_face_primal)
    gK_face_primal  = my.grad_bubble_K_face(K, F, L_face_primal)

    # intersected mesh
    L_inter = my.barycentric_coords(  
                    pt_inter.reshape(K.num*12, len(xi_ref), 3),
                    J_primal.repeat(12, axis=0),
                    v0_primal.repeat(12, axis=0)
                    ).reshape(K.num, 12, len(xi_ref), 4)

    filter_primal = K_inter.face_ident_primal[:,0]

    # Build inverse map
    M = filter_primal.max()
    counts = np.bincount(filter_primal, minlength=M+1) # count how many n's map to each m
    sorted_indices = np.argsort(filter_primal) # Sort n by their m
    inverse_map_all = np.split(sorted_indices, np.cumsum(counts)[:-1]) # split by counts
    inverse_map = [arr for arr in inverse_map_all if len(arr) > 0] # remove empty arrays 

    L_face_inter = my.barycentric_coords_groups(  
                    pt_face_inter,
                    J_primal,
                    v0_primal,
                    inverse_map
                    )

    L_face_inter[np.abs(L_face_inter) < 1e-13] = 0


    bK_inter  = my.bubble_K(L_inter)
    bF_inter = my.bubble_F(L_inter, indices)

    bK_face_inter = my.bubble_K(L_face_inter)
    bF_face_inter = my.bubble_F(L_face_inter, indices)

    # print('bubble on inter mesh done.')


    gK_inter  = my.grad_bubble_K(L_inter, grads_inter)
    lapK_inter = my.laplace_bubble_K(L_inter, grads_inter)

    gF_inter = my.grad_bubble_F(L_inter, grads_inter, indices)
    lapF_inter = my.laplace_bubble_F(L_inter, grads_inter, indices)

    print('bubble function values computed.')


    # ------------------------------------- COMPUTE MORLEY VALUE ------------------------------

    # as only interested in asymptotics here, we can set all constants to one:
    cP, C_ell, C_S, C_Sprime, C_GN, const, C_inf, cTr, cR, cRprime, cUSR = 1,1,1,1,1,1,1,1,1,1,1

    total_time_space = 0
    for n in range(Nt) :

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [ht,aux_rho_0] = pickle.load(open(file_path,'rb')) # load data
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n+1)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [ht,aux_rho_p] = pickle.load(open(file_path,'rb')) # load data
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [vertex_val,betaKF] = pickle.load(open(file_path,'rb')) # load data
        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_morley at time step'+str(n+1)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [vertex_val_p,betaKF_p] = pickle.load(open(file_path,'rb')) # load data

        if n % 50 == 0 :
            print('time ',n*ht)
        
        morley_primal_0, q0_primal_0 = my.get_morley_val(K,pt_primal,bF_primal,bK_primal,vertex_val,betaKF)
        morley_primal_p, q0_primal_p = my.get_morley_val(K,pt_primal,bF_primal,bK_primal,vertex_val_p,betaKF_p)
        morley_inter_0, q0_inter_0 = my.get_morley_val(K,pt_inter,bF_inter,bK_inter,vertex_val,betaKF)
        morley_inter_p, q0_inter_p = my.get_morley_val(K,pt_inter,bF_inter,bK_inter,vertex_val_p,betaKF_p)

        morley_inter_faces = my.get_morley_val_face(K,pt_face_inter,vertex_val,bF_face_inter,bK_face_inter,betaKF,inverse_map)

        grad_morley_primal_faces = my.get_grad_morley_val_face(K,bF_face_primal,gK_face_primal,vertex_val,betaKF,face_to_tet,loc_face_ind)
        grad_morley_inter = my.get_grad_morley_val(K,bF_inter,bK_inter,gF_inter,gK_inter,vertex_val_p,betaKF_p)
        lap_morley_inter = my.get_lap_morley_val(bF_inter,bK_inter,gF_inter,gK_inter,lapF_inter,lapK_inter,betaKF_p)


        # ------------------------------------ COMPUTE C FE VALUE --------------------------------------

        pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_c at time step'+str(n)+'.p'
        file_path = os.path.join(folder_path, pickle_name)
        [aux_c,aux_v] = pickle.load(open(file_path,'rb')) # load data
        
        filter1 = np.array(K_inter.K_primal)
        filter2 = np.array(K_inter.K_dual, dtype = int)
        dual_index = np.asarray(K.pt_ident)[np.asarray(K.simplices)[filter1]]  # shape (N_inter,4)

        # sort filter_primal to group by primal index
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

        sub_index_sorted = np.zeros_like(fp_sorted) # compute sub-index within each primal group
        
        group_start = np.concatenate(([True], fp_sorted[1:] != fp_sorted[:-1])) # find where primal index changes
        sub_index_sorted = np.arange(N_inter) - np.maximum.accumulate(np.where(group_start, np.arange(N_inter), 0))

        c_val = my.get_c_val(aux_c, hats_loc, xi_ref, dual_index, fp_sorted, order, sub_index_sorted, N_primal, N_sub)

        grad_c = np.zeros((N_primal, N_sub, 3), dtype=aux_v.dtype)
        grad_c[fp_sorted, sub_index_sorted, :] = aux_v[fd_sorted]

        # get FE jump term
        plus  = aux_v[K_dual.face_to_tet[:,0]]
        minus = aux_v[K_dual.face_to_tet[:,1]]

        jump_grad_c = np.sum((plus - minus) * K_dual.face_normals, axis=1)


        # --------------------------------------- A POSTERIORI ESTIMATOR ------------------------------------------------
        total = 0

        # Lemma 6.1 ...
        res_inter = ((morley_inter_p - morley_inter_0)/ht + np.einsum('tkni,tki->tkn', grad_morley_inter, grad_c) - lap_morley_inter)**2
        weighted_sum = np.einsum('tki,i->tk', res_inter, weights_tet)
        quadrature = np.sum(K.diam**2 @ (area_inter*weighted_sum))

        total += quadrature

        jumps_FV = np.einsum('tki,ti->tk', grad_morley_primal_faces[:,0,...] - grad_morley_primal_faces[:,1,...], normals)**2
        weighted_sum = np.matmul(jumps_FV,weights2D)
        quadrature = cR*(diamFs @ (areaFs*weighted_sum))

        total += quadrature
        #... Lemma 6.1

        if n==0 :

            grad_morley_inter_diff = my.get_grad_morley_val(K,bF_inter,bK_inter,gF_inter,gK_inter,np.asarray(vertex_val_p) - np.asarray(vertex_val),np.asarray(betaKF_p) - np.asarray(betaKF))

            # Lemma 6.2 ...
            diff_time2 = ((morley_primal_p - morley_primal_0)/ht - (aux_rho_p[:,None] - aux_rho_0[:,None])/ht)**2
            weighted_sum = np.matmul(diff_time2,weights_tet)
            quadrature = (np.dot(K.area,weighted_sum))

            total += quadrature
            # ... Lemma 6.2

            # Lemma 6.3 ...
                
            diff_morleys = (morley_primal_p - morley_primal_0)**2
            weighted_sum = np.matmul(diff_morleys,weights_tet)
            quadrature_diff = (np.dot(K.area,weighted_sum))
                
            max_morley_p = np.max(vertex_val_p) + np.max(np.sum(betaKF_p,axis=1))
            max_diff_morley_p = np.max(np.abs(vertex_val_p - vertex_val)) + np.max(np.sum(np.abs(np.asarray(betaKF_p) - np.asarray(betaKF)),axis=1)) # betas can be estimated tighter if necessary

            quadrature = C_ell*(max_morley_p + max_diff_morley_p)*quadrature_diff 
            
            total += quadrature

            val = my.get_conv_terms(K_inter,aux_v,aux_rho_0,morley_inter_faces)
            weighted_sum = np.matmul(val**2,weights2D)
            quadrature = cRprime*(K_inter.face_diam @ (K_inter.face_areas*weighted_sum))

            total += quadrature

            # FE scheme ...
            res_FE = (c_val - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', res_FE, weights_tet)
            quadrature_res = (np.sum(K.diam**2 @ (area_inter*weighted_sum)))**(1/2)

            quadrature_jump = (K_dual.face_diam @ (K_dual.face_areas*jump_grad_c**2))

            diff_q0_morley = (q0_inter_0 - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', diff_q0_morley, weights_tet)
            quadrature3 = (np.sum(K.diam**2 @ (area_inter*weighted_sum)))

            diff_FV_morley = (aux_rho_0[:,None] - morley_primal_0)**2
            weighted_sum = np.matmul(diff_FV_morley,weights_tet)
            quadrature4 = (np.dot(K.area,weighted_sum))**(1/2)

            quadrature = cUSR*(quadrature_res + 1/2*quadrature_jump + quadrature3 + 4/3 * quadrature4) 

            total += quadrature *max_diff_morley_p
            # ... FE scheme

            # ... Lemma 6.3

            # first time step ...
            early_inter = np.linalg.norm(np.einsum('tki,tkj->tkij',morley_inter_p - morley_inter_0,grad_c) + grad_morley_inter_diff,axis=3)**2
            weighted_sum = np.einsum('tki,i->tk', early_inter, weights_tet)
            quadrature = (np.sum(K.diam**2 @ (area_inter*weighted_sum)))

            total += quadrature
            # ... first time step


        else:

            pickle_name = method+test+'_fineness'+str(fineness)+'_Nt'+str(Nt)+'_rho at time step'+str(n-1)+'.p'
            file_path = os.path.join(folder_path, pickle_name)
            [ht,aux_rho_m] = pickle.load(open(file_path,'rb')) # load data

            # Lemma 6.2 ...
            diff_time1 = ((aux_rho_p[:,None] - aux_rho_0[:,None])/ht - (aux_rho_0[:,None] - aux_rho_m[:,None])/ht)**2
            diff_time2 = ((morley_primal_p - morley_primal_0)/ht - (aux_rho_p[:,None] - aux_rho_0[:,None])/ht)**2
            weighted_sum = np.matmul(diff_time2,weights_tet)
            quadrature = (np.dot(K.area,weighted_sum+diff_time1[:,0]))

            total += quadrature
            # ... Lemma 6.2

            # Lemma 6.3 ...
            diff_morleys_prev = quadrature_diff
                
            diff_morleys = (morley_primal_p - morley_primal_0)**2
            weighted_sum = np.matmul(diff_morleys,weights_tet)
            quadrature_diff = (np.dot(K.area,weighted_sum))

            max_morley_0 = max_morley_p
                
            max_morley_p = np.max(vertex_val) + np.max(np.sum(betaKF,axis=1))

            max_diff_morley_p = np.max(np.abs(vertex_val_p - vertex_val)) + np.max(np.sum(np.abs(np.asarray(betaKF_p) - np.asarray(betaKF)),axis=1)) # betas can be estimated tighter if necessary

            quadrature = C_ell*(max_morley_p + max_diff_morley_p)*quadrature_diff  + max_morley_0*diff_morleys_prev
            
            total += quadrature

            val = my.get_conv_terms(K_inter,aux_v,aux_rho_0,morley_inter_faces)
            weighted_sum = np.matmul(val**2,weights2D)
            quadrature = cRprime*(K_inter.face_diam @ (K_inter.face_areas*weighted_sum))

            total += quadrature

            # FE scheme ...
            res_FE = (c_val - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', res_FE, weights_tet)
            quadrature_res = (np.sum(K.diam**2 @ (area_inter*weighted_sum)))**(1/2)

            quadrature_jump = (K_dual.face_diam @ (K_dual.face_areas*jump_grad_c**2))

            diff_q0_morley = (q0_inter_0 - morley_inter_0)**2
            weighted_sum = np.einsum('tki,i->tk', diff_q0_morley, weights_tet)
            quadrature3 = (np.sum(K.diam**2 @ (area_inter*weighted_sum)))

            diff_FV_morley = (aux_rho_0[:,None] - morley_primal_0)**2
            weighted_sum = np.matmul(diff_FV_morley,weights_tet)
            quadrature4 = (np.dot(K.area,weighted_sum))

            quadrature = cUSR*(quadrature_res + 1/2*quadrature_jump + quadrature3 + 4/3 * quadrature4) 
            
            total += quadrature *max_morley_0
            # ... FE scheme

            # ... Lemma 6.3

        # print('total: ',total)

        total_time_space += total*ht

    total_time_space = np.sqrt(total_time_space)
    estimator.append(total_time_space)

    print('estimator :',estimator)

    if fineness > spatial[0] : # generate estimated order of convergence for Table 2.
        eoc = []
        for i in range(fineness-spatial[0]) :
            eoc.append((np.log((estimator[i])/(estimator[i+1])))/(np.log((hx[i])/(hx[i+1]))))

        print('eoc: ',eoc)

toc = time.time()
print('time for eval: ', (toc-tic)/60)

