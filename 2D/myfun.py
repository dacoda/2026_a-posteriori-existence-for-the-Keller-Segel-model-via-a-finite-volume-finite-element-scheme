import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
import pickle
import scipy as sp

import itertools



# ------------------------------ MESH CLASSES ----------------------------------------------- 

class primal_mesh :
    def __init__(self, fineness):
        
        hx = 1/2
        [tri,points,nppoints,numtri] = initialmesh()
        el = nppoints[tri.simplices]

        # refine zerost mesh subject to fineness
        for i in range(fineness) :
            hx = 1/2*hx
            [tri,points,nppoints,numtri] = refinemesh(points, el, numtri)
            el = nppoints[tri.simplices]

        
        # make the mesh connect periodically
        # 1) identify neighboring edges (across boundary)
        neighbors = tri.neighbors.copy()
        simplices = tri.simplices

        indices = [[1,2],[0,2],[0,1]]

        for j in range(len(simplices)):

            for k in range(3):

                if neighbors[j][k] == -1:   # boundary edge

                    F = el[j][indices[k]]
                    midF = 0.5*(F[0] + F[1])

                    # x periodic
                    if midF[0] > 1 - 1e-6:
                        pt = midF - np.array([1,0]) + 1e-12*np.array([1,0])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = l

                    elif midF[0] < 1e-6:
                        pt = midF + np.array([1,0]) - 1e-12*np.array([1,0])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = l

                    # y periodic
                    elif midF[1] > 1 - 1e-6:
                        pt = midF - np.array([0,1]) + 1e-12*np.array([0,1])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = l

                    elif midF[1] < 1e-6:
                        pt = midF + np.array([0,1]) - 1e-12*np.array([0,1])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = l

                    else:
                        print("Warning: periodic neighbor not assigned")

        # 2) identify mesh nodes
        pt_ident = []
        pt_reduced = []

        for i in range(len(points)):
            pt = points[i]

            if pt[0] > 1 - 1e-6:
                pt = [pt[0]-1, pt[1]]

            if pt[1] > 1 - 1e-6:
                pt = [pt[0], pt[1]-1]

            j = linear_search(pt_reduced, pt)

            if j > -0.5:
                pt_ident.append(j)
            else:
                pt_reduced.append(pt)
                pt_ident.append(len(pt_reduced)-1)


        self.pt_reduced = pt_reduced
        self.pt_ident = pt_ident

        order = np.argsort(simplices,axis=1)

        # Use advanced indexing to apply permutation to simplices
        rows = np.arange(simplices.shape[0])[:, None]  # shape (M,1)
        self.simplices = simplices[rows, order]       # shape (M,3)

        # Apply same permutation to neighbors
        self.neighbors = neighbors[rows, order]

        self.hx = hx
        self.points = np.asarray(points)
        self.el = self.points[self.simplices]
        self.num = len(self.el)     


        areaK = []
        auxCC = []
        for i in range(self.num) :
            CCi = Voronoi(self.el[i]).vertices[0] #circumcenter(self.el[i])
            auxCC.append(CCi)
        self.CC = np.asarray(auxCC)

        areaK = np.abs(np.cross(self.el[:,1] - self.el[:,0], self.el[:,2] - self.el[:,0])) / 2
        self.area =  np.asarray(areaK)

        self.TinK = [[] for i in range(self.num)]


        # ---- Build adjacency ----
        pt_ident = np.asarray(self.pt_ident)
        adjacent = [[] for _ in range(len(self.pt_reduced))]

        # Precompute simplex membership map
        vertex_to_elements = [[] for _ in range(len(self.points))]

        for elem_id, tet in enumerate(self.simplices):
            for v in tet:
                vertex_to_elements[v].append(elem_id)

        # Assign reduced-point adjacency
        for reduced_id in range(len(self.pt_reduced)):
            original_vertices = np.where(pt_ident == reduced_id)[0]
            neigh = set()
            for v in original_vertices:
                neigh.update(vertex_to_elements[v])
            adjacent[reduced_id] = list(neigh)

        self.adjacent = adjacent

        weights = []
        for i in range(len(self.pt_reduced)):

            x0 = self.pt_reduced[i]

            CCi = self.CC[self.adjacent[i]] # all circumcenters touching vertex i
            d = CCi - x0 # vectors to circumcenters

            dx = d[:,0]
            dy = d[:,1]

            Rx = dx.sum()
            Ry = dy.sum()

            Ixx = np.sum(dx*dx)
            Iyy = np.sum(dy*dy)
            Ixy = np.sum(dx*dy)

            denom = Ixx*Iyy - Ixy**2

            lambdax = (Ixy*Ry - Iyy*Rx) / denom
            lambday = (Ixy*Rx - Ixx*Ry) / denom

            wi = 1 + lambdax*dx + lambday*dy
            weights.append(wi.tolist())

        self.weights = weights

        # Compute pairwise distances for each triangle
        diff01 = self.el[:, 0, :] - self.el[:, 1, :]
        diff02 = self.el[:, 0, :] - self.el[:, 2, :]
        diff12 = self.el[:, 1, :] - self.el[:, 2, :]
        
        dist01 = np.linalg.norm(diff01, axis=1)
        dist02 = np.linalg.norm(diff02, axis=1)
        dist12 = np.linalg.norm(diff12, axis=1)
        
        # Diameter is the maximum distance
        diameters = np.maximum(np.maximum(dist01, dist02), dist12)

        self.diam = diameters

        self.hat = [hat(self, i) for i in range(self.num)]
   

    def init_edges(self) :
        self.E = edges_of_element(self)


class edges_of_element:
    def __init__(self, K):

        N = K.num                 # number of triangles
        el = K.el                 # triangle coordinates (N,3,2)
        simplices = K.simplices   # triangle vertex indices (N,3)

        # Define edge vertex pairs for a triangle
        edge_pairs = np.array([[1,2],[0,2],[0,1]])  # each triangle has 3 edges

        # Preallocate arrays
        auxE = np.zeros((N,3,2,2))    # coordinates of edges (N,3 edges, 2 vertices, 2 coords)
        auxEindex = np.zeros((N,3,2), dtype=int)  # vertex indices of edges
        areaE = np.zeros((N,3))       # edge lengths
        nKE = np.zeros((N,3,2))       # unit outer normals
        dE = np.full((N,3), np.inf)   # distance to neighbor circumcenters

        neighbors = K.neighbors   # shape (N,3)

        # Vectorized edge coordinates and indices 
        for j in range(3):
            auxE[:,j] = el[:,edge_pairs[j]]               # coordinates of edge j
            auxEindex[:,j] = simplices[:,edge_pairs[j]]   # indices of edge j

        #  Vectorized edge lengths 
        edge_vecs = auxE[:,:,0,:] - auxE[:,:,1,:]  # vector from vertex 0 to 1 for all edges
        areaE = np.linalg.norm(edge_vecs, axis=2)  # edge lengths

        #  Unit outer normals (vectorized with broadcasting) 
        # Assuming unitouternormal expects triangle coords + edge coords
        # Here we vectorize by passing arrays of shape (N,3,2,2)
        for i in range(N):
            for j in range(3):
                nKE[i,j] = unitouternormal(el[i], auxE[i,j])  # still per-edge call

        #  Distance to neighbor circumcenters (vectorized partially) 
        CC = K.CC
        for i in range(N):
            for j in range(3):
                nb = neighbors[i,j]
                if nb >= 0:
                    val = np.linalg.norm(CC[i] - CC[nb])
                    # apply periodic shifts
                    for k in range(2):
                        if val > 0.5:
                            shift = np.zeros(2)
                            shift[k] = 1 if CC[i,k] < 0.1 else -1
                            val = np.linalg.norm(CC[i]+shift - CC[nb])
                    dE[i,j] = val

        # Save arrays
        self.el = auxE
        self.simplices = auxEindex
        self.area = areaE
        self.dist = dE
        self.n = nKE
        self.num = N
        self.dual = -np.ones((N,3,2))  # dual placeholders
   


class edges:
    def __init__(self, K):
        """
        Vectorized 2D edges class with simplices and opposite vertices.
        Computes unique edges, adjacency, and interior/boundary flags.
        """
        indices = [[1,2],[0,2],[0,1]]
        max_edges = 3 * K.num  # max possible edges

        # Preallocate arrays
        aux_el = np.zeros((max_edges, 2, 2))       # coordinates of edges
        aux_indKs = -np.ones((max_edges, 2, 2), dtype=int)  # triangle/edge references
        aux_simplices = np.zeros((max_edges, 2), dtype=int) # vertex indices of edge
        aux_opp = -np.ones((max_edges, 2), dtype=int)      # vertex opposite in triangle

        indexE = 0
        edge_dict = {}  # key: sorted tuple of vertex indices, value: edge index

        # Loop over all triangles and edges
        for i in range(K.num):
            neighbors = K.neighbors[i]
            for j in range(3):
                nb = neighbors[j]
                # global vertex indices of this edge
               
                edge_key= tuple(sorted([i, nb]))

                if nb >= 0:  # interior edge
                    if edge_key in edge_dict:  # already processed
                        
                        idx = edge_dict[edge_key]

                        aux_indKs[idx][1] = [i, j]
                        aux_opp[idx][1] = K.simplices[i][j]

                    else:  # new edge

                        aux_el[indexE] = K.E.el[i][j]
                        aux_indKs[indexE][0] = [i, j]
                        aux_simplices[indexE] = K.simplices[i][indices[j]]
                        aux_opp[indexE][0] = K.simplices[i][j]
                        edge_dict[edge_key] = indexE
                        indexE += 1

                else:  # boundary edge
                    print('Warning: periodic boundary for primal mesh broken')

        # Trim arrays to actual number of edges
        self.num = indexE
        self.el = aux_el[:indexE]
        self.indKs = aux_indKs[:indexE]
        self.simplices = aux_simplices[:indexE]
        self.opp = aux_opp[:indexE]




def initialmesh() :

    points = [[0,0],[1,0],[0,1],[1,1]]
    nppoints = np.array(points)

    points.append(1/2*(nppoints[0]+nppoints[1]))
    points.append(1/2*(nppoints[0]+nppoints[2]))
    points.append(1/2*(nppoints[3]+nppoints[1]))
    points.append(1/2*(nppoints[3]+nppoints[2]))

    points.append(0.35*nppoints[0]+0.65*nppoints[3])
    points.append(0.65*nppoints[0]+0.35*nppoints[3])

    points.append(0.295*nppoints[1]+0.705*nppoints[2]) 
    points.append(0.705*nppoints[1]+0.295*nppoints[2])

    nppoints = np.array(points)

    tri = Delaunay(points) # d=2
    
    # tri.simplices gives indices that form the triangles
    # points[tri.simplices] gives vertices of the mesh elements
    # see documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

    numtri = 14 # number of triangles in mesh

    return [tri, points, nppoints, numtri]

def refinemesh(points, K, numtri) :

    for i in range(len(K)) :
        newpoints = [1/2*(K[i][0]+K[i][1]), 1/2*(K[i][0]+K[i][2]), 1/2*(K[i][1]+K[i][2])]
        numtri += 3
        for j in range(3) :
            points.append(newpoints[j])

    auxtpl = list(set([tuple(x) for x in points])) # remove duplicates
    points = [list(ele) for ele in auxtpl]
    nppoints = np.array(points)

    tri = Delaunay(points)
    K = nppoints[tri.simplices]

    return [tri, points,nppoints,numtri]


def linear_search(list, x):
    # basic line search algorithm

    for i in range(len(list)):
        if list[i][0] == x[0] and list[i][1] == x[1]:
            return i
    return -1


def hat(K, k):

    coeff = []

    for i in K.simplices[k]:
        # vertices opposite to i
        L = np.array([K.points[j] for j in K.simplices[k] if not j == i])  # (2,2)

        # edge vector
        e = L[1] - L[0]

        # perpendicular vector 
        n = np.array([e[1], -e[0]])   # normal to edge

        # normalization factor (area-related)
        delta = np.dot(L[0] - K.points[i], n)

        # affine coefficients: λ_i(x) = a*x + b*y + c
        aux_coeff = np.append(- 1/delta * n, 1 - np.dot(- 1/delta * n, K.points[i]))

        coeff.append(aux_coeff)

    return coeff

def unitouternormal(K,E) :

    # calculate normal
    edge = E[1]-E[0]
    normal = np.array([edge[1],-edge[0]]) # np.dot(normal,edge) =!= 0
    normal = 1/np.linalg.norm(normal)*normal # normalization


    # check for orientation
    center = (K[0]+K[1]+K[2])/3 # convex compination of vertices (always in interior triangle)
    lengthplus = np.linalg.norm((E[0]+E[1])/2+normal-center)
    lengthminus = np.linalg.norm((E[0]+E[1])/2-normal-center)

    if lengthplus<lengthminus :
        normal = -normal

    return normal

def diam(K) :
    indices = [[1,2],[0,2],[0,1]]
    val_max = 0
    for j in range(3) :
        E = K[indices[j]]
        val = np.linalg.norm(E[1]-E[0])
        if val > val_max :
            val_max = val

    return val_max


# ------------------------------------------------------------------------------------------


class dual_mesh:
    def __init__(self, K, E):
        points_primal = len(K.points)
        
        # Combine primal points + circumcenters
        self.points = np.vstack((np.array(K.points), np.array(K.CC)))
        self.pt_dual_reduced = np.vstack((np.array(K.pt_reduced), np.array(K.CC)))
        self.pt_ident = np.array(K.pt_ident + [len(K.pt_reduced)+i for i in range(K.num)])

        self.el = []
        self.simplices = []
        self.primal_ind = []
        self.primal_area = []

        degenerate = 0
        index_Kdual = 0

        # Loop over edges
        for i in range(E.num):
            i1, j1 = E.indKs[i][0]
            CCi1 = K.CC[i1]

            # if E.is_interior[i]:
            i2, j2 = E.indKs[i][1]
            CCi2 = K.CC[i2]

            midk = 0.5*(K.E.el[i1][j1][0]+K.E.el[i1][j1][1])
            pt = get_pt(CCi2, midk) 

            for k in range(2): # two dual elements per edge (one per node of edge)
                e0 = K.E.el[i1][j1][k]  # edge endpoints

                el = np.array([e0, CCi1, pt])
                i_el = [K.E.simplices[i1][j1][k], points_primal + i1, points_primal + i2]

                # Skip degenerate elements
                if np.linalg.norm(CCi1 - CCi2) < 1e-12 or np.linalg.matrix_rank(el - el[0]) < 2:
                    degenerate += 1
                    continue

                # Orient element consistently
                el, i_el = orientation(el, i_el)

                self.el.append(el)
                self.simplices.append(i_el)
                self.primal_ind.append([i1, i2])


                # Compute area
                area = np.abs(np.cross(el[1]-el[0], el[2]-el[0])) / 2
                self.primal_area.append([area, area]) # same area for both parts

                K.E.dual[i1][j1][k] = index_Kdual
                # dual_area
                K.E.dual[i2][j2][k] = index_Kdual
                index_Kdual += 1


        if degenerate > 0:
            print(f"Warning: The dual mesh contains {degenerate} degenerate elements.")
            self.quality = degenerate
        else:
            self.quality = 0

        self.el = np.array(self.el)
        self.simplices = np.array(self.simplices)
        self.primal_ind = np.array(self.primal_ind)
        self.primal_area = np.array(self.primal_area)
        self.num = len(self.el)

        # Diameters (max edge length per element)
        self.diam = np.max(np.linalg.norm(self.el[:, np.array([[0,1],[0,2],[1,2]])[:,0]] - self.el[:, np.array([[0,1],[0,2],[1,2]])[:,1]], axis=2), axis=1)

        # Volumes / areas
        self.vol = np.array([np.abs(np.cross(el[1]-el[0], el[2]-el[0])) / 2 for el in self.el])
        self.gradients = []

        self.neighbors = np.asarray(compute_edge_neighbors(self.el))

        # Compute face properties

        # Each dual element has 3 edges
        max_faces = 3*self.num
        aux_el = np.zeros((max_faces,2,2))
        aux_ind = -np.ones((max_faces,2), dtype=int)

        face_dict = {}
        indexF = 0
        indices = [[1,2],[0,2],[0,1]]

        for i in range(self.num):
            neighbors = self.neighbors[i]
            for j in range(3):
                ni = neighbors[j]
                if ni < 0:
                    print('Warning: periodic boundary face encountered')
                face_key = tuple(sorted([i, ni]))  # unique key for face

                if face_key in face_dict:
                    idx = face_dict[face_key]
                    aux_ind[idx][1] = i

                else:
                    aux_el[indexF] = self.el[i][indices[j]] 
                    aux_ind[indexF][0] = i
                    face_dict[face_key] = indexF
                    indexF += 1

        self.face_el =  np.asarray(aux_el[:indexF])
        self.face_to_tet = np.asarray(aux_ind[:indexF])

        v0 = self.face_el[:,0]
        v1 = self.face_el[:,1]

        edge_vec = v1 - v0

        # Edge lengths
        edge_length = np.linalg.norm(edge_vec, axis=1)

        face_normals = np.zeros_like(edge_vec)
        face_normals[:,0] = -edge_vec[:,1]
        face_normals[:,1] =  edge_vec[:,0]
        norm_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / norm_len

        face_centers = 0.5 * (v0 + v1)  # midpoint of edge

        # Flip normals to point from first to second element
        tri_centers = np.mean(self.points[self.simplices], axis=1)  # triangle centroids
        plus_centers = tri_centers[self.face_to_tet[:,0]]           # + triangle per edge
        vec = face_centers - plus_centers
        flip = np.sum(face_normals * vec, axis=1) < 0
        face_normals[flip] *= -1
        
        self.face_normals = face_normals
        self.num_faces = len(face_normals)
        self.edge_length = edge_length


def get_pt(x,ref) :
    
    if np.abs(ref[0] - x[0]) > 0.7 :
                        
        if x[0]> 0.9 :

            aux_pt = x
            x = [aux_pt[0]-1,aux_pt[1]]

        elif x[0] < 0.1 :
            aux_pt = x
            x = [aux_pt[0]+1,aux_pt[1]]
        
    if np.abs(ref[1] - x[1]) > 0.7 :
        
        if x[1]> 0.9 :

            aux_pt = x
            x = [aux_pt[0],aux_pt[1]-1]

        elif x[1] < 0.1 :
            aux_pt = x
            x = [aux_pt[0],aux_pt[1]+1]

    return x
 
def orientation(el, i_el):
    X_K = np.array([
        [1, 1, 1],
        [el[0][0], el[1][0], el[2][0]],
        [el[0][1], el[1][1], el[2][1]]
    ])
    
    if np.linalg.det(X_K) < 0:
        # swap two vertices to flip orientation
        aux_el = [el[0], el[2], el[1]]
        aux_i_el = [i_el[0], i_el[2], i_el[1]]
        el = aux_el
        i_el = aux_i_el
    
    return [el, i_el]

def compute_edge_neighbors(el):
# Compute triangle edge neighbors with periodic BCs on unit square.

    N = el.shape[0]

    # Unique node indexing:
    all_vertices = el.reshape(-1, 2)
    unique_vertices, inverse = np.unique(all_vertices, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)

    # Define triangle edges:
    edge_ids = np.array([[0, 1], [1, 2], [2, 0]])

    # Extract edges and map to triangles:
    edge_map = {}
    neighbors = -np.ones((N, 3), dtype=int)
    unmatched_edges = []

    for tri_i in range(N):
        for e_i, (a, b) in enumerate(edge_ids):
            nodes = triangles[tri_i, [a, b]]
            key = tuple(sorted(nodes))

            if key in edge_map:
                t2, e2 = edge_map[key]
                neighbors[tri_i, e_i] = t2
                neighbors[t2, e2] = tri_i
            else:
                edge_map[key] = (tri_i, e_i)
                unmatched_edges.append((tri_i, e_i, unique_vertices[list(nodes)]))

    # Handle periodic edges:
    def wrap(p):
        return p - np.floor(p)  # map coordinates into [0,1)

    periodic_hash = {}
    for tri_i, e_i, coords in unmatched_edges:
        wrapped = wrap(coords.copy())
        # Sort vertices for permutation invariance
        wrapped_sorted = np.sort(wrapped, axis=0)
        key = tuple(np.round(wrapped_sorted.flatten(), 12))

        if key not in periodic_hash:
            periodic_hash[key] = []
        periodic_hash[key].append((tri_i, e_i))

    # Match periodic edges:
    for key, owners in periodic_hash.items():
        if len(owners) < 2:
            # Periodic topology broken if not paired
            continue
        for i in range(0, len(owners), 2):
            (t0, e0) = owners[i]
            (t1, e1) = owners[i + 1]
            neighbors[t0, e0] = t1
            neighbors[t1, e1] = t0

    return neighbors



# ------------------------------------------------------------------------------------------

class intersect_mesh :

    def __init__(self,K,E):
            
        points_inter = []
        points_primal = len(K.points)

        for pt in K.points :
            points_inter.append(pt)

        circum_points = len(K.CC)
        for i in range(K.num) :
            points_inter.append(K.CC[i])

        for i in range(E.num) :
            midE = 1/2*(E.el[i][0]+E.el[i][1])
            points_inter.append(midE)

        self.simplices = []

        self.E = edges_inter_mesh(K)

        self.K_primal = []
        self.K_dual = []

        aux_el = []
        areaK = []
        interE = []
        index_Kinter = 0
        for i in range(E.num) :

            [i1,j1] = E.indKs[i][0]
            i1 = int(i1)
            j1 = int(j1)
            midE = 1/2*(E.el[i][0]+E.el[i][1])
            CCi1 = K.CC[i1]
            
   
            [i2,j2] = E.indKs[i][1]
            i2 = int(i2)
            j2 = int(j2)
            CCi2 = K.CC[i2]


            indices_neigh = [1,0] # indices of touching primal edges (through point) / dual elements w.r.t. primal face i1 j1

            indices_face = [[2,1],[2,0],[1,0]] # indices of faces and indices of touching faces (through edge)
            indices_face_neigh = [[1,1],[0,1],[0,0]] # indices of edges of touching faces

            for k in range(2) : # two dual elements per edge / go through endpoints of edge
                
                pt1 = CCi1
                pt2 = CCi2
                ptE = midE

                if np.linalg.norm(K.E.el[i1][j1]-K.E.el[i2][j2]) > 0.1: 

                    midk1 = 1/2*(K.E.el[i1][j1][0] + K.E.el[i1][j1][1])

                    pt1 = get_pt(CCi1,midk1)
                    pt2 = get_pt(CCi2,midk1)
                    ptE = get_pt(midE,midk1)

                # get intersected element
                aux_el.append([K.E.el[i1][j1][k],pt1,ptE]) 
                self.simplices.append([K.E.simplices[i1][j1][k],points_primal+i1,points_primal+circum_points+i])
                
                # get area
                S = np.cross(aux_el[-1][1]-aux_el[-1][0],aux_el[-1][2]-aux_el[-1][0]) # B = K[0], C = K[1], A = K[2]
                areaK.append(np.abs(S)/2) # get area of triangle K

                # get intersected edges
                aux_interE = []

                aux_interE.append([K.E.el[i1][j1][k],ptE]) # is in primal edge E
                self.E.SinE[index_Kinter][0][0] = i1 # set primal element as intersected face lies in interior
                self.E.SinE[index_Kinter][0][1] = i2
                self.E.SinE[index_Kinter][0][2] = K.E.dual[i1][j1][k] # dual element in which inter edge S lies
                
                aux_interE.append([pt1,ptE]) # not in primal face F
                k_neigh = indices_neigh[k]
                self.E.SinE[index_Kinter][1][2] = K.E.dual[i1][j1][k] # dual element on which we work
                self.E.SinE[index_Kinter][1][3] = K.E.dual[i1][j1][k_neigh] # dual element that is neighbor and defined on same primal face
                self.E.SinE[index_Kinter][1][0] = i1  # primal element in which inter face S lies

                aux_interE.append([K.E.el[i1][j1][k],pt1]) # not in primal face F
                j_neigh = indices_face[j1][k]
                k_neigh = indices_face_neigh[j1][k]
                self.E.SinE[index_Kinter][2][2] = K.E.dual[i1][j1][k] # dual element on which we work
                self.E.SinE[index_Kinter][2][3] = K.E.dual[i1][j_neigh][k_neigh] # dual element that is neighbor but defined on a different primal face
                self.E.SinE[index_Kinter][2][0] = i1  # primal element in which inter face S lies

                interE.append(aux_interE) # intersected faces of one intersected element

                # structure intersected element
                K.TinK[i1].append(index_Kinter) #  say primal mesh which intersected mesh elements are in primal element
                self.K_primal.append(i1) # intersected element T is in primal element with index i1
                self.K_dual.append(int(K.E.dual[i1][j1][k])) # intersected element T is in dual element with index written here
                
                index_Kinter += 1

                # do same once again for pair [i2,j2] :
                # get intersected element
                aux_el.append([K.E.el[i1][j1][k],pt2,ptE]) 
                self.simplices.append([K.E.simplices[i2][j2][k],points_primal+i2,points_primal+circum_points+i])
                
                # get area
                S = np.cross(aux_el[-1][1]-aux_el[-1][0],aux_el[-1][2]-aux_el[-1][0]) # B = K[0], C = K[1], A = K[2]
                areaK.append(np.abs(S)/2) # get area of triangle K

                # get intersected edges
                aux_interE = []

                aux_interE.append([K.E.el[i1][j1][k],ptE]) # is in primal edge E
                self.E.SinE[index_Kinter][0][0] = i1 # set primal element as intersected edge lies in interior
                self.E.SinE[index_Kinter][0][1] = i2 # gives primal indices to determine primal edge E
                self.E.SinE[index_Kinter][0][2] = K.E.dual[i2][j2][k] # dual element in which inter edge S lies
                
                aux_interE.append([pt2,ptE]) # not in primal face F
                k_neigh = indices_neigh[k]
                self.E.SinE[index_Kinter][1][2] = K.E.dual[i2][j2][k] # dual element on which we work
                self.E.SinE[index_Kinter][1][3] = K.E.dual[i2][j2][k_neigh] # dual element that is neighbor and defined on same primal face
                self.E.SinE[index_Kinter][1][0] = i2  # primal element in which inter face S lies

                aux_interE.append([K.E.el[i1][j1][k],pt2]) # not in primal face F
                j_neigh = indices_face[j2][k]
                k_neigh = indices_face_neigh[j2][k]
                self.E.SinE[index_Kinter][2][2] = K.E.dual[i2][j2][k] # dual element on which we work
                self.E.SinE[index_Kinter][2][3] = K.E.dual[i2][j_neigh][k_neigh] # dual element that is neighbor and defined on a different primal face
                self.E.SinE[index_Kinter][2][0] = i2  # primal element in which inter face S lies

                interE.append(aux_interE) # intersected faces of one intersected element
        
                # structure intersected element
                K.TinK[i2].append(index_Kinter) #  say primal mesh which intersected mesh elements are in primal element
                self.K_primal.append(i2) # intersected element T is in primal element with index i1
                self.K_dual.append(int(K.E.dual[i2][j2][k])) # intersected element T is in dual element with index written here
                
                index_Kinter += 1


        self.area = np.array(areaK)
        self.el = np.array(aux_el)
        self.num = len(aux_el)

        self.E.el = np.array(interE)

        # get array of faces
        self.neighbors = np.asarray(compute_edge_neighbors(self.el))

        indexE = 0
        edge_dict = {}

        max_edges = 2 * self.num

        aux_el = np.zeros((max_edges, 2, 2))
        aux_indKs = -np.ones((max_edges, 2), dtype=int)
        ident_primal = -np.ones((max_edges, 2), dtype=int)
        ident_dual = -np.ones((max_edges, 2), dtype=int)

        for i in range(self.num):  # loop over elements
            neighbors = self.neighbors[i]  # 3 neighbors per triangle

            for j in range(3):  # loop over edges of the triangle
                ni = neighbors[j]
                edge_key = tuple(sorted([i, ni]))  # unique edge key

                if ni >= 0:  # interior edge
                    if edge_key in edge_dict:
                        idx = edge_dict[edge_key]

                        # swap condition: all x-coords close to 1
                        if aux_el[idx][0,0] > 0.999999 and aux_el[idx][1,0] > 0.999999:
                            aux_el[idx] = self.E.el[i][j]  # swap edge vertices
                            aux_indKs[idx][1] = aux_indKs[idx][0].copy()
                            aux_indKs[idx][0] = i

                            # primal / dual assignment
                            if self.E.SinE[i][j][3] < 0:  # primal edge
                                ident_primal[idx][1] = ident_primal[idx][0].copy()
                                ident_primal[idx][0] = self.E.SinE[i][j][0]
                                ident_dual[idx][0] = self.E.SinE[i][j][2]
                            elif self.E.SinE[i][j][1] < 0 :
                                ident_dual[idx][1] = ident_dual[idx][0].copy()
                                ident_dual[idx][0] = self.E.SinE[i][j][2]
                                ident_primal[idx][0] = self.E.SinE[i][j][0]
                            else :
                                print('Warning: Something is wrong with intersected indexing.')

                        else:
                            aux_indKs[idx][1] = i
                    else:
                        # new interior edge
                        aux_el[indexE] = self.E.el[i][j]
                        aux_indKs[indexE][0] = i

                        if self.E.SinE[i][j][3] < 0 : # on primal face
                            ident_primal[indexE][0] = self.E.SinE[i][j][0]
                            ident_primal[indexE][1] = self.E.SinE[i][j][1]
                            ident_dual[indexE][0] = self.E.SinE[i][j][2]
                        elif self.E.SinE[i][j][1] < 0 : # on dual face
                            ident_dual[indexE][0] = self.E.SinE[i][j][2]
                            ident_dual[indexE][1] = self.E.SinE[i][j][3]
                            ident_primal[indexE][0] = self.E.SinE[i][j][0]
                        else :
                            print('Warning: Something is wrong with intersected indexing.')

                        edge_dict[edge_key] = indexE
                        indexE += 1

                else:  # boundary edge
                    print("Warning: boundary edge encountered")

        self.edge_el = np.asarray(aux_el[:indexE])
        self.edge_to_tri = np.asarray(aux_indKs[:indexE])
        self.edge_ident_primal = np.asarray(ident_primal[:indexE])
        self.edge_ident_dual = np.asarray(ident_dual[:indexE])

        v0 = self.edge_el[:, 0]
        v1 = self.edge_el[:, 1]

        # edge vectors
        edge_vec = v1 - v0

        # edge lengths
        edge_length = np.linalg.norm(edge_vec, axis=1)

        # 2D normals
        edge_normals = np.stack([-edge_vec[:,1], edge_vec[:,0]], axis=1)
        edge_normals /= np.linalg.norm(edge_normals, axis=1, keepdims=True)

        # edge centers
        edge_centers = (v0 + v1) / 2

        # triangle centers
        tri_centers = np.mean(self.el, axis=1)
        plus_centers = tri_centers[self.edge_to_tri[:,0]]

        # flip normals to point from + triangle to − triangle
        flip = np.sum(edge_normals * (edge_centers - plus_centers), axis=1) < 0
        edge_normals[flip] *= -1

        self.edge_normals = edge_normals
        self.num_edges = len(edge_normals)
        self.edge_length = edge_length


class edges_inter_mesh :

            def __init__(self,K):
                self.el = []
                self.area = []
                self.normal = []
                self.SinE = - np.ones([6*K.num,3,4],dtype= int)
                self.is_interior = np.ones([6*K.num,3])


def post_process(K_inter, K):
    # Extract points of each element in primal mesh
    K_el = K.points[K.simplices]
    
    for i in range(K_inter.num_edges):
        # Compute face midpoints in K_inter
        midS = np.mean(K_inter.edge_el[i], axis=0)
        
        # Index of corresponding primal element in K
        k = K_inter.edge_ident_primal[i][0]
        
        # Compute midpoint of the corresponding element in K
        midprim = np.mean(K_el[k], axis=0)

        # if distance between midpoints is significant
        if np.linalg.norm(midS - midprim) > 0.5:
            
            # correct x coordinate if necessary
            if np.abs(midS[0] - midprim[0]) > 0.5:
                if midS[0] > 0.9:
                    K_inter.edge_el[i] -= np.array([[1, 0], [1, 0]])
                elif midS[0] < 0.1:
                    K_inter.edge_el[i] += np.array([[1, 0], [1, 0]])
            
            # correct y coordinate if necessary
            if np.abs(midS[1] - midprim[1]) > 0.5:
                if midS[1] > 0.9:
                    K_inter.edge_el[i] -= np.array([[0, 1], [0, 1]])
                elif midS[1] < 0.1:
                    K_inter.edge_el[i] += np.array([[0, 1], [0, 1]])


def post_process_TinK(K_inter, K):
    # extract points of each element in primal mesh
    K_el = K.points[K.simplices]
    
    for i in range(K.num):
        # midpoint of the primal element
        midprim = np.mean(K_el[i], axis=0)
        
        # loop over elements in K_inter corresponding to this K element
        for j in K.TinK[i]:
            # midpoint of the current element in K_inter
            midint = np.mean(K_inter.el[j], axis=0)
            
            # if distance between midpoints is significant
            if np.linalg.norm(midint - midprim) > 0.5:
                
                # correct x coordinate if necessary
                if np.abs(midint[0] - midprim[0]) > 0.5:
                    if midint[0] > 0.9:
                        K_inter.el[j] -= np.array([[1, 0], [1, 0], [1, 0]])
                    elif midint[0] < 0.1:
                        K_inter.el[j] += np.array([[1, 0], [1, 0], [1, 0]])
                
                # correct y coordinate if necessary
                if np.abs(midint[1] - midprim[1]) > 0.5:
                    if midint[1] > 0.9:
                        K_inter.el[j] -= np.array([[0, 1], [0, 1], [0, 1]])
                    elif midint[1] < 0.1:
                        K_inter.el[j] += np.array([[0, 1], [0, 1], [0, 1]])

    

# ------------------------------------- NUMERICAL SCHEME -----------------------------------

# FV FE scheme
def assemble_FE_matrix(K_dual):
    # see Fig. 3.14. in Bartels2015

    # u = np.zeros(len(points_dual))
    # tu_D = np.zeros(len(points_dual))
    # b = np.zeros(len(points_dual))

    iter = 0
    iter_max = 9*len(K_dual.el)

    I = np.zeros(iter_max)
    J = np.zeros(iter_max)
    X_diff = np.zeros(iter_max)
    X_reac = np.zeros(iter_max)

    m_loc = 1/12*(np.ones([3,3]) + np.array([[1,0,0],[0,1,0],[0,0,1]]))

    gradients = []
    volumes = []
    for i in range(K_dual.num) :

        X_K = np.array([[1, 1, 1],[K_dual.el[i][0][0], K_dual.el[i][1][0], K_dual.el[i][2][0]],[K_dual.el[i][0][1], K_dual.el[i][1][1], K_dual.el[i][2][1]]])
        rhs = np.array([[0,0],[1,0],[0,1]])
        grads_K = np.linalg.solve(X_K,rhs)
        gradients.append(grads_K)
        vol_K = np.linalg.det(X_K)/2
        volumes.append(vol_K)

        if vol_K < 0 :
            print('Warning: FE scheme works with negative volumes.')

        for m in [0,1,2] :
            #b[tri_dual.simplices[i][m]] += 1/3* vol_K * f(mid_K) # rhs
            for n in [0,1,2] :
                I[iter] = K_dual.pt_ident[K_dual.simplices[i][m]] # identify_points
                J[iter] = K_dual.pt_ident[K_dual.simplices[i][n]] # identify_points
                X_diff[iter] = vol_K * np.dot(grads_K[m],grads_K[n])
                X_reac[iter] = vol_K*m_loc[m][n]
                iter += 1

    K_dual.vol = np.array(volumes)
    K_dual.grads = np.array(gradients)

    sparseA_diff = csc_matrix((X_diff[:iter], (I[:iter], J[:iter])), shape=(len(K_dual.pt_dual_reduced),len(K_dual.pt_dual_reduced))) # use reduced len(K_dual.points)
    sparseA_reac = csc_matrix((X_reac[:iter], (I[:iter], J[:iter])), shape=(len(K_dual.pt_dual_reduced),len(K_dual.pt_dual_reduced)))

    sparseA = sparseA_diff + sparseA_reac

    return [sparseA,sparseA_reac]

def assemble_FV_matrix(ht,K,eps) :
    row = []
    col = []
    data = []

    for i in range(K.num) :

        areaK = K.area[i]

        neighbors = K.neighbors[i] 

        indices = [[1,2],[0,2],[0,1]] # possible combinations of vertices to compose an edge, where j = 0,1,2 is not contained , order is impotant!
        
        valK = 1 # reset valK to one

        for j in range(3) : # go through edges

            areaE = K.E.area[i][j]

            if neighbors[j] >= 0 : # i.e. edge E is interior

                dE = K.E.dist[i][j]

                valK += ht*eps/areaK*areaE/dE
                valL = -ht*eps/areaK*areaE/dE

                row.append(i)
                col.append(neighbors[j])
                data.append(valL)

            else : # i.e. edge E is part of the boundary
                valL = 0

        row.append(i) # diagonal entries of matrix
        col.append(i)
        data.append(valK)

    # creating sparse matrix
    sparseA = csc_matrix((data, (row, col)), shape = (K.num, K.num))#.toarray()

    return sparseA


def getc_FE(rhoFV,g,K_dual,sparseA,M) :

    b = np.zeros(len(K_dual.pt_dual_reduced))

    for i in range(K_dual.num) :

        vol_K = K_dual.vol[i]
        mid_K = 1/3*(K_dual.el[i][0]+K_dual.el[i][1]+K_dual.el[i][2])

        [i1,i2] = K_dual.primal_ind[i]

        for m in [0,1,2] : #  "m = 1:(d+1)"
            if i2 > -0.5 :
                b[K_dual.pt_ident[K_dual.simplices[i][m]]] += 1/3* vol_K * (g(mid_K) + 1/2*(rhoFV[i1] + rhoFV[i2])) # using midpoint rule on g and exact integration of rhoFV; 1/4 = 1/(d+1) -> see Bartels Lemma 3.10
            else :
                b[K_dual.pt_ident[K_dual.simplices[i][m]]] += 1/3* vol_K * (g(mid_K) + rhoFV[i1]) # using midpoint rule on g and exact integration of rhoFV; 1/4 = 1/(d+1)

    c, content = sp.sparse.linalg.cg(sparseA,b) # c_h^n(x) = sum_i (c_i*hat_i(x))
    return c, b

def get_gradc(K_dual,c) :
    
    val = []
    
    for i in range(K_dual.num) :
        val.append(np.matmul(np.transpose(K_dual.grads[i]),c[K_dual.pt_ident[K_dual.simplices[i]]]))

    return np.array(val)


def get_c_val(cc, hats_loc, points, dual_index, fp_sorted, order, sub_index_sorted, N_primal, N_sub):

    N_pts = len(points)

    points_aff = np.hstack([points, np.ones((N_pts, 1), dtype=points.dtype)])  # (N_pts, 3)

    #  Evaluate per-intersection values 
    # hats_loc: (N_inter, 3, 3), points_aff.T: (3, N_pts)
    phi = np.einsum('ijk,kp->ijp', hats_loc, points_aff.T)  # (N_inter, 3, N_pts)

    # Select coefficients for intersections
    cc_loc = cc[dual_index]  # (N_inter, 3)

    # Dot product over barycentric axis
    val_sorted = np.einsum('ij,ijp->ip', cc_loc, phi)  # (N_inter, N_pts)

    # Group by primal element and sub-element 
    val_sorted = val_sorted[order]

    output = np.zeros((N_primal, N_sub, N_pts), dtype=val_sorted.dtype)
    output[fp_sorted, sub_index_sorted, :] = val_sorted

    return output


def finitevolumescheme_rho_expl(u_old,ht,K,vv,f,sparseA,M) :
# explicit euler for total advection term

    # Precompute
    areaK = K.area
    inv_areaK = 1 / areaK
    log_u_old = np.zeros_like(u_old)
    small_mask = u_old < 1e-10
    log_u_old[~small_mask] = np.log(u_old[~small_mask])
    
    rhs = ht * inv_areaK * np.array([avrgK(K.el[i], areaK[i], f)[0] for i in range(K.num)]) + u_old

    # Loop over edges (only 3 edges per triangle)
    for j in range(3):
        neighbors_j = K.neighbors[:, j]
        interior_mask = neighbors_j >= 0
        i_idx = np.arange(K.num)[interior_mask]
        nj_idx = neighbors_j[interior_mask].astype(int)
        
        # Skip cells with tiny values
        skip_mask = small_mask[i_idx] | small_mask[nj_idx]
        i_idx = i_idx[~skip_mask]
        nj_idx = nj_idx[~skip_mask]
        
        if len(i_idx) == 0:
            continue
        
        areaE = K.E.area[i_idx, j]
        dual_ids = np.array(K.E.dual[i_idx, j],dtype=int)  # shape [N, 2]
        n_vec = K.E.n[i_idx, j]
        
        # Compute vKE = 0.5 * dot(vv[idual] + vv[jdual], n)
        vKE = 0.5 * np.einsum('ij,ij->i', vv[dual_ids[:,0]] + vv[dual_ids[:,1]], n_vec)
        
        u_diff = u_old[i_idx] - u_old[nj_idx]
        log_diff = log_u_old[i_idx] - log_u_old[nj_idx]
        
        # Where difference is tiny, fallback to cell value
        small_diff = np.abs(u_diff) < 1e-10
        val = np.empty_like(u_diff)
        val[small_diff] = u_old[i_idx[small_diff]]
        val[~small_diff] = u_diff[~small_diff] / log_diff[~small_diff]
        
        rhs[i_idx] -= ht * areaE / areaK[i_idx] * val * vKE

    # creating sparse matrix
    uh, content = sp.sparse.linalg.cg(sparseA,rhs,tol=10**-12)

    return uh, rhs


def avrgK(K,areaK,g) :
# compute average of g:IR²->IR over K, area of K

    wi = [1/6,1/6,1/6] # weights*area_unitK(=1/2) (Exactness 2 -> Order 3)
    xi = np.array([[1/6,2/3],[1/6,1/6],[2/3,1/6]]) # points in unit triangle

    unitK = [[0,0],[1,0],[0,1]]

    val = 0
    for i in range(len(wi)) :
        [trafoxi,M] = tritrafo(unitK,K,xi[i])
        val += wi[i]*g(trafoxi)

    avrg = val*2 # = devided by 1/2
    integral = avrg*areaK

    return [integral,avrg]

def tritrafo(K,L,pt) : # transform K to L
    # input : triangles K and L, point pt from K
    # output : pt transformed to L

    A = np.array([[K[0][0], K[1][0], K[2][0]],  #   | xa1 xa2 xa3 |
                  [K[0][1], K[1][1], K[2][1]],  #A =| ya1 ya2 ya3 |
                  [      1,       1,       1]]) #   |  1   1   1  |
    B = np.array([[L[0][0], L[1][0], L[2][0]],  #   | xa1 xa2 xa3 |
                  [L[0][1], L[1][1], L[2][1]],  #A =| ya1 ya2 ya3 |
                  [      1,       1,       1]]) #   |  1   1   1  |

    invA = np.linalg.solve(A,np.eye(3))
    #invA = np.linalg.inv(A)
    M = np.matmul(B,invA)

    auxx = np.array([pt[0],pt[1],1])

    auxtrafo = np.matmul(M,auxx)
    trafo = [auxtrafo[0],auxtrafo[1]]

    return [trafo,M]


# morley reconstruction
def getinterpolationRHS(K,uh) :
# get the right hand side to compute the morley reconstruction (preservation of diffusive fluxes)

    FKED = [] # nicaise paper notation

    for i in range(K.num) :
        neighbors = K.neighbors[i] 

        auxFKED = []
        for j in range(3) : # go through neighboring triangles K[neighbors[j]] / go through edges of K[i]

            areaE = K.E.area[i][j]

            if neighbors[j] >= 0 : # i.e. edge E is interior

                dE = K.E.dist[i][j]
                auxFKED.append(areaE/dE*(uh[neighbors[j]]-uh[i]))

            else : # i.e. edge E is part of the boundary 
                print('Warning: periodic boundary is broken.')

        FKED.append(auxFKED)

    return FKED 

def getq0(K,uh,string: str) :

    vertex_val = []
    for i in range(len(K.points)) :
        neighborstri = K.adjacent[K.pt_ident[i]] #find_neighbors(i, K.tri)

        #calculate vertex_val
        if string == 'least-squares' : # weighted mean with inner angles as weights

            wi = []
            qi = []
            for k in neighborstri : # go through all touching triangles
                qi.append(uh[k])

            wi = np.array(K.weights[K.pt_ident[i]])
            qi = np.array(qi)

            if np.sum(wi) < 10**-10 :
                vertex_val.append(np.sum(qi)/len(qi))
            else :
                vertex_val.append(np.dot(wi,qi)/np.sum(wi))

        elif string == 'arithmetic-mean'  : # calssic, arithmetic mean

            aux = 0
            for k in neighborstri :
                aux += uh[k]
            vertex_val.append(aux/len(neighborstri))

        else :
            print('Error: invalid string input')

    return np.array(vertex_val)

def getbetaE(K,vertex_val,FKED) :

    betaKE = [] # initilize
    numerKE = []
    denomKE = []

    for i in range(K.num) :

        areaK = K.area[i]
        
        gradq0 = 0
        for j in range(3) :
            pt_i = K.simplices[i][j]
            gradq0 += vertex_val[pt_i]*np.array([K.hat[i][j][0],K.hat[i][j][1]])
        
        betaE = []
        numer = []
        denom = []

        for j in range(3) : # go through neighboring triangles K[neighbors[j]] / go through edges of K[i]

            areaE = K.E.area[i][j]
            normalKE = K.E.n[i][j]
           
            I = areaE*np.dot(gradq0,normalKE) # get integral over gradq0*normal i.e. normal derivative of q0

            III = - (9/5)*(areaE**2)/areaK # = int_E b_E ( grad b_K \cdot n) dS(x)

            aux_numer = (FKED[i][j] - I)
            
            aux_denom = III
            aux_beta = (aux_numer)/(aux_denom)

            betaE.append(aux_beta)
            numer.append(aux_numer)
            denom.append(aux_denom)

        betaKE.append(betaE)
        numerKE.append(numer)
        denomKE.append(denom)

    return [betaKE,numerKE,denomKE]


# ----------------------------------  QUDRATURE FOR POLYNOMIALS and BUBBLE FUNCTION / MORLEY EVALUATION -----------------------------


def tritrafo_quad_tri(L, pt_quad):
# Vectorized affine map from multiple triangles to reference points.

    # base vertex
    v0 = L[:,0,:]                # (M,2)

    # edge vectors
    v1v0 = L[:,1,:] - v0         # (M,2)
    v2v0 = L[:,2,:] - v0         # (M,2)

    # Jacobian 
    J = np.stack([v1v0, v2v0], axis=-1)  # (M,2,2)

    # map reference points to physical triangles
    ptL = (pt_quad[:,0][None,:,None]*v1v0[:,None,:] +
           pt_quad[:,1][None,:,None]*v2v0[:,None,:] +
           v0[:,None,:])  # (M,N,2)

    # inverse transpose Jacobian
    invJt = np.linalg.inv(J.transpose(0,2,1))  # (M,2,2)

    # gradients of barycentric coordinates
    grads12 = invJt.transpose(0,2,1)           # ∇λ1, ∇λ2 (M,2,2)
    grad3 = -np.sum(grads12, axis=1, keepdims=True)  # ∇λ3

    grads = np.concatenate([grads12, grad3], axis=1)  # (M,3,2)

    return ptL, invJt, v0, grads

def tritrafo_quad_edge_and_bary(E_el, pt_ref):
# map reference points on 1D edges to physical edges and compute barycentric coordinates.

    N_edges = E_el.shape[0]
    N_pts = pt_ref.shape[0]

    #  affine map to physical edge 
    v0 = E_el[:,0]  # (N_edges,2)
    v1 = E_el[:,1]  # (N_edges,2)

    e1 = v1 - v0

    xi = pt_ref  # reference points

    pts = v0[:,None,:] + xi[None,:,None]*e1[:,None,:]  # (N_edges, N_pts, 2)

    # barycentric coordinates 
    lambda_edge = np.zeros((N_edges, N_pts, 2))
    lambda_edge[:,:,1] = xi[None,:]       # λ1
    lambda_edge[:,:,0] = 1 - xi[None,:]   # λ0

    assert np.allclose(lambda_edge.sum(axis=2), 1.0)

    return pts, lambda_edge


def barycentric_coords(X, Ainv, v0):
# compute barycentric coordinates of points in multiple triangles.

    # shift points by first vertex of each triangle
    X_shifted = X - v0[:, np.newaxis, :]   # (N_tri, N_points, 2)

    # compute first two barycentric coordinates
    l12 = np.einsum('tpi,tij->tpj', X_shifted, Ainv)  # (N_tri, N_points, 2)

    # third barycentric coordinate
    l3 = 1 - np.sum(l12, axis=2, keepdims=True)       # (N_tri, N_points, 1)

    # stack all three coordinates
    L = np.concatenate([l12, l3], axis=2)             # (N_tri, N_points, 3)

    return L

def barycentric_coords_groups(X, Ainv, v0, inverse_map):
# compute barycentric coordinates for all points in their corresponding triangles.

    N = X.shape[0]
    N_pt = X.shape[1]

    L_all = np.empty((N, N_pt, 3), dtype=X.dtype)

    for m, points_idx in enumerate(inverse_map):
        if len(points_idx) == 0:
            continue

        # shift points
        X_shifted = X[points_idx] - v0[m]                 # (n_local, N_pt, 2)

        # compute first two barycentric coordinates
        l12 = np.einsum('npi,ij->npj', X_shifted, Ainv[m])  # (n_local, N_pt, 2)

        # third barycentric coordinate
        l3 = 1 - np.sum(l12, axis=2, keepdims=True)         # (n_local, N_pt, 1)

        # store
        L_all[points_idx] = np.concatenate([l12, l3], axis=2)  # (n_local, N_pt, 3)

    return L_all



def bubble_K(L):
# Element bubble for triangle: b_K = 27 λ1 λ2 λ3
    return 27.0 * np.prod(L, axis=-1)

def grad_bubble_K(L, grads):
    # detect shapes
    add_dummy_L = L.ndim == 3       # (N_tri, N_pt, 3)
    add_dummy_g = grads.ndim == 3   # (N_tri, 3, 2)

    if add_dummy_L:
        L = L[:, None, :, :]        # (N_tri, 1, N_pt, 3)
    if add_dummy_g:
        grads = grads[:, None, :, :]  # (N_tri, 1, 3, 2)

    # indices of j ≠ i 
    idx = np.array([
        [1,2],   # exclude 0
        [0,2],   # exclude 1
        [0,1]    # exclude 2
    ])

    # extract λj, λk
    L_ex = L[..., idx]              # (..., 3, 2)

    # product of the two remaining lambdas
    prod_except = np.prod(L_ex, axis=-1)   # (..., 3)

    g = grads[:, :, None, :, :]     # (..., 1, 3, 2)

    # contract over barycentric index
    grad = np.einsum('...i,...ij->...j', prod_except, g)

    # remove dummy axes if added
    if add_dummy_L:
        grad = grad[:, 0, :, :]
    if add_dummy_g:
        grad = grad[:, :, :]

    return 27.0 * grad

def grad_bubble_K_edge(K, E, lambda_edge):
# gradient of the element bubble restricted to an edge.

    N_edge, N_pt, _ = lambda_edge.shape

    grad_edge = np.zeros((N_edge, 2, N_pt, 2))  # same structure as your 3D version

    for i in range(2):  # typically two adjacent triangles per edge
        tri_vertices = K.points[
            np.hstack((E.simplices, E.opp[:, i, np.newaxis]))
        ]  # (N_edges, 3, 2)

        # triangle Jacobian 
        v0 = tri_vertices[:, 0, :]
        v1 = tri_vertices[:, 1, :]
        v2 = tri_vertices[:, 2, :]  # opposite vertex

        # jacobian
        J = np.stack([v1 - v0, v2 - v0], axis=2)   # (N_edges, 2, 2)

        grad_lam12 = np.linalg.inv(J).transpose(0, 2, 1)  # (N_edges, 2, 2)

        # gradient of opposite barycentric coordinate λ2
        grad_lambda2 = grad_lam12[:, :, 1]  # (N_edges, 2)

        #  bubble gradient on edge: 
        # b_K = 27 λ0 λ1 λ2, where on edge: λ2 = 0, but gradient survives: grad(b_K) = 27 * (λ0 λ1) * ∇λ2
        prod_lambda = lambda_edge[..., 0] * lambda_edge[..., 1]  # (N_edges, N_pts)

        grad_edge[:, i, ...] = (27.0 * prod_lambda[..., None] * grad_lambda2[:, None, :])

    return grad_edge

def laplace_bubble_K(L, grads):

    # index pairs (i,j) and corresponding remaining k
    i_idx = np.array([0, 0, 1])
    j_idx = np.array([1, 2, 2])
    k_idx = np.array([2, 1, 0])

    #  λk terms 
    lam_k = L[..., k_idx]   # (Nt, Ne, Np, 3)

    #  grad_i · grad_j 
    g_i = grads[:, :, i_idx, :][:, :, None, :, :]  # (Nt,Ne,1,3,2)
    g_j = grads[:, :, j_idx, :][:, :, None, :, :]  # (Nt,Ne,1,3,2)

    # dot products: (Nt, Ne, Np, 3)
    Gdot = np.einsum('...pc,...pc->...p', g_i, g_j)

    # sum contributions
    lap = np.sum(lam_k * Gdot, axis=-1)  # (Nt, Ne, Np)

    return 54.0 * lap

def bubble_E(L, edges):

    edges = np.array(edges)  # (E,2)

    l_i = L[..., edges[:,0]]  # (..., E)
    l_j = L[..., edges[:,1]]

    return 4.0 * l_i * l_j

def bubble_E_edge(L):

    l_i = L[..., 0]
    l_j = L[..., 1]

    return 4.0 * l_i * l_j

def grad_bubble_E(L, grads, edges):

    L = np.asarray(L, dtype=np.float32)
    grads = np.asarray(grads, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int32)

    add_dummy_L = L.ndim == 3
    add_dummy_g = grads.ndim == 3

    if add_dummy_L:
        L = L[:, None, :, :]          # (Nt,1,Np,3)
    if add_dummy_g:
        grads = grads[:, None, :, :]  # (Nt,1,3,2)

    N_tri, nE, N_pt, _ = L.shape
    E = edges.shape[0]

    gradE = np.zeros((N_tri, nE, N_pt, E, 2), dtype=np.float32)

    for e_idx, (i, j) in enumerate(edges):
        li = L[..., i]
        lj = L[..., j]

        gi = grads[..., i, :]   # (Nt,nE,2)
        gj = grads[..., j, :]

        gradE[..., e_idx, :] = (
            lj[..., None] * gi[:, :, None, :] +
            li[..., None] * gj[:, :, None, :]
        )

    if add_dummy_L:
        gradE = gradE[:, 0, :, :, :]

    return 4.0 * gradE

def laplace_bubble_E(L, grads, edges):

    edges = np.asarray(edges)
    E = edges.shape[0]

    # i, λj
    l_i = L[..., edges[:,0]]
    l_j = L[..., edges[:,1]]

    # ∇λi, ∇λj
    grad_i = grads[..., edges[:,0], :][:, :, None, :, :]  # (...,1,E,2)
    grad_j = grads[..., edges[:,1], :][:, :, None, :, :]

    # dot product
    dot_ij = np.einsum('...c,...c->...', grad_i, grad_j)  # (...,1,E)

    # no λ factors remain after Laplacian 
    lapE = 2.0 * dot_ij

    return 4.0 * lapE   # total factor = 8


def get_morley_val(K, pt, bF, bK, vertex_val, betaKF):
# Evaluate Morley finite element values at points for 2D triangles.

    if pt.ndim == 3:
        # PRIMAL POINTS
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:2],1))], axis=-1)  # (N_tri, N_pt, 3)
        hx = np.einsum('tij,tpj->tpi', np.array(K.hat), pt_h)             # (N_tri, N_pt, 3)
        vv = vertex_val[K.simplices]                                       # (N_tri, 3)
        q0 = np.einsum('ti,tpi->tp', vv, hx)                               # (N_tri, N_pt)

        # Edge/element bubble contribution
        b_vec = bF * bK[..., None]                                         # (N_tri, N_pt, 3)
        aux = np.einsum('ti,tpi->tp', betaKF, b_vec)                       # (N_tri, N_pt)

    elif pt.ndim == 4:
        # INTERSECTED POINTS or POINTS ON EDGES
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:3],1))], axis=-1)   # (N_tri, 12, N_pt, 3)
        hx = np.einsum('tij,tkpj->tkpi', np.array(K.hat), pt_h)
        vv = vertex_val[K.simplices]                                       # (N_tri, 3)
        q0 = np.einsum('ti,tkpi->tkp', vv, hx)                             # (N_tri, 12, N_pt)

        b_vec = bF * bK[..., None]                                         # (N_tri, 12, N_pt, 3)
        aux = np.einsum('ti,tkpi->tkp', betaKF, b_vec)                     # (N_tri, 12, N_pt)

    elif pt.ndim == 5:
        # EDGES INTERSECTED
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:4],1))], axis=-1)
        hx = np.einsum('tij,tlkpj->tlkpi', np.array(K.hat), pt_h)
        vv = vertex_val[K.simplices]                                       # (N_tri, 3)
        q0 = np.einsum('ti,tlkpi->tlkp', vv, hx)                           # (N_tri, 12, N_pt)

        b_vec = bF * bK[..., None]                                         # (N_tri, 12, N_pt, 3)
        aux = np.einsum('ti,tlkpi->tlkp', betaKF, b_vec)                   # (N_tri, 12, N_pt)

    return q0 + aux, q0

def get_morley_val_edge(K, X, vertex_val, bE, bK, betaKE, inverse_map):
# Compute q0 and aux for points X using inverse_map per triangle

    N, N_pt, _ = X.shape

    q0_all = np.zeros((N, N_pt), dtype=X.dtype)
    aux_all = np.zeros((N, N_pt), dtype=X.dtype)

    for m, points_idx in enumerate(inverse_map):
        if len(points_idx) == 0:
            continue

        pt = X[points_idx]  # (num_points_in_m, N_pt, 2)

        # homogeneous coordinates 
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:2], 1))], axis=-1)  # (num_points_in_m, N_pt, 3)

        # apply hat matrices
        hx = np.einsum('ij,tpj->tpi', K.hat[m], pt_h)  # (num_points_in_m, N_pt, 3)

        # dot with vertex values 
        vv = vertex_val[K.simplices[m]]  # (3,)
        q0 = np.einsum('i,tpi->tp', vv, hx)  # (num_points_in_m, N_pt)

        # compute aux with bubble functions 
        aux = bK[points_idx, :] * np.einsum('i,npi->np', betaKE[m], bE[points_idx])  # (num_points_in_m, N_pt)

        # assign to full arrays
        q0_all[points_idx] = q0
        aux_all[points_idx] = aux

    return q0_all + aux_all

def get_grad_morley_val_primal(K, bK, gK, bF, gF, vertex_val, betaKF):
# compute Morley gradient contributions for triangles

    #  hat matrices: (N_tri, 2, 3) 
    hat_mat = np.array(K.hat)[:,:,:2].transpose(0,2,1)  # (N_tri, 2, 3)
    vv = vertex_val[K.simplices]                         # (N_tri, 3)
    
    N_tri, N_pt = bK.shape

    # compute grad_q0 = hat_mat @ vertex values 
    # Broadcast vv to (N_tri, N_pt, 3)
    vv_broadcast = np.broadcast_to(vv[:, None, :], (N_tri, N_pt, 3))
    # Compute grad_q0: (N_tri, N_pt, 2)
    grad_q0 = np.einsum('tij,tkj->tki', hat_mat, vv_broadcast)

    # contribution from gK * bF 
    # gF = 0, so only gK term remains
    # gK: (N_tri, N_pt, 2), bF: (N_tri, N_pt, 3)
    b_mat = gK[..., None, :] * bF[..., :, None] + gF[...] * bK[..., None, None]         # (N_tri, N_pt, 3, 2)

    # contract with betaKF: (N_tri, 3)
    val = np.einsum('ti,tpij->tpj', betaKF, b_mat)      # (N_tri, N_pt, 2)

    # combine contributions
    result = grad_q0 + val                                # (N_tri, N_pt, 2)

    return result

def get_grad_morley_val(K, bE, bK, gE, gK, vertex_val, betaKE):
# compute Morley gradient contributions for triangles

    # hat matrices (take first 2 rows for 2D)
    hat_mat = np.array(K.hat)[:, :2, :]  # (N_tri, 2, 3)

    # vertex values per triangle
    vv = vertex_val[K.simplices]  # (N_tri, 3)

    N_tri = bK.shape[0]
    N_sub = bK.shape[1]
    N_pt  = bK.shape[2]

    # broadcast vertex values and hat matrices
    vv_broadcast  = np.broadcast_to(vv[:, None, None, :], (N_tri, N_sub, N_pt, 3))        # (N_tri, N_sub, N_pt, 3)
    hat_broadcast = np.broadcast_to(hat_mat[:, None, None, :, :], (N_tri, N_sub, N_pt, 2, 3))  # (N_tri, N_sub, N_pt, 2, 3)

    # grad_q0 = hat_mat @ vertex values
    grad_q0 = np.einsum('tnlij,tnlj->tnli', hat_broadcast, vv_broadcast)  # (N_tri, N_sub, N_pt, 2)

    # compute contribution from bubble gradients
    term1 = gE * bK[..., None, None]       # (N_tri, N_sub, N_pt, 3, 2)
    term2 = gK[..., None, :] * bE[..., :, None]     # (N_tri, N_sub, N_pt, 3, 2)
    b_mat = term1 + term2                  # (N_tri, N_sub, N_pt, 3, 2)

    # contract with betaKE over last axis of 3 barycentric coordinates
    val = np.einsum('ti,tkpil->tkpl', betaKE, b_mat)  # (N_tri, N_sub, N_pt, 2)

    # total gradient
    return grad_q0 + val

def get_grad_morley_val_edge(K, bE, gK, vertex_val, betaKE, edge_to_tri, loc_edge_ind):
# compute gradient contributions of Morley edge elements

    N_edge, N_pt = bE.shape
    betaKE = np.asarray(betaKE)

    # Hat matrices: (N_tri, 2, 3)
    hat_mat = np.array(K.hat)[:, :, :2]  # 2D

    # Vertex values: (N_tri, 3)
    vv = vertex_val[K.simplices]

    # grad_q0 = hat_mat @ vv -> (N_tri, 2)
    grad_q0 = np.einsum('tij,ti->tj', hat_mat, vv)  # (N_tri, 2) CORRECT?!

    # Output array
    val = np.zeros((N_edge, 2, N_pt, 2), dtype=np.float32)

    for i in range(2):
        # Map edge to triangle and local index
        tri_mask = edge_to_tri[:, i]
        loc_mask = loc_edge_ind[:, i]

        # Bubble contribution: gK * bE
        b_mat = gK[:, i, ...] * bE[..., None]  # (N_tri, N_pt, 3, 2)

        # Contract with betaKE along barycentric axis
        bubble_contr = b_mat * betaKE[tri_mask, loc_mask][:, None, None]  # (N_edge, N_pt, 3, 2)

        # Sum grad_q0 + bubble_contr
        val[:, i, :, :] = grad_q0[tri_mask, None, :] + bubble_contr  # pick first barycentric axis

    return val

def get_lap_morley_val(bE, bK, gE, gK, lapE, lapK, betaKE):
# Compute Laplacian contribution for Morley triangles (2D).


    # Laplacian of edges times interior bubble
    term1 = lapE * bK[..., None]  # (..., 3)

    # Laplacian of interior bubble times edge bubble
    term2 = lapK[..., None] * bE  # (..., 3)

    # 2 * dot(grad_edge, grad_interior)
    term3 = 2 * np.einsum('tnpij,tnpij->tnpi', gE, gK[..., None, :])  # (..., 3)

    # Combine contributions
    b_vec = term1 + term2 + term3  # (..., 3)

    # Contract with betaKE
    val = np.einsum('ti,tnpi->tnp', betaKE, b_vec)  # (N_tri, N_sub, N_pt)

    return val


# -------------------------------------- A POSTERIORI ERROR ESTIMATOR --------------------------------------------

def get_conv_terms(K_inter, vv, rho, morley_inter_edges):
# compute convection terms for Morley elements along edges

    num_edges = K_inter.num_edges
    grad_c_aux = vv[K_inter.edge_ident_dual[:, 0]]  # (num_edges,2)
    dual_id = K_inter.edge_ident_dual[:, 1]          # (num_edges,)
    primal0 = K_inter.edge_ident_primal[:, 0]
    primal1 = K_inter.edge_ident_primal[:, 1]
    normals = K_inter.edge_normals                     # (num_edges,2)
    morley = morley_inter_edges                        # (num_edges, N_pt)

    aux_rho_safe = np.maximum(rho, 1e-300)

    mask_dual_neg = dual_id < 0
    mask_dual_pos = ~mask_dual_neg

    val = np.zeros([num_edges, morley.shape[1]], dtype=grad_c_aux.dtype)

    # --------------------------
    # Case 1: dual_id < 0
    # --------------------------
    if np.any(mask_dual_neg):
        i1 = primal0[mask_dual_neg]
        i2 = primal1[mask_dual_neg]
        rho1 = rho[i1]
        rho2 = rho[i2]
        rho1_safe = aux_rho_safe[i1]
        rho2_safe = aux_rho_safe[i2]

        F = np.zeros_like(rho1)
        mask_zero = (rho1 < 1e-12) | (rho2 < 1e-12)
        mask_continuous = (~mask_zero) & (np.abs(rho1 - rho2) < 1e-12)
        mask_logmean = (~mask_zero) & (~mask_continuous)

        F[mask_continuous] = rho1[mask_continuous]
        F[mask_logmean] = (rho1[mask_logmean] - rho2[mask_logmean]) / \
                          (np.log(rho1_safe[mask_logmean]) - np.log(rho2_safe[mask_logmean]))

        grad_neg = grad_c_aux[mask_dual_neg]
        normals_neg = normals[mask_dual_neg]
        morley_neg = morley[mask_dual_neg]

        val[mask_dual_neg] = np.einsum('ij,ij->i', grad_neg, normals_neg)[:, np.newaxis] * \
                             (morley_neg - F[:, np.newaxis])

    # --------------------------
    # Case 2: dual_id >= 0
    # --------------------------
    if np.any(mask_dual_pos):
        grad_pos = grad_c_aux[mask_dual_pos]
        dual_other = dual_id[mask_dual_pos].astype(int)
        grad_other = vv[dual_other]
        normals_pos = normals[mask_dual_pos]
        morley_pos = morley[mask_dual_pos]

        jump_c = np.einsum('ij,ij->i', grad_pos - grad_other, normals_pos)
        val[mask_dual_pos] = jump_c[:, np.newaxis] * morley_pos

    return val


def assemble_FE_matrix_q(K):
    # see Fig. 3.14. in Bartels2015

    # u = np.zeros(len(points_dual))
    # tu_D = np.zeros(len(points_dual))
    # b = np.zeros(len(points_dual))

    iter = 0
    iter_max = 9*len(K.el)

    I = np.zeros(iter_max)
    J = np.zeros(iter_max)
    X_diff = np.zeros(iter_max)
    X_reac = np.zeros(iter_max)

    m_loc = 1/12*(np.ones([3,3]) + np.array([[1,0,0],[0,1,0],[0,0,1]]))

    gradients = []
    volumes = []
    for i in range(K.num) :

        X_K = np.array([[1, 1, 1],[K.el[i][0][0], K.el[i][1][0], K.el[i][2][0]],[K.el[i][0][1], K.el[i][1][1], K.el[i][2][1]]])
        rhs = np.array([[0,0],[1,0],[0,1]])
        grads_K = np.linalg.solve(X_K,rhs)
        gradients.append(grads_K)
        vol_K = np.linalg.det(X_K)/2
        volumes.append(vol_K)
        if vol_K < 0 :
            print('Warning: FE scheme works with negative volumes.')
            # print(vol_K)

        for m in [0,1,2] :
            #b[tri_dual.simplices[i][m]] += 1/3* vol_K * f(mid_K) # rhs
            for n in [0,1,2] :
                I[iter] = K.pt_ident[K.simplices[i][m]] # identify_points
                J[iter] = K.pt_ident[K.simplices[i][n]] # identify_points
                X_diff[iter] = vol_K * np.dot(grads_K[m],grads_K[n])
                X_reac[iter] = vol_K * m_loc[m][n]
                iter += 1

    sparseA_diff = csc_matrix((X_diff[:iter], (I[:iter], J[:iter])), shape=(len(K.pt_reduced),len(K.pt_reduced)))
    sparseA_reac = csc_matrix((X_reac[:iter], (I[:iter], J[:iter])), shape=(len(K.pt_reduced),len(K.pt_reduced)))

    sparseA = sparseA_diff + sparseA_reac

    return [sparseA,gradients,sparseA_reac]

def getq_FE(K,sparseA,grad_morley) :

    b = np.zeros(len(K.pt_reduced))

    for i in range(K.num) :
        vol_K = K.area[i]

        for m in [0,1,2] : #  "m = 1:(d+1)"
            
            b[K.pt_ident[K.simplices[i][m]]] += 1/3*vol_K*grad_morley[i] # exact quadrature
           
    q, content = sp.sparse.linalg.cg(sparseA,b)

    return q,b

def get_gradq(K,grads,q) :
    
    val = []
    for i in range(K.num) :
        val.append(np.matmul(np.transpose(grads[i]),q[np.asarray(K.pt_ident)[K.simplices[i]]]))

    return np.array(val)


# --------------------------------------- COMPARE STABILITY ----------------------------------------------

def newton_method1D(f,dx_f,x0,tol,maxiter) :
# determine root of f:\R \to \R

    xnew = x0
    for i in range(maxiter) :
        xold = xnew

        xnew = xold - f(xold)/dx_f(xold)

        if np.abs(xold-xnew) < tol :
            break

    print('Newton steps:',i)
    if i == maxiter-1 :
        print('Warning: Newton-method reached maxiter.')
        return -1

    return xnew
