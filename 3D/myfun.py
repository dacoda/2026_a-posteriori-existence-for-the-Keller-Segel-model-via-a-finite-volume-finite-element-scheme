import numpy as np
import itertools
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.spatial import Delaunay, Voronoi
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ------------------------------ MESH CLASSES ----------------------------------------------- 

class primal_mesh :
    def __init__(self, fineness):
    
        [nppoints,pt_reduced,pt_ident,simplices,neighbors] = getmesh(fineness)

        self.pt_reduced = pt_reduced
        self.pt_ident = pt_ident

        self.points = np.asarray(nppoints)
        
        order = np.argsort(simplices,axis=1)

        rows = np.arange(simplices.shape[0])[:, None]  # shape (M,1)
        self.simplices = simplices[rows, order]       # shape (M,4)

        # apply same permutation to neighbors
        self.neighbors = neighbors[rows, order]

        K_el = self.points[self.simplices]
        self.num = len(self.simplices)  

        auxCC = np.zeros((self.num, 3))
        outsideCC = 0
        for i in range(self.num) :
            CCi = Voronoi(K_el[i]).vertices[0]
            auxCC[i] = CCi
            
            if not pointInside(K_el[i],CCi) : # check well-centered property
                outsideCC += 1   

        self.CC = np.array(auxCC)

        # tetrahedron volumes 
        A = K_el[:, 0]
        B = K_el[:, 1]
        C = K_el[:, 2]
        D = K_el[:, 3]

        vol = np.abs(np.einsum("ij,ij->i",np.cross(B - A, C - A),D - A)) / 6.0
        self.area = vol

        # diameter
        edge_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

        edges = K_el[:, edge_pairs[:,0]] - K_el[:, edge_pairs[:,1]]
        edge_lengths = np.linalg.norm(edges, axis=2)

        self.diam = edge_lengths.max(axis=1)

        # build adjacency 
        pt_ident = np.asarray(self.pt_ident)
        adjacent = [[] for _ in range(len(self.pt_reduced))]

        # precompute simplex membership map
        vertex_to_elements = [[] for _ in range(len(self.points))]

        for elem_id, tet in enumerate(self.simplices):
            for v in tet:
                vertex_to_elements[v].append(elem_id)

        # assign reduced-point adjacency
        for reduced_id in range(len(self.pt_reduced)):
            original_vertices = np.where(pt_ident == reduced_id)[0]
            neigh = set()
            for v in original_vertices:
                neigh.update(vertex_to_elements[v])
            adjacent[reduced_id] = list(neigh)

        self.adjacent = adjacent

        # compute hat basis coefficients 
        self.hat = [hat(self, i) for i in range(self.num)]

        self.TinK = [[] for i in range(self.num)]

        if outsideCC > 0 :
            print("Warning: The circumcenter of the primal mesh lie outside of the respective tetrahedron %i times." %outsideCC)
        else :
            print("All circumcenters of the primal mesh lie inside the respective element.")
        self.quality = outsideCC

        
    def init_edges(self) :
        self.F = faces_of_element(self)

class faces_of_element :
    def __init__(self, K):

        N = K.num
        K_el = K.points[K.simplices]

        indices = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        edge_pairs = np.array([[0,1],[0,2],[1,2]])

        # Preallocate arrays
        CC = np.zeros((N,4,3))
        dF = np.zeros((N,4))
        areaF = np.zeros((N,4))
        diam = np.zeros((N,4))
        nKF = np.zeros((N,4,3))
        auxF = np.zeros((N,4,3,3))
        auxFindex = np.zeros((N,4,3), dtype=int)

        almost_coin = 0
        
        for i in range(N):
            neighbors = K.neighbors[i]
            tet = K_el[i]

            # Compute faces (4 per tetrahedron)
            faces = tet[indices]  # shape (4,3,3)
            auxF[i] = faces
            auxFindex[i] = K.simplices[i][indices]

            # Face computations
            for j in range(4):
                F = faces[j]
                # area
                S = np.cross(F[1]-F[0], F[2]-F[0])
                areaF[i,j] = np.linalg.norm(S)/2

                # unit outer normal
                nKF[i,j] = unitouternormal(tet, F)

                # circumcenter of face (2D)
                CC[i,j] = circumcenter2D(F)

                # diameter
                edge_vecs = F[edge_pairs[:,0]] - F[edge_pairs[:,1]]
                diam[i,j] = np.linalg.norm(edge_vecs, axis=1).max()

                # distance to neighbor circumcenter (periodic)
                if neighbors[j] >= 0:
                    val = np.linalg.norm(K.CC[i]-K.CC[neighbors[j]])

                    # apply periodic shifts
                    for k in range(3):
                        if val > 0.5:
                            shift = np.zeros(3)
                            shift[k] = 1 if K.CC[i,k] < 0.1 else -1
                            val = np.linalg.norm(K.CC[i]+shift-K.CC[neighbors[j]])
                    dF[i,j] = val

                    if val < 1e-10:
                        almost_coin += 1
                else:
                    dF[i,j] = np.inf

        # Warning / quality metric
        if almost_coin > 0:
            print(f"Warning: Neighboring primal circumcenters almost coincide {almost_coin//2} times.")
            self.quality = almost_coin//2
        else:
            self.quality = 0

        # Save arrays
        self.CC = CC
        self.el = auxF
        self.num = N
        self.simplices = auxFindex
        self.area = areaF
        self.dist = dF
        self.n = nKF
        self.diam = diam

        # Dual placeholders
        self.dual = -np.ones((N,4,3))
        self.dual_area = -np.ones((N,4,3))
   
class faces :

    def __init__(self,K):
        
        N = K.num
        indices = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        max_faces = 4 * N

        # Preallocate arrays
        aux_el = np.zeros((max_faces, 3, 3))
        auxCC = np.zeros((max_faces, 3))
        simp = np.zeros((max_faces, 3), dtype=int)
        opp = np.zeros((max_faces, 2), dtype=int)
        aux_indKs = -np.ones((max_faces, 2, 2), dtype=int)
        aux_interior = np.zeros(max_faces, dtype=int)

        indexF = 0
        # Use dictionary for fast face lookup
        face_dict = {}
        
        for i in range(N):
            neighbors = K.neighbors[i]
            for j in range(4):
                ni = neighbors[j]
                face_key = tuple(sorted([i, ni]))  # unique key for face

                if ni >= 0:  # interior face
                    if face_key in face_dict:
                        idx = face_dict[face_key]
                        # Replace face if on right side
                        if aux_el[idx][0,0] > 0.999999 and aux_el[idx][1,0] > 0.999999 and aux_el[idx][2,0] > 0.999999:
                            # Swap previous with current
                            aux_el[idx] = K.F.el[i][j]
                            aux_indKs[idx][1] = aux_indKs[idx][0].copy()
                            aux_indKs[idx][0] = [i, j]
                            opp[idx][1] = K.simplices[i][j]
                        else:
                            aux_indKs[idx][1] = [i, j]
                            opp[idx][1] = K.simplices[i][j]
                        aux_interior[idx] = 1
                    else:
                        # New interior face
                        aux_el[indexF] = K.F.el[i][j]
                        auxCC[indexF] = K.F.CC[i][j]
                        simp[indexF] = K.simplices[i][indices[j]]
                        opp[indexF][0] = K.simplices[i][j]
                        aux_indKs[indexF][0] = [i, j]
                        aux_interior[indexF] = 1
                        face_dict[face_key] = indexF
                        indexF += 1
                else:  # boundary face (periodic)
                    aux_el[indexF] = K.F.el[i][j]
                    auxCC[indexF] = K.F.CC[i][j]
                    aux_indKs[indexF][0] = [i, j]
                    aux_interior[indexF] = 0
                    face_dict[face_key] = indexF
                    indexF += 1
                    print("Warning: periodic boundary face encountered")

        # Trim arrays to actual size
        self.num = indexF
        self.el = aux_el[:indexF]
        self.CC = auxCC[:indexF]
        self.simplices = simp[:indexF]
        # self.simplices = np.sort(simp[:indexF],axis=1)
        self.opp = opp[:indexF]
        self.indKs = aux_indKs[:indexF]
        self.is_interior = aux_interior[:indexF]


def getmesh(fineness) :

    a = 3/4
    b = 3/4

    I, J, K = np.meshgrid(
        np.arange(-5, 5),
        np.arange(-5, 5),
        np.arange(-5, 5),
        indexing="ij"
    )

    pts = np.stack([
        (I + J/2 + K/2) / 3,
        (a * J) / 3,
        (b * K) / 3
    ], axis=-1).reshape(-1, 3)

    mask = (
        (pts[:,0] >= 0) & (pts[:,0] <= 1.2) &
        (pts[:,1] >= 0) & (pts[:,1] <= 1.0) &
        (pts[:,2] >= 0) & (pts[:,2] <= 1.0)
    )

    points = pts[mask]
 
    const = 1/(fineness+1)
    points_all = points * const
    points_less = points_all[points_all[:,0] < 1]

    shifts = generate_translations(fineness) * const
    # Decide which point set to use
    base = points_all if fineness == 0 else points_less
    # Broadcast translation in one shot

    points = (base[:, None, :] + shifts[None, :, :]).reshape(-1, 3)

    # Deduplicate points
    pts = np.round(points, 10)
    pts_unique, inv_map = np.unique(pts, axis=0, return_inverse=True)
    points = pts_unique.tolist()

    # Delaunay
    tri = Delaunay(points)
    simplices = tri.simplices.copy()
    neighbors = tri.neighbors.copy()
    nppoints = np.asarray(points)
    el = nppoints[simplices]

    # remove tets that are on boundary where actually there is non due to periodicity
    # Precompute masks
    keep = np.ones(len(el), dtype=bool)
    CC = []
    i2j = np.full(len(el), -1)
    edit_boundary = set()
    edit_neighbors = set()
  

    #  Classify tetrahedra 
    for i in range(len(el)): # go through all elements including elements that we need to remove
        tet = el[i]

        if check_coplanar(tet):
            keep[i] = False
            edit_boundary.add(i)
            continue

        # Remove tets fully outside right x-boundary
        if np.all(tet[:, 0] > 0.999999):
            keep[i] = False
            edit_neighbors.add(i)
            continue

        CCj = Voronoi(tet).vertices[0]

        if pointInside(tet, CCj):
            CC.append(CCj)
        else:
            keep[i] = False
            edit_neighbors.add(i)

    #  Apply mask once 
    simplices = simplices[keep]
    el = el[keep]
    neighbors = neighbors[keep]

    #  Build old to new index map 
    old_ids = np.where(keep)[0]
    for new_id, old_id in enumerate(old_ids):
        i2j[old_id] = new_id
        

    # modify neighbors such that the boundary truely connects periodically
    indices = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]

    for j in range(len(simplices)) : 

        for k in range(4) :

            if neighbors[j][k] in edit_boundary : # does not happen as all those elements are removed
                print('boundary')
                neighbors[j][k] = -1

            if neighbors[j][k] in edit_neighbors :
                F = el[j][indices[k]] # face between element and repspective neighbor
                midF = 1/3*(F[0]+F[1]+F[2])
                if midF[0] > 1/2 : #1/2**fineness : # right side
                    pt = midF+10**-12*np.array([1,0,0])-np.array([1,0,0])
                    l = tri.find_simplex(pt)
                    neighbors[j][k] = i2j[l]
                elif midF[0] < 1/2 : #- 1/2*1/2**fineness  < 0 : # left side
                    pt = midF-10**-12*np.array([1,0,0])+np.array([1,0,0])
                    l = tri.find_simplex(pt)
                    neighbors[j][k] = i2j[l]
                
            else :
                if neighbors[j][k] == -1 : # boundary stays boundary
                
                    F = el[j][indices[k]] 
                    midF = 1/3*(F[0]+F[1]+F[2])

                    if midF[1] > 1 - 1e-6 :
                        pt = midF+10**-12*np.array([0,1,0])-np.array([0,1,0])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = i2j[l]
                       
                    elif midF[1] < 1e-6 :
                        pt = midF-10**-12*np.array([0,1,0])+np.array([0,1,0])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = i2j[l]
                    
                    elif midF[2] > 1 - 1e-6 :
                        pt = midF+10**-12*np.array([0,0,1])-np.array([0,0,1])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = i2j[l]

                    elif midF[2] < 1e-6 :
                        pt = midF-10**-12*np.array([0,0,1])+np.array([0,0,1])
                        l = tri.find_simplex(pt)
                        neighbors[j][k] = i2j[l]
                        
                    else :
                        print('Warning: Neighbors not correctly assigned.')

                else :

                    neighbors[j][k] = i2j[neighbors[j][k]]


    print('num. tet: ',len(el))
    print('num. pts: ',len(points))

    pt_ident = []
    pt_reduced = []
    for i in range(len(points)) :
        pt = points[i]

        if pt[0]> 1 - 1e-6 :
            
            pt2 = pt
            pt = [pt2[0]-1,pt2[1],pt2[2]]
        
        if pt[1]> 1 - 1e-6 :
            
            pt2 = pt
            pt = [pt2[0],pt2[1]-1,pt2[2]]

        if pt[2]> 1 - 1e-6 :
            
            pt2 = pt
            pt = [pt2[0],pt2[1],pt2[2]-1]

        j = linear_search(pt_reduced, pt)
        if j > -0.5 :
            pt_ident.append(j)
        else :
            pt_reduced.append(pt)
 
    return [points,pt_reduced,pt_ident,simplices,neighbors]

def generate_translations(n):
    grid = np.stack(np.meshgrid(
        np.arange(n+1),
        np.arange(n+1),
        np.arange(n+1),
        indexing="ij"
    ), axis=-1).reshape(-1, 3)
    return grid

def check_coplanar(el): 

    [A,B,C,D] = el

    [x1,y1,z1] = A
    [x2,y2,z2] = B
    [x3,y3,z3] = C
    [x4,y4,z4] = D
     
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
     
    # equation of plane is: a*x + b*y + c*z = 0 #
     
    # checking if the 4th point satisfies the above equation
    if np.abs(a * x4 + b * y4 + c * z4 + d) < 10**-8:
        return 1
   
    else:
        return 0
    
def pointInside(el,p):
  # Find the transform matrix from orthogonal to tetrahedron system
  v1 = el[0]
  v2 = el[1]
  v3 = el[2]
  v4 = el[3]

  M1=tetraCoord(v1,v2,v3,v4)
  # apply the transform to P (v1 is origin)
  newp = M1.dot(p-v1)
  # perform test
  return (np.all(newp>=0) and np.all(newp <=1) and np.sum(newp)<=1)

def tetraCoord(A,B,C,D):

  v1 = B-A ; v2 = C-A ; v3 = D-A
  mat = np.array((v1,v2,v3)).T
  # mat is 3x3 here
  M1 = np.linalg.inv(mat)
  return(M1)

def linear_search(list, x):
    # basic line search algorithm

    for i in range(len(list)):
        if np.abs(list[i][0] - x[0]) < 10**-8 and np.abs(list[i][1] - x[1]) < 10**-8 and np.abs(list[i][2] - x[2]) < 10**-8:
            return i
    return -1


def hat(K,k) :
    # 3d hat function on tet
   
    coeff = []
    for i in K.simplices[k] :

        L = np.array([K.points[j] for j in K.simplices[k] if not j==i])

        n = np.cross(L[1] - L[0],L[2] - L[0])
        delta = np.dot(L[0] - K.points[i], np.cross(L[1]- K.points[i],L[2]- K.points[i]))

        aux_coeff = np.append(- 1/delta * n, 1 - np.dot(- 1/delta * n , K.points[i]))

        coeff.append(aux_coeff) # x-y-z

    return coeff


def unitouternormal(K,F) :

    # calculate normal
    normal = np.cross(F[1]-F[0],F[2]-F[0]) # np.dot(normal,edge) =!= 0
    normal = 1/np.linalg.norm(normal)*normal # normalization

    # check for orientation
    center = (K[0]+K[1]+K[2]+K[3])/4 # convex compination of vertices (always in interior triangle)
    lengthplus = np.linalg.norm((F[0]+F[1]+F[2])/3+normal-center)
    lengthminus = np.linalg.norm((F[0]+F[1]+F[2])/3-normal-center)

    if lengthplus<lengthminus : # flip vector
        normal = -normal

    return normal

def circumcenter2D(el) :
    A = el[0]
    B = el[1]
    C = el[2]

    aux1 = - (A[1]-B[1])*(C[2]-A[2]) + (C[1]-A[1])*(A[2]-B[2])
    aux2 = (A[0]-B[0])*(C[2]-A[2]) - (C[0]-A[0])*(A[2]-B[2])
    aux3 = - (A[0]-B[0])*(C[1]-A[1]) + (C[0]-A[0])*(A[1]-B[1])

    M = [[A[0]-B[0], A[1]-B[1], A[2]-B[2]], 
         [B[0]-C[0], B[1]-C[1], B[2]-C[2]], 
         [     aux1,      aux2,      aux3]]
    
    auxb = A[0]*B[1]*C[2] + B[0]*C[1]*A[2] + C[0]*A[1]*B[2] - A[2]*B[1]*C[0] - B[2]*C[1]*A[0] - C[2]*A[1]*B[0]

    b = [(A[0]**2-B[0]**2)/2+(A[1]**2-B[1]**2)/2+(A[2]**2-B[2]**2)/2,(B[0]**2-C[0]**2)/2+(B[1]**2-C[1]**2)/2+(B[2]**2-C[2]**2)/2,auxb]
    
    CC = np.linalg.solve(M,b)

    return CC


# ------------------------------------------------------------------------------------------

class dual_mesh :
    def __init__(self,K,F) :

        # Dual points = primal points + element circumcenters
        self.points = np.vstack((np.array(K.points), np.array(K.CC)))
        self.pt_dual_reduced = np.vstack((np.array(K.pt_reduced), np.array(K.CC)))
        self.pt_ident = np.array(K.pt_ident + [len(K.pt_reduced)+i for i in range(K.num)])

        self.simplices = []
        self.el = []
        self.primal_ind = []
        self.primal_area = []

        degenerate = 0
        index_Kdual = 0

        # Precompute midpoint indices for edges of a face
        edge_indices = np.array([[1,2],[0,2],[0,1]])
        for i in range(F.num):
            i1,j1 = F.indKs[i][0]
            i1,j1 = int(i1), int(j1)
            CCi1 = K.CC[i1]
            CCF = K.F.CC[i1][j1]
                       
            if F.is_interior[i] : # either 1 (interior face) or 0 (boundary face), always 1 for periodic boundary

                i2,j2 = F.indKs[i][1]
                i2,j2 = int(i2), int(j2)
                CCi2 = K.CC[i2]     

                for k in range(3):  # three dual elements per face (one per edge)
                    e0,e1 = K.F.el[i1][j1][edge_indices[k]]
                    midk = 0.5*(e0+e1)
                    pt = get_pt(CCi2, midk) 

                    el = np.array([e0,e1,CCi1,pt])
                    i_el = [K.F.simplices[i1][j1][edge_indices[k][0]],
                            K.F.simplices[i1][j1][edge_indices[k][1]],
                            len(K.points)+i1, len(K.points)+i2]

                    if check_coplanar(el) or np.linalg.norm(CCi1-CCi2)<1e-8:
                        degenerate += 1
                        continue

                    el, i_el = orientation(el,i_el)
     
                    self.el.append(el)
                    self.simplices.append(i_el)

                    # Compute area and volumes
                    S = np.cross(el[1]-el[0], CCF-el[0])
                    area1 = np.linalg.norm(S)/2
                    K.F.dual[i1][j1][k] = index_Kdual
                    K.F.dual_area[i1][j1][k] = area1
                    K.F.dual[i2][j2][k] = index_Kdual
                    K.F.dual_area[i2][j2][k] = area1

                    vol1 = np.abs(np.dot(np.cross(el[1]-el[0], el[2]-el[0]), CCF-el[0]))/6
                    vol2 = np.abs(np.dot(np.cross(el[1]-el[0], el[3]-el[0]), CCF-el[0]))/6

                    self.primal_ind.append([i1,i2])
                    self.primal_area.append([vol1,vol2])

                    index_Kdual += 1
            else:
                print("Warning: periodic boundary dual mesh fails!")
        
     
        if degenerate > 0 :
            print("Warning: The dual mesh contains %i degenerate element(s)." %degenerate)
            self.quality = degenerate
        else :
            print("The dual mesh DOES NOT contain degenerate elements.")
            self.quality = 0

        self.num = len(self.el)
        self.el = np.asarray(self.el)
        self.simplices = np.asarray(self.simplices)
        self.primal_ind = np.asarray(self.primal_ind)
        self.primal_area = np.asarray(self.primal_area)

        # Compute diameters
        self.diam = np.linalg.norm(self.el[:,0]-self.el[:,1], axis=1)
        self.gradients = []
        self.vol = []

        self.neighbors = np.asarray(compute_face_neighbors(self.el))

        indexF = 0
        # Use dictionary for fast face lookup
        face_dict = {}
        aux_el = self.el

        indices = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
                ])
        
        max_faces = 2*self.num

        # Preallocate arrays
        aux_el = np.zeros((max_faces, 3, 3))
        aux_indKs = -np.ones((max_faces, 2), dtype=int)

        for i in range(self.num):
            neighbors = self.neighbors[i]
            for j in range(4):
                ni = neighbors[j]
                face_key = tuple(sorted([i, ni]))  # unique key for face

                if ni >= 0:  # interior face
                    if face_key in face_dict:
                        idx = face_dict[face_key]
                        # Replace face if on right side
                        if aux_el[idx][0,0] > 0.999999 and aux_el[idx][1,0] > 0.999999 and aux_el[idx][2,0] > 0.999999:
                            # Swap previous with current
                            aux_el[idx] =  self.el[i][indices[j]]
                            aux_indKs[idx][1] = aux_indKs[idx][0].copy()
                            aux_indKs[idx][0] = i
                        else:
                            aux_indKs[idx][1] = i
                    else:
                        # New interior face
                        aux_el[indexF] = self.el[i][indices[j]]
                        aux_indKs[indexF][0] = i
                        face_dict[face_key] = indexF
                        indexF += 1
                else:  # boundary face (periodic)
                    aux_el[indexF] =  self.el[i][indices[j]]
                    aux_indKs[indexF][0] = i
                    face_dict[face_key] = indexF
                    indexF += 1
                    print("Warning: periodic boundary face encountered")

        self.face_el = np.asarray(aux_el[:indexF])
        self.face_to_tet = np.asarray(aux_indKs[:indexF])
  
        v0 = self.face_el[:,0]
        v1 = self.face_el[:,1]
        v2 = self.face_el[:,2]

        cross = np.cross(v1 - v0, v2 - v0)   # shape (N_faces, 3)

        # face areas: 1/2 * |cross|
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Unit normals
        face_normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)

        # Face centers
        face_centers = (v0 + v1 + v2)/3

        # Tet centers of + tetrahedra
        tet_centers = np.mean(self.points[self.simplices], axis=1)
        plus_centers = tet_centers[self.face_to_tet[:,0]]

        # Flip normals to point from + tet to − tet
        flip = np.sum(face_normals * (face_centers - plus_centers), axis=1) < 0
        face_normals[flip] *= -1

        # Edge lengths per face
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)

        # Diameter = max edge length of triangular face
        face_diameter = np.maximum.reduce([e01, e12, e20])

        self.face_normals = face_normals
        self.face_areas = 0.5 * face_areas
        self.face_diam = face_diameter
        self.num_faces = len(face_areas)


def get_pt(x,ref) :
    
    if np.abs(ref[0] - x[0]) > 0.7 :
                        
        if x[0]> 0.9 :

            aux_pt = x
            x = [aux_pt[0]-1,aux_pt[1],aux_pt[2]]

        elif x[0] < 0.1 :
            aux_pt = x
            x = [aux_pt[0]+1,aux_pt[1],aux_pt[2]]
        
    if np.abs(ref[1] - x[1]) > 0.7 :
        
        if x[1]> 0.9 :

            aux_pt = x
            x = [aux_pt[0],aux_pt[1]-1,aux_pt[2]]

        elif x[1] < 0.1 :
            aux_pt = x
            x = [aux_pt[0],aux_pt[1]+1,aux_pt[2]]

    if np.abs(ref[2] - x[2]) > 0.7 :
        
        if x[2]> 0.9 :

            aux_pt = x
            x = [aux_pt[0],aux_pt[1],aux_pt[2]-1]

        elif x[2] < 0.1 :
            aux_pt = x
            x = [aux_pt[0],aux_pt[1],aux_pt[2]+1]

    return x
 
def orientation(el,i_el) :

    possibilities = itertools.permutations([0, 1, 2, 3])
    for poss in possibilities :

        X_K = np.array([[1, 1, 1, 1],[el[poss[0]][0], el[poss[1]][0], el[poss[2]][0], el[poss[3]][0]],[el[poss[0]][1], el[poss[1]][1], el[poss[2]][1], el[poss[3]][1]], [el[poss[0]][2], el[poss[1]][2], el[poss[2]][2], el[poss[3]][2]]])
        if np.linalg.det(X_K) <= 0 :
            continue
        else :
            return [[el[poss[0]],el[poss[1]],el[poss[2]],el[poss[3]]],[i_el[poss[0]],i_el[poss[1]],i_el[poss[2]],i_el[poss[3]]]]
        
    return [el,i_el]


def compute_face_neighbors(el):
# Compute tetrahedron face neighbors with periodic BCs on unit cube. Supports multiple periodic faces per tetrahedron.

    # unique node indexing
    all_vertices = el.reshape(-1, 3)
    unique_vertices, inverse = np.unique(all_vertices, axis=0, return_inverse=True)
    tets = inverse.reshape(-1, 4)

    N = tets.shape[0]

    face_ids = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])

    # extract faces
    faces = []
    for tet_i in range(N):
        for f_i, face in enumerate(face_ids):
            nodes = tets[tet_i, face]
            coords = unique_vertices[nodes]
            faces.append((tet_i, f_i, nodes, coords))

    # match interior faces
    neighbors = -np.ones((N, 4), dtype=int)
    face_map = {}

    unmatched_faces = []

    for tet_i, f_i, nodes, coords in faces:
        key = tuple(sorted(nodes))
        if key in face_map:
            t2, f2 = face_map[key]
            neighbors[tet_i, f_i] = t2
            neighbors[t2, f2] = tet_i
        else:
            face_map[key] = (tet_i, f_i)

    # Collect unmatched
    for key, (tet_i, f_i) in face_map.items():
        if neighbors[tet_i, f_i] == -1:
            coords = unique_vertices[list(key)]
            unmatched_faces.append((tet_i, f_i, coords))

    # build periodic hash by wrapped geometry
    def wrap(p):
        return p - np.floor(p)  # map into [0,1)

    periodic_hash = {}

    for tet_i, f_i, coords in unmatched_faces:
        wrapped = wrap(coords.copy())

        # Sort vertices for permutation invariance
        wrapped_sorted = np.sort(wrapped, axis=0)

        # Round to stabilize floating-point
        key = tuple(np.round(wrapped_sorted.flatten(), 12))

        if key not in periodic_hash:
            periodic_hash[key] = []
        periodic_hash[key].append((tet_i, f_i))

    # match periodic faces globally 
    for key, owners in periodic_hash.items():
        if len(owners) < 2:
            # If this happens, periodic topology is broken
            continue

        # Pair in sequence
        for i in range(0, len(owners), 2):
            (t0, f0) = owners[i]
            (t1, f1) = owners[i + 1]

            neighbors[t0, f0] = t1
            neighbors[t1, f1] = t0

    return neighbors



# ------------------------------------------------------------------------------------------


class intersect_mesh :

    def __init__(self,K,F):
            
        points_inter = []
        points_primal = len(K.points)

        for pt in K.points :
            points_inter.append(pt)

        circum3D_points = len(K.CC)
        for i in range(K.num) :
            points_inter.append(K.CC[i])

        for i in range(F.num) :
            points_inter.append(F.CC[i])

        self.simplices = []

        self.F = faces_inter_mesh(K)

        self.K_primal = []
        self.K_dual = []

        aux_el = []
        areaK = []
        interF = []
        index_Kinter = 0
        degenerate = 0
        for i in range(F.num) :

            [i1,j1] = F.indKs[i][0]
            i1 = int(i1)
            j1 = int(j1)
            CCFi = F.CC[i]
            CCi1 = K.CC[i1]
            
            if F.is_interior[i] : # either 1 (interior face) or 0 (boundary face)
                
                [i2,j2] = F.indKs[i][1]
                i2 = int(i2)
                j2 = int(j2)
                CCi2 = K.CC[i2]

                indices = [[1,2],[0,2],[0,1]] # indices of edges
                indices_neigh = [[2,1],[2,0],[1,0]] # indices of touching edges (through point)
 
                indices_face = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]] # indices of faces and indices of touching faces (through edge), MAYBE NOT IN RIGHT ORDER
                indices_face_neigh = [[0,0,0],[0,1,1],[1,1,2],[2,2,2]] # indices of edges of touching faces 

                #j_neigh \in indices_face = k in tet indices 
                #k_neigh \in indices_face_neigh = j1 in face indices; indices_face tells you what j to consider, then again in indices_face go to the entry associated with the new j value and find entry of indices such that remaining values from indices_faces match with the ones from new j
                # for tupel j,k go through like: this is the new j=0 and then we want to get [2,3] via indices [1,2]
                for k in range(3) : # three dual elements per face / go through points of face, where indices[k] goes through edges of face
                    
                    pt1 = CCi1
                    pt2 = CCi2
                    ptF = CCFi

                    if np.linalg.norm(K.F.el[i1][j1]-K.F.el[i2][j2]) > 0.1: 

                        midk1 = 1/2*(K.F.el[i1][j1][indices[k][0]] + K.F.el[i1][j1][indices[k][1]])

                        pt1 = get_pt(CCi1,midk1)
                        pt2 = get_pt(CCi2,midk1)
                        ptF = get_pt(CCFi,midk1)

                    # get intersected element
                    aux_el.append([K.F.el[i1][j1][indices[k][0]],K.F.el[i1][j1][indices[k][1]],pt1,ptF]) 
                    self.simplices.append([K.F.simplices[i1][j1][indices[k][0]],K.F.simplices[i1][j1][indices[k][1]],points_primal+i1,points_primal+circum3D_points+i])
                    
                    if check_coplanar(aux_el[-1]) :
                        degenerate += 1
                    
                    # get area
                    aux = np.cross(aux_el[-1][1]-aux_el[-1][0],aux_el[-1][2]-aux_el[-1][0]) # C = K[0], B = K[1], D = K[2], A = K[3]
                    aux = np.dot(aux,aux_el[-1][3]-aux_el[-1][0])
                    areaK.append(np.abs(aux)/6) # get area of tetrahedron K

                    # get intersected faces
                    aux_interF = []

                    aux_interF.append([K.F.el[i1][j1][indices[k][0]],K.F.el[i1][j1][indices[k][1]],ptF]) # is in primal face F
                    self.F.SinF[index_Kinter][0][0] = i1 # set primal element as intersected face lies in interior
                    self.F.SinF[index_Kinter][0][1] = i2 
                    self.F.SinF[index_Kinter][0][2] = K.F.dual[i1][j1][k] # dual element in which inter face S lies
                    
                    aux_interF.append([K.F.el[i1][j1][indices[k][0]],pt1,ptF]) # not in primal face F
                    k_neigh = indices_neigh[k][0]
                    self.F.SinF[index_Kinter][1][2] = K.F.dual[i1][j1][k] # dual element on which we work
                    self.F.SinF[index_Kinter][1][3] = K.F.dual[i1][j1][k_neigh] # dual element that is neighbor and defined on same primal face; one of two
                    self.F.SinF[index_Kinter][1][0] = i1 # primal element in which inter face S lies

                    aux_interF.append([K.F.el[i1][j1][indices[k][1]],pt1,ptF]) # not in primal face F
                    k_neigh = indices_neigh[k][1]
                    self.F.SinF[index_Kinter][2][2] = K.F.dual[i1][j1][k] # dual element on which we work
                    self.F.SinF[index_Kinter][2][3] = K.F.dual[i1][j1][k_neigh] # dual element that is neighbor and defined on same primal face; two of two
                    self.F.SinF[index_Kinter][2][0] = i1 # primal element in which inter face S lies

                    aux_interF.append([K.F.el[i1][j1][indices[k][0]],K.F.el[i1][j1][indices[k][1]],pt1]) # not in primal face F
                    j_neigh = indices_face[j1][k]
                    k_neigh = indices_face_neigh[j1][k]
                    self.F.SinF[index_Kinter][3][2] = K.F.dual[i1][j1][k] # dual element on which we work
                    self.F.SinF[index_Kinter][3][3] = K.F.dual[i1][j_neigh][k_neigh] # dual element that is neighbor but defined on a different primal face
                    self.F.SinF[index_Kinter][3][0] = i1 # primal element in which inter face S lies

                    interF.append(aux_interF) # intersected faces of one intersected element

                    # structure intersected element
                    K.TinK[i1].append(index_Kinter) #  say primal mesh which intersected mesh elements are in primal element
                    self.K_primal.append(i1) # intersected element T is in primal element with index i1
                    self.K_dual.append(K.F.dual[i1][j1][k]) # intersected element T is in dual element with index written here
                    
                    index_Kinter += 1
    
                    # do same once again for pair [i2,j2] :
                    # get intersected element
                    aux_el.append([K.F.el[i1][j1][indices[k][0]],K.F.el[i1][j1][indices[k][1]],pt2,ptF]) 
                    self.simplices.append([K.F.simplices[i1][j1][indices[k][0]],K.F.simplices[i1][j1][indices[k][1]],points_primal+i2,points_primal+circum3D_points+i])
                    
                    if check_coplanar(aux_el[-1]) :
                        degenerate += 1
                               
                    # get area
                    aux = np.cross(aux_el[-1][1]-aux_el[-1][0],aux_el[-1][2]-aux_el[-1][0]) # C = K[0], B = K[1], D = K[2], A = K[3]
                    aux = np.dot(aux,aux_el[-1][3]-aux_el[-1][0])
                    areaK.append(np.abs(aux)/6) # get area of tetrahedron K

                    # get intersected faces
                    aux_interF = []

                    aux_interF.append([K.F.el[i1][j1][indices[k][0]],K.F.el[i1][j1][indices[k][1]],ptF]) # is in primal face F
                    self.F.SinF[index_Kinter][0][0] = i1 # set primal element as intersected face lies in interior
                    self.F.SinF[index_Kinter][0][1] = i2 # gives primal indices to determine primal face F
                    self.F.SinF[index_Kinter][0][2] = K.F.dual[i2][j2][k] # dual element in which inter face S lies

                    aux_interF.append([K.F.el[i1][j1][indices[k][0]],pt2,ptF]) # not in primal face F
                    k_neigh = indices_neigh[k][0]
                    self.F.SinF[index_Kinter][1][2] = K.F.dual[i2][j2][k] # dual element on which we work
                    self.F.SinF[index_Kinter][1][3] = K.F.dual[i2][j2][k_neigh] # dual element that is neighbor and defined on same primal face
                    self.F.SinF[index_Kinter][1][0] = i2 # primal element in which inter face S lies

                    aux_interF.append([K.F.el[i1][j1][indices[k][1]],pt2,ptF]) # not in primal face F
                    k_neigh = indices_neigh[k][1]
                    self.F.SinF[index_Kinter][2][2] = K.F.dual[i2][j2][k] # dual element on which we work
                    self.F.SinF[index_Kinter][2][3] = K.F.dual[i2][j2][k_neigh] # dual element that is neighbor and defined on same primal face
                    self.F.SinF[index_Kinter][2][0] = i2 # primal element in which inter face S lies

                    aux_interF.append([K.F.el[i1][j1][indices[k][0]],K.F.el[i1][j1][indices[k][1]],pt2]) # not in primal face F
                    j_neigh = indices_face[j2][k]
                    k_neigh = indices_face_neigh[j2][k]
                    self.F.SinF[index_Kinter][3][2] = K.F.dual[i2][j2][k] # dual element on which we work
                    self.F.SinF[index_Kinter][3][3] = K.F.dual[i2][j_neigh][k_neigh] # dual element that is neighbor but defined on a different primal face
                    self.F.SinF[index_Kinter][3][0] = i2 # primal element in which inter face S lies

                    interF.append(aux_interF) # intersected faces of one intersected element

                    # structure intersected element
                    K.TinK[i2].append(index_Kinter) #  say primal mesh which intersected mesh elements are in primal element
                    self.K_primal.append(i2) # intersected element T is in primal element with index i1
                    self.K_dual.append(K.F.dual[i2][j2][k]) # intersected element T is in dual element with index written here
                    
                    index_Kinter += 1

                
            else :

                print('Warning: periodic boundary does not work for the intersected mesh.')

        self.area = np.array(areaK)
        self.el = np.array(aux_el)
        self.num = len(aux_el)

        if degenerate > 0 :
            print('Warning: The "intersected" mesh contains %i degenerate element(s).' %degenerate)
            self.quality = degenerate
        else :
            print('The "intersected" mesh DOES NOT contain degenerate elements.')
            self.quality = 0

        self.F.el = np.array(interF)

        # get array of faces
        self.neighbors = np.asarray(compute_face_neighbors(self.el))

        indexF = 0
        # Use dictionary for fast face lookup
        face_dict = {}
        aux_el = self.el

        indices = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
                ])

        # Preallocate arrays
        max_faces = 2*self.num

        aux_el = np.zeros((max_faces, 3, 3))
        aux_indKs = -np.ones((max_faces, 2), dtype=int)
        ident_primal = -np.ones((max_faces, 2), dtype=int)
        ident_dual = -np.ones((max_faces, 2), dtype=int)

        for i in range(self.num):
            neighbors = self.neighbors[i]
            for j in range(4):
                ni = neighbors[j]
                face_key = tuple(sorted([i, ni]))  # unique key for face

                if ni >= 0:  # interior face
                    if face_key in face_dict:
                        idx = face_dict[face_key]
                        # Replace face if on right side
                        if aux_el[idx][0,0] > 0.999999 and aux_el[idx][1,0] > 0.999999 and aux_el[idx][2,0] > 0.999999:
                            # Swap previous with current
                            aux_el[idx] =  self.F.el[i][j]
                            aux_indKs[idx][1] = aux_indKs[idx][0].copy()
                            aux_indKs[idx][0] = i

                            if self.F.SinF[i][j][3] < 0  :
                                ident_primal[idx][1] = ident_primal[idx][0].copy()
                                ident_primal[idx][0] = self.F.SinF[i][j][0]
                                ident_dual[idx][0] = self.F.SinF[i][j][2]
                            elif self.F.SinF[i][j][1] < 0 :
                                ident_dual[idx][1] = ident_dual[idx][0].copy()
                                ident_dual[idx][0] = self.F.SinF[i][j][2]
                                ident_primal[idx][0] = self.F.SinF[i][j][0]
                            else :
                                print('Warning: Something is wrong with intersected indexing.')

                        else:
                            aux_indKs[idx][1] = i
                    else:
                        # New interior face
                        aux_el[indexF] = self.F.el[i][j]
                        aux_indKs[indexF][0] = i

                        if self.F.SinF[i][j][3] < 0 : # on primal face
                            ident_primal[indexF][0] = self.F.SinF[i][j][0]
                            ident_primal[indexF][1] = self.F.SinF[i][j][1]
                            ident_dual[indexF][0] = self.F.SinF[i][j][2]
                        elif self.F.SinF[i][j][1] < 0 : # on dual face
                            ident_dual[indexF][0] = self.F.SinF[i][j][2]
                            ident_dual[indexF][1] = self.F.SinF[i][j][3]
                            ident_primal[indexF][0] = self.F.SinF[i][j][0]
                        else :
                            print('Warning: Something is wrong with intersected indexing.')

                        face_dict[face_key] = indexF
                        indexF += 1
                else:  # boundary face, cannot happen for periodic

                    print("Warning: periodic boundary face encountered")

        self.face_el = np.asarray(aux_el[:indexF])
        self.face_to_tet = np.asarray(aux_indKs[:indexF])
        self.face_ident_primal = np.asarray(ident_primal[:indexF])
        self.face_ident_dual = np.asarray(ident_dual[:indexF])

        v0 = self.face_el[:,0]
        v1 = self.face_el[:,1]
        v2 = self.face_el[:,2]

        cross = np.cross(v1 - v0, v2 - v0)   # shape (N_faces, 3)

        # Face areas: 1/2 * |cross|
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Unit normals
        face_normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)

        # Face centers
        face_centers = (v0 + v1 + v2)/3

        # Tet centers of + tetrahedra
        tet_centers = np.mean(self.el, axis=1)
        plus_centers = tet_centers[self.face_to_tet[:,0]]

        # Flip normals to point from + tet to − tet
        flip = np.sum(face_normals * (face_centers - plus_centers), axis=1) < 0
        face_normals[flip] *= -1

        # Edge lengths per face
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)

        # Diameter = max edge length of triangular face
        face_diameter = np.maximum.reduce([e01, e12, e20])

        self.face_normals = face_normals
        self.face_areas = 0.5 * face_areas
        self.face_diam = face_diameter
        self.num_faces = len(face_areas)

    
class faces_inter_mesh :

            def __init__(self,K):
                self.el = []
                self.area = []
                self.normal = []
                self.SinF = - np.ones([12*K.num,4,4],dtype= int)
                self.is_interior = np.ones([12*K.num,4])
                self.diam = []


def post_process(K_inter,K) :

    K_el = K.points[K.simplices]
    for i in range(K_inter.num_faces) :
        midS = (K_inter.face_el[i][0]+K_inter.face_el[i][1]+K_inter.face_el[i][2])/3
        k = K_inter.face_ident_primal[i][0]
        midprim = (K_el[k][0]+K_el[k][1]+K_el[k][2]+K_el[k][3])/4

        if np.linalg.norm(midS-midprim)>0.5 : 

            if np.abs(midS[0] - midprim[0]) > 0.5 :
                
                if midS[0]> 0.9 :
    
                    K_inter.face_el[i] = K_inter.face_el[i] - np.array([[1,0,0],[1,0,0],[1,0,0]])

                elif midS[0] < 0.1 :
                
                    K_inter.face_el[i] =  K_inter.face_el[i] + np.array([[1,0,0],[1,0,0],[1,0,0]])
                
            if np.abs(midS[1] - midprim[1]) > 0.5 :
                
                if midS[1]> 0.9 :
    
                    K_inter.face_el[i] =  K_inter.face_el[i] - np.array([[0,1,0],[0,1,0],[0,1,0]])

                elif midS[1] < 0.1 :
                    
                    K_inter.face_el[i] =  K_inter.face_el[i] + np.array([[0,1,0],[0,1,0],[0,1,0]])


            if np.abs(midS[2] - midprim[2]) > 0.5 :
                
                if midS[2]> 0.9 :
    
                    K_inter.face_el[i] = K_inter.face_el[i] - np.array([[0,0,1],[0,0,1],[0,0,1]])

                elif midS[2] < 0.1 :

                    K_inter.face_el[i] =  K_inter.face_el[i] + np.array([[0,0,1],[0,0,1],[0,0,1]])

def post_process_TinK(K_inter,K) :

    K_el = K.points[K.simplices]
    for i in range(K.num) :
        midprim = 1/4*(K_el[i][0]+K_el[i][1]+K_el[i][2]+K_el[i][3])
        for j in K.TinK[i]:
            midint = 1/4*(K_inter.el[j][0]+K_inter.el[j][1]+K_inter.el[j][2]+K_inter.el[j][3])
            
            if np.linalg.norm(midint-midprim)>0.5 : 

                if np.abs(midint[0] - midprim[0]) > 0.5 :
                    
                    if midint[0]> 0.9 :
        
                        K_inter.el[j] = K_inter.el[j] - np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0]])

                    elif midint[0] < 0.1 :
                    
                        K_inter.el[j] = K_inter.el[j] + np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
                    
                if np.abs(midint[1] - midprim[1]) > 0.5 :
                    
                    if midint[1]> 0.9 :
        
                        K_inter.el[j] = K_inter.el[j] - np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0]])

                    elif midint[1] < 0.1 :
                        
                        K_inter.el[j] = K_inter.el[j] + np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0]])


                if np.abs(midint[2] - midprim[2]) > 0.5 :
                    
                    if midint[2]> 0.9 :
        
                        K_inter.el[j] = K_inter.el[j] - np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1]])

                    elif midint[2] < 0.1 :

                        K_inter.el[j] = K_inter.el[j] + np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1]])


# ------------------------------------- NUMERICAL SCHEME -----------------------------------

# FV FE scheme
def assemble_FE_matrix(K_dual): # see Fig. 3.14. in Bartels2015

    # u = np.zeros(len(points_dual))
    # tu_D = np.zeros(len(points_dual))
    # b = np.zeros(len(points_dual))

    iter = 0
    iter_max = 16*K_dual.num # 16 = (3+1)^2 = (d+1)^2

    I = np.zeros(iter_max)
    J = np.zeros(iter_max)
    X_diff = np.zeros(iter_max)
    X_reac = np.zeros(iter_max)

    m_loc = 1/20*(np.ones([4,4]) + np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])) # see Figu. 3.23 in Bartels Numerics for PDEs, 20 = 4*5 = (3+1)*(3+2) = (d+1)*(d+2)
    # m_lumped = 1/4*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) # 1/4 = 1/(d+1)

    gradients = []
    volumes = []
    for i in range(K_dual.num) :

        X_K = np.array([[1, 1, 1, 1],[K_dual.el[i][0][0], K_dual.el[i][1][0], K_dual.el[i][2][0], K_dual.el[i][3][0]],[K_dual.el[i][0][1], K_dual.el[i][1][1], K_dual.el[i][2][1], K_dual.el[i][3][1]], [K_dual.el[i][0][2], K_dual.el[i][1][2], K_dual.el[i][2][2], K_dual.el[i][3][2]]])
        rhs = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        grads_K = np.linalg.solve(X_K,rhs) # gradient values of hat functions evaluated at vertices of tetrahedron
        gradients.append(grads_K)
        vol_K = np.linalg.det(X_K)/6 # 6 = 3*2*1 = 3! = d!

        volumes.append(vol_K)

        if vol_K <= 0 :
            print('Warning: FE scheme works with non-positive volumes.')

        for m in [0,1,2,3] : # go through vertices of tet
            #b[tri_dual.simplices[i][m]] += 1/(d+1)* vol_K * f(mid_K) # 1/(d+1) * midpoint rule
            for n in [0,1,2,3] : # go through vertices of tet
                I[iter] = K_dual.pt_ident[K_dual.simplices[i][m]] # identify_points
                J[iter] = K_dual.pt_ident[K_dual.simplices[i][n]] # identify_points
                X_diff[iter] = vol_K*np.dot(grads_K[m],grads_K[n])
                X_reac[iter] = vol_K*m_loc[m][n]
                #X_reac[iter] = vol_K*m_lumped[m][n]
                iter += 1

    sparseA_diff = csc_matrix((X_diff[:iter], (I[:iter], J[:iter])), shape=(len(K_dual.pt_dual_reduced),len(K_dual.pt_dual_reduced))) # use reduced len(K_dual.points)
    sparseA_reac = csc_matrix((X_reac[:iter], (I[:iter], J[:iter])), shape=(len(K_dual.pt_dual_reduced),len(K_dual.pt_dual_reduced)))

    sparseA = sparseA_diff + sparseA_reac

    K_dual.grads = np.array(gradients)
    K_dual.vol = np.array(volumes)

    return sparseA

def assemble_FV_matrix(ht,K,eps) :

    row = []
    col = []
    data = []

    for i in range(K.num) :

        areaK = K.area[i]

        neighbors = K.neighbors[i] 
        
        valK = 1 # reset valK to one

        for j in range(4) : # go through edges

            areaF = K.F.area[i][j]

            if neighbors[j] >= 0 : # i.e. edge E is interior

                dE = K.F.dist[i][j]      
                valK += eps*ht/areaK*areaF/dE
                valL = -eps*ht/areaK*areaF/dE

                row.append(i)
                col.append(neighbors[j])
                data.append(valL)

            else : # i.e. edge E is part of the boundary
                
                print('Warning: Something is wrong with the periodic boundary!')
                valL = 0

        row.append(i) # diagonal entries of matrix
        col.append(i)
        data.append(valK)

    # creating sparse matrix
    sparseA = csc_matrix((data, (row, col)), shape = (K.num, K.num))#.toarray()

    return sparseA


def getc_FE(rhoFV,g,K_dual,sparseA) :

    b = np.zeros(len(K_dual.pt_dual_reduced))

    for i in range(K_dual.num) :

        vol_K = K_dual.vol[i]
        mid_K = 1/4*(K_dual.el[i][0]+K_dual.el[i][1]+K_dual.el[i][2]+K_dual.el[i][3])

        [i1,i2] = K_dual.primal_ind[i]
        [vol1,vol2] = K_dual.primal_area[i]

        for m in [0,1,2,3] : #  "m = 1:(d+1)"; go through vertices of tet
            # b[K_dual.pt_ident[K_dual.simplices[i][m]]] += 1/4*vol_K*g(mid_K)
            if i2 > -0.5 : # interior
                b[K_dual.pt_ident[K_dual.simplices[i][m]]] += 1/4*(vol_K*g(mid_K) + vol1*rhoFV[i1] + vol2*rhoFV[i2]) # using midpoint rule on g and exact integration of rhoFV; 1/4 = 1/(d+1) -> see Bartels Lemma 3.10
            else : # boundary
                print('Warning: FE does not work correctly with periodic boundary')
                b[K_dual.pt_ident[K_dual.simplices[i][m]]] += 1/4*vol_K*(g(mid_K) + rhoFV[i1]) # using midpoint rule on g and exact integration of rhoFV; 1/4 = 1/(d+1)

    c, content = sp.sparse.linalg.cg(sparseA,b,tol=10**-12) # c_h^n(x) = sum_i (c_i*hat_i(x))
    return c

def get_gradc(K_dual,c) :
    
    val = []
    for i in range(K_dual.num) :
        val.append(np.matmul(np.transpose(K_dual.grads[i]),c[K_dual.pt_ident[K_dual.simplices[i]]]))
    
    return np.array(val) # constant gradient on each element


def finitevolumescheme_rho_expl(u_old,ht,K,vv,f,sparseA) : # explicit euler for full advection term

    rhs = []
    K_el = K.points[K.simplices]

    for i in range(K.num) :

        areaK = K.area[i]
        [fK,avrg] = avrgK(K_el[i],areaK,f) # get RHS of PDE fK, local mean values over element K

        val_rhs = ht/areaK*fK + u_old[i]

        neighbors = K.neighbors[i]
        for j in range(4) : # go through faces

            if neighbors[j] >= 0 : # i.e. edge E is interior

                if u_old[i] < 10**-12 or u_old[neighbors[j]] < 10**-12 :  # log-mean not well-defined
                    val_rhs -= 0
                
                else :

                    [idual,jdual,kdual] = K.F.dual[i][j]
                    [iarea,jarea,karea] = K.F.dual_area[i][j]
                    vKF = np.dot(iarea*vv[int(idual)]+jarea*vv[int(jdual)]+karea*vv[int(kdual)],K.F.n[i][j]) # exact integral (!), areaF factor already here

                    if np.abs(np.log(u_old[i]) - np.log(u_old[neighbors[j]])) < 10**-12 : # log-mean not well-defined
                        val_rhs -= ht/areaK*u_old[i]*vKF
                    
                    else : # log-mean well-defined
                        val_rhs -= ht/areaK*(u_old[i]-u_old[neighbors[j]])/(np.log(u_old[i]) - np.log(u_old[neighbors[j]]))*vKF # * 1 = areaF/areaF
 
            else : # i.e. edge E is part of the boundary

                val_rhs -= 0
                print('Warning: Something is wrong with the periodic boundary!')

        rhs.append(val_rhs) # get vector for right-hand-side of linear system of equations

    # creating sparse matrix
    uh, content = sp.sparse.linalg.gmres(sparseA, rhs, tol=10**-12) # iterative solver

    return uh

def avrgK(K,areaK,g) :

    # gauß qadrature, https://www.cfd-online.com/Wiki/Code:_Quadrature_on_Tetrahedra
    xi = [[0.5854101966249685,0.1381966011250105,0.1381966011250105],[0.1381966011250105,0.1381966011250105,0.1381966011250105],[0.1381966011250105,0.1381966011250105,0.5854101966249685],[0.1381966011250105,0.5854101966249685,0.1381966011250105]]
    wi= np.array([0.25, 0.25, 0.25, 0.25])/6

    unitK = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]

    val = 0
    for i in range(len(wi)) :
        [trafoxi,M] = tritrafo(unitK,K,xi[i])
        val += wi[i]*g(trafoxi)

    avrg = val*6 # divided by 1/6, where 1/6 is volume of unit triangle
    integral = avrg*areaK

    return [integral,avrg]

def tritrafo(K,L,pt):
# Affine map from tetrahedron K to L, transform point pt. Returns transformed point and Jacobian matrix J (3x3).

    K = np.array(K)
    L = np.array(L)
    
    # Linear part
    J = np.column_stack((L[1]-L[0], L[2]-L[0], L[3]-L[0])) @ \
        np.linalg.inv(np.column_stack((K[1]-K[0], K[2]-K[0], K[3]-K[0])))
    
    # Translation
    b = L[0] - J @ K[0]
    
    ptL = np.asarray(J @ pt + b)
    return ptL, J


# morley reconstruction
def getinterpolationRHS(K,uh) :

    FKED = [] # nicaise paper notation

    for i in range(K.num) :
        neighbors = K.neighbors[i] 

        auxFKED = []
        for j in range(4) : # go through neighboring triangles K[neighbors[j]] / faces of K[i]

            areaF = K.F.area[i][j]

            if neighbors[j] >= 0 : # i.e. interior face

                dF = K.F.dist[i][j]
                auxFKED.append(areaF/dF*(uh[neighbors[j]]-uh[i])) # diffusive flux

            else : # i.e. face F is part of the boundary 

                print('Warning: Something is wrong with the periodic boundary!')

        FKED.append(auxFKED)

    return FKED 

def getq0(K,uh) :

    vertex_val = []
    for i in range(len(K.points)) :
        neighborstri = K.adjacent[K.pt_ident[i]] # indices of triangles that touch vertex associated with pindex

        #calculate vertex_val
        aux = 0
        for k in neighborstri :
            aux += uh[k]

        vertex_val.append(aux/len(neighborstri))

    return np.array(vertex_val)

def getbetaE(K,vertex_val,FKED) :

    betaKE = [] # initilize
    numerKE = []
    denomKE = []
   
    for i in range(K.num) :

        areaK = K.area[i]
        
        gradq0 = 0
        for j in range(4) :
            pt_i = K.simplices[i][j]
            gradq0 += vertex_val[pt_i]*np.array([K.hat[i][j][0],K.hat[i][j][1],K.hat[i][j][2]])

        betaE = []
        numer = []
        denom = []

        for j in range(4) : # go through neighboring triangles K[neighbors[j]] / go through edges of K[i]
            
            areaF = K.F.area[i][j]
            normalKF = K.F.n[i][j]
            I = areaF*np.dot(gradq0,normalKF) # get integral over gradq0*normal i.e. normal derivative of q0

            III = - (16/35)*(areaF**2)/areaK # = int_F b_F ( grad b_K \cdot n) dS(x)
            
            aux_numer = (FKED[i][j]-I)
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


def tritrafo_quad_tet(L, pt_quad):
# Vectorized affine map from multiple tetrahedra to reference points.

    # Edge vectors
    v0 = L[:,0,:]           # (M,3)
    v1v0 = L[:,1,:] - v0
    v2v0 = L[:,2,:] - v0
    v3v0 = L[:,3,:] - v0

    # Jacobian matrix (3x3 edge vectors)
    J = np.stack([v1v0, v2v0, v3v0], axis=-1)  # (M,3,3)

    # Map reference points to physical tetrahedra
    # pt_quad: (N,3) = (xi, eta, zeta)
    ptL = (pt_quad[:,0][None,:,None]*v1v0[:,None,:] +
           pt_quad[:,1][None,:,None]*v2v0[:,None,:] +
           pt_quad[:,2][None,:,None]*v3v0[:,None,:] +
           v0[:,None,:])  # (M,N,3)

    # Gradients of barycentric coordinates in physical space
    # Solve linear system: J @ grad_ref = grad_phys -> grad_phys = inv(J^T)
    # Since we have affine tetrahedron, the gradient is constant
    invJt = np.linalg.inv(J.transpose(0,2,1))  # (M,3,3)
    grads123 = invJt.transpose(0,2,1)          # ∇λ1, ∇λ2, ∇λ3, shape (M,3,3)
    grad4 = -np.sum(grads123, axis=1, keepdims=True)  # ∇λ4
    grads = np.concatenate([grads123, grad4], axis=1)  # (M,4,3)

    return ptL, invJt, v0, grads

def tritrafo_quad_face_and_bary(F_el, pt_ref):
    
    N_faces = F_el.shape[0]
    N_pts = pt_ref.shape[0]

    # canonical affine map to physical shared face
    v0 = F_el[:,0]  # (N_faces,3)
    v1 = F_el[:,1]
    v2 = F_el[:,2]

    e1 = v1 - v0
    e2 = v2 - v0

    xi = pt_ref[:,0]
    eta = pt_ref[:,1]

    pts = v0[:,None,:] + xi[None,:,None]*e1[:,None,:] + eta[None,:,None]*e2[:,None,:] # shape (N_faces,N_pts,3)


    # Triangle barycentric coordinates
    # Solve xi,eta from X = v0 + xi*e1 + eta*e2
    E = np.stack([e1,e2], axis=2)  # (N_faces,3,2)
    rhs = pts  # same points
    ET_E = np.einsum('fij,fik->fjk', E, E)
    ET_E_inv = np.linalg.inv(ET_E)
    ET_rhs = np.einsum('fji,fpj->fpi', E, rhs - v0[:,None,:])
    xi_eta = np.einsum('fij,fpj->fpi', ET_E_inv, ET_rhs)

    lambda_face = np.zeros((N_faces,N_pts,3))
    lambda_face[:,:,1] = xi_eta[:,:,0]
    lambda_face[:,:,2] = xi_eta[:,:,1]
    lambda_face[:,:,0] = 1 - xi_eta[:,:,0] - xi_eta[:,:,1]

    assert np.allclose(lambda_face.sum(axis=2), 1)

    return pts, lambda_face



# Barycentric coordinates for many points
def barycentric_coords(X, Ainv, v0):
# Compute barycentric coordinates of N_points in N_tet tetrahedra.

    # Shift points by first vertex of each tetrahedron
    X_shifted = X - v0[:, np.newaxis, :]  # (N_tet, N_points, 3)

    # Multiply by Ainv per tetrahedron
    l123 = np.einsum('tpi,tij->tpj', X_shifted, Ainv)  # (N_tet, N_points, 3)

    # Fourth barycentric coordinate
    l4 = 1 - np.sum(l123, axis=2, keepdims=True)       # (N_tet, N_points, 1)

    # Stack all four coordinates
    L = np.concatenate([l123, l4], axis=2)            # (N_tet, N_points, 4)

    return L

def barycentric_coords_groups(X, Ainv, v0, inverse_map):
# Compute barycentric coordinates for all points in their corresponding tetrahedra

    N = X.shape[0]
    N_pt = X.shape[1]
    L_all = np.empty((N, N_pt, 4), dtype=X.dtype)

    for m, points_idx in enumerate(inverse_map):
        if len(points_idx) == 0:
            continue
        # Shift points and multiply by inverse Jacobian
        X_shifted = X[points_idx] - v0[m]               # shape: (num_points_in_m, 33, 3)
        l123 = np.einsum('npi,ij->npj', X_shifted, Ainv[m])  # multiply each subpoint by Ainv
        l4 = 1 - np.sum(l123, axis=2, keepdims=True)         # shape: (num_points_in_m, 33, 1)
        L_all[points_idx] = np.concatenate([l123, l4], axis=2) # shape: (num_points_in_m, 33, 4)

    return L_all


# bubble functions and derivatives
def bubble_K(L):
# Element bubble b_K = 256 λ1 λ2 λ3 λ4

    return 256.0 * np.prod(L, axis=-1)

def grad_bubble_K(L, grads):
    # Determine if we need to add a dummy axis
    add_dummy_L = L.ndim == 3       # (N_tet, N_pt, 4)
    add_dummy_g = grads.ndim == 3   # (N_tet, 4, 3)

    if add_dummy_L:
        L = L[:, None, :, :]        # shape -> (N_tet, 1, N_pt, 4)
    if add_dummy_g:
        grads = grads[:, None, :, :]  # shape -> (N_tet, 1, 4, 3)

    # indices of j≠i
    idx = np.array([
        [1,2,3],   # exclude 0
        [0,2,3],   # exclude 1
        [0,1,3],   # exclude 2
        [0,1,2]    # exclude 3
    ]) 

    # Extract the 3 lambdas for j≠i:
    L_ex = L[..., idx]  # shape (..., 4, 3)

    # Product over the 3 lambdas
    prod_except = np.prod(L_ex, axis=-1)  # shape (..., 4)

    g = grads[:, :, None, :, :]  # shape (..., 1, 4, 3)

    # contraction over the barycentric index i = 0..3
    grad = np.einsum('...i,...ij->...j', prod_except, g)

    # remove dummy axes if they were added
    if add_dummy_L:
        grad = grad[:, 0, :, :]
    if add_dummy_g:
        grad = grad[:, :, :]  # no change needed, just for clarity

    return 256.0 * grad

def grad_bubble_K_face(K, F, lambda_face):
# Gradient of the tetrahedron interior bubble restricted to a triangular face.

    N_face, N_pt, _ = lambda_face.shape

    grad_face = np.zeros([N_face, 2 , N_pt, 3])

    for i in range(2) :
        tet_vertices = K.points[np.hstack((F.simplices,F.opp[:,i,np.newaxis]))]
        # tetrahedron Jacobian
        v0 = tet_vertices[:,0,:]  # (N_faces,3)
        v1 = tet_vertices[:,1,:]
        v2 = tet_vertices[:,2,:]
        v3 = tet_vertices[:,3,:]  # opposite vertex

        # Jacobian
        J = np.stack([v1-v0, v2-v0, v3-v0], axis=2)   # (N_faces,3,3)
        grad_lam123 = np.linalg.inv(J).transpose(0,2,1) # (N_faces,3,3)
        grad_lambda3 = grad_lam123[:,:,2]              # (N_faces,3)

        # compute bubble gradient on the face
        # grad(b_K) = 256 * (λ0*λ1*λ2) * ∇λ3
        prod_lambda = lambda_face[...,0] * lambda_face[...,1] * lambda_face[...,2]  # (N_faces, N_pts)
        grad_face[:,i,...] = 256.0 * prod_lambda[...,None] * grad_lambda3[:,None,:]           # (N_faces, N_pts, 3)

    return grad_face

def laplace_bubble_K(L, grads):

    i_idx = np.array([0,0,0,1,1,2])
    j_idx = np.array([1,2,3,2,3,3])
    k_idx = np.array([2,1,1,0,0,0])
    l_idx = np.array([3,3,2,3,2,1])

    # λk * λl terms 
    # (Nt, Ne, Np, 6)
    lam_k = L[..., k_idx]
    lam_l = L[..., l_idx]
    terms = lam_k * lam_l

    #  grad_i * grad_j 
    # grads[:, i_idx, :] -> (Nt, 6, 3)
    g_i = grads[:, :, i_idx, :][:, :, None, :, :]  # (Nt,1,1,6,3)
    g_j = grads[:, :, j_idx, :][:, :, None, :, :]  # (Nt,1,1,6,3)

    # Dot product over spatial dim
    # result: (Nt, Ne, Np, 6)
    Gdot = np.einsum('...pc,...pc->...p', g_i, g_j)

    # Sum over the 6 cross terms
    lap = np.sum(terms * Gdot, axis=-1)  # (Nt, Ne, Np)

    return 512.0 * lap

def bubble_F(L, faces):
 
    faces = np.array(faces)  # (F,3)

    l_i = L[... , faces[:,0]]  # (N_tet,Q,F)
    l_j = L[... , faces[:,1]]
    l_k = L[... , faces[:,2]]

    return 27.0 * l_i * l_j * l_k

def bubble_F_face(L) :

    l_i = L[... , 0]  # (N_face,Q,3)
    l_j = L[... , 1]
    l_k = L[... , 2]

    return 27.0 * l_i * l_j * l_k

def grad_bubble_F(L, grads, faces):

    L = np.asarray(L, dtype=np.float32)
    grads = np.asarray(grads, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # Dummy axis 
    add_dummy_L = L.ndim == 3      # (N_tet, N_pt, 4)
    add_dummy_g = grads.ndim == 3  # (N_tet, 4, 3)

    if add_dummy_L:
        L = L[:, None, :, :]        # shape -> (N_tet, 1, N_pt, 4)
    if add_dummy_g:
        grads = grads[:, None, :, :]  # shape -> (N_tet, 1, 4, 3)

    N_tet, n12, N_pt, _ = L.shape
    F = faces.shape[0]

    gradF = np.zeros((N_tet, n12, N_pt, F, 3), dtype=np.float32)

    # Loop over faces 
    for f_idx, (i, j, k) in enumerate(faces):
        # Avoid fancy indexing: pick columns manually
        li = L[..., i]       # shape (N_tet, n12, N_pt)
        lj = L[..., j]
        lk = L[..., k]

        gi = grads[..., i, :]  # shape (N_tet, n12, 3)
        gj = grads[..., j, :]
        gk = grads[..., k, :]

        # Broadcast N_pt for grad arrays 
        gradF[..., f_idx, :] = (lj * lk)[..., None] * gi[:, :, None, :] + \
                               (li * lk)[..., None] * gj[:, :, None, :] + \
                               (li * lj)[..., None] * gk[:, :, None, :]

    # Remove dummy axes if added
    if add_dummy_L:
        gradF = gradF[:, 0, :, :, :]
    if add_dummy_g:
        pass  # already correct

    return 27.0 * gradF

def grad_bubble_F_face(L, grads):

    N_face, Q, _ = L.shape

    gradF = np.zeros((N_face, Q, 3), dtype=np.float32)

    li = L[..., 0]   # (N_face, Q)
    lj = L[..., 1]
    lk = L[..., 2]

    gi = grads[..., 0, :]  # (N_face, 3)
    gj = grads[..., 1, :]
    gk = grads[..., 2, :]

    gradF = (
        (lj * lk)[..., None] * gi[:, None, :] +
        (li * lk)[..., None] * gj[:, None, :] +
        (li * lj)[..., None] * gk[:, None, :]
    )

    return 27.0 * gradF

def laplace_bubble_F(L, grads, faces):

    faces = np.array(faces)          # (F,3)
    F = faces.shape[0]

    # λi, λj, λk
    l_i = L[..., faces[:,0]]
    l_j = L[..., faces[:,1]]
    l_k = L[..., faces[:,2]]

    # ∇λ_i, ∇λ_j, ∇λ_k → expand for points
    grad_i = grads[..., faces[:,0], :][:, :, None, :, :]  # (Nt,12,1,F,3)
    grad_j = grads[..., faces[:,1], :][:, :, None, :, :]
    grad_k = grads[..., faces[:,2], :][:, :, None, :, :]

    # Dot products along last axis
    dot_ij = np.einsum('...c,...c->...', grad_i, grad_j)  # (Nt,12,1,F)
    dot_ik = np.einsum('...c,...c->...', grad_i, grad_k)
    dot_jk = np.einsum('...c,...c->...', grad_j, grad_k)

    # Broadcast along N_pt
    lapF = 2.0 * (l_k*dot_ij + l_j*dot_ik + l_i*dot_jk)

    return 54.0 * lapF  # shape (Nt,12,N_pt,F)


def get_morley_val(K,pt,bF,bK,vertex_val,betaKF) :

    if pt.ndim == 3 :
        # PRIMAL POINTS
        # Homogeneous coordinates
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:2], 1))], axis=-1)  # (N_tet, N_pt, 4)
        # Apply hat matrices
        hx = np.einsum('tij,tpj->tpi', np.array(K.hat), pt_h)  # (N_tet, N_pt, 4)
        # Dot with vertex values
        vv = vertex_val[K.simplices]  # (N_tet, 4)
        q0 = np.einsum('ti,tpi->tp', vv, hx)  # (N_tet, N_pt)
        # Compute aux with bubble functions
        b_vec = bF * bK[..., None]           # (N_tet, N_pt, 4)
        aux = np.einsum('ti,tpi->tp', betaKF, b_vec)  # (N_tet, N_pt)


    elif pt.ndim == 4 :
        # INTERSECTED POINTS or POINTS ON FACES PRIMAL
        # Homogeneous coordinates
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:3], 1))], axis=-1)
        # Apply hat matrices
        hx = np.einsum('tij,tkpj->tkpi', np.array(K.hat), pt_h)
        # Dot with vertex values
        vv = vertex_val[K.simplices]  # (N_tet, 4)
        q0 = np.einsum('ti,tkpi->tkp', vv, hx)  # (N_tet, 12, N_pt)
        # Compute aux with bubble functions
        b_vec = bF * bK[..., None]           # (N_tet, 12, N_pt, 4)
        aux = np.einsum('ti,tkpi->tkp', betaKF, b_vec)  # (N_tet, 12, N_pt)

    elif pt.ndim == 5 :
        # FACES INTERSECTED
        # Homogeneous coordinates
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:4], 1))], axis=-1)
        # Apply hat matrices
        hx = np.einsum('tij,tlkpj->tlkpi', np.array(K.hat), pt_h)
        # Dot with vertex values
        vv = vertex_val[K.simplices]  # (N_tet, 4)
        q0 = np.einsum('ti,tlkpi->tlkp', vv, hx)  # (N_tet, 12, N_pt)
        # Compute aux with bubble functions
        b_vec = bF * bK[..., None]           # (N_tet, 12, N_pt, 4)
        aux = np.einsum('ti,tlkpi->tlkp', betaKF, b_vec)  # (N_tet, 12, N_pt)

    return q0 + aux, q0

def get_morley_val_face(K, X, vertex_val, bF, bK, betaKF, inverse_map):
# Compute q0 and aux for points X using inverse_map per tetrahedron.

    N, N_pt, _ = X.shape

    q0_all = np.zeros((N, N_pt), dtype=X.dtype)
    aux_all = np.zeros((N, N_pt), dtype=X.dtype)

    for m, points_idx in enumerate(inverse_map): # m is primal element index, points_idx is intersected face indices
        if len(points_idx) == 0:
            continue

        pt = X[points_idx]                       # (num_points_in_m, 33, 3)

        # Homogeneous coordinates
        pt_h = np.concatenate([pt, np.ones((*pt.shape[:2], 1))], axis=-1)  # (num_points_in_m, 33, 4)

        # Apply hat matrices
        hx = np.einsum('ij,tpj->tpi', K.hat[m], pt_h)   # (num_points_in_m, 33, 4)

        # Dot with vertex values
        vv = vertex_val[K.simplices[m]]                 # (4,)
        q0 = np.einsum('i,tpi->tp', vv, hx)            # (num_points_in_m, 33)

        # Compute aux with bubble functions
        aux = bK[points_idx,:]*np.einsum('i,npi->np', betaKF[m], bF[points_idx])   # (num_points_in_m,33)

        # Assign to full arrays
        q0_all[points_idx] = q0
        aux_all[points_idx] = aux

    return q0_all + aux_all

def get_grad_morley_val(K,bF,bK,gF,gK,vertex_val,betaKF) :

    hat_mat = np.array(K.hat)[:,:,:3].transpose(0,2,1) #np.array(K.hat)[:, :3, :]  # Take first 3 rows -> (N_tet, 3, 4)

    # Vertex values: (N_tet, 4)
    vv = vertex_val[K.simplices]         # (N_tet, 4)

    N_tet = bK.shape[0]
    N_sub = bK.shape[1]
    N_pt = bK.shape[2]

    # Broadcast vertex values over N_sub elements and N_pt points
    vv_broadcast = np.broadcast_to(vv[:, None, None, :], (N_tet, N_sub, N_pt, 4))  # (N_tet, N_sub, N_pt, 4)
    # Broadcast hat matrices over N_sub elements
    hat_broadcast = np.broadcast_to(hat_mat[:, None, None, :, :], (N_tet, N_sub, N_pt, 3, 4))  # (N_tet, N_sub, N_pt, 3, 4)
    # Multiply: gradq0 = hat_mat @ vv
    grad_q0 = np.einsum('tnilj,tnij->tnil', hat_broadcast, vv_broadcast)  # (N_tet, N_sub, N_pt, 3)
        
    # Compute grad contributions
    # (grad_bF * bK[..., None]) + (grad_bK[..., None, :] * bF[..., :, None])
    term1 = gF * bK[..., None, None]                # (N_tet, N_sub, N_pt, 4, 3)
    term2 = gK[..., None, :] * bF[..., :, None]     # (N_tet, N_sub, N_pt, 4, 3)
    b_mat = term1 + term2                                # (N_tet, N_sub, N_pt, 4, 3)

    # Contract with betaKF over last axis of 4
    # betaKF: (N_tet, 4)
    val = np.einsum('ti,tkpil->tkpl', betaKF, b_mat)        # (N_tet, N_sub, N_pt, 3)

    return grad_q0 + val

def get_grad_morley_val_primal(K, bK, gK, bF, vertex_val, betaKF):
# Compute Morley gradient contributions assuming:

    hat_mat = np.array(K.hat)[:,:,:3].transpose(0,2,1)  # (N_tet, 3, 4)
    vv = vertex_val[K.simplices]         # (N_tet, 4)
    
    N_tet, N_pt = bK.shape

    # Compute grad_q0 = hat_mat @ vertex values 
    # hat_mat: (N_tet, 3, 4), vv: (N_tet, 4)
    # Broadcast vv to (N_tet, N_pt, 4)
    vv_broadcast = np.broadcast_to(vv[:, None, :], (N_tet, N_pt, 4))
    # Compute grad_q0: (N_tet, N_pt, 3)
    grad_q0 = np.einsum('tij,tkj->tki', hat_mat, vv_broadcast)

    # Compute contribution from gK * bF
    # gF = 0, so first term drops
    # gK: (N_tet, N_pt, 3), bF: (N_tet, N_pt, 4)
    # Expand dims for broadcasting: gK[..., None] * bF[..., :, None]
    b_mat = gK[..., None,:] * bF[..., :, None]  # (N_tet, N_pt, 4, 3)

    # Contract with betaKF (N_tet, 4) over axis 2
    val = np.einsum('ti,tpij->tpj', betaKF, b_mat)  # (N_tet, N_pt, 3)

    # Combine contributions
    result = grad_q0 + val  # (N_tet, N_pt, 3)

    return grad_q0,val

def get_grad_morley_val_face(K,bF,gK,vertex_val,betaKF,face_to_tet,loc_face_ind) :

    N_face, N_pt = bF.shape
    betaKF = np.asarray(betaKF)

    hat_mat = np.array(K.hat)[:, :, :3]  # Take first 3 columns -> (N_tet, 4, 3), CORRECT TO USE COLUMNS NOT ROWS?!
    # Vertex values:
    vv = vertex_val[K.simplices]    # (N_tet, 4), USE LEAST SQUARES APPROACH TO FOR VV?!
    # Multiply: gradq0 = hat_mat @ vv
    grad_q0 = np.einsum('tin,ti->tn', hat_mat, vv)  # (N_tet, 3)

    # Compute bubble grad contributions
    val = np.zeros([N_face,2,N_pt,3])
    for i in range(2) :
        # (grad_bF * bK[..., None]) + (grad_bK[..., None, :] * bF[..., :, None])
        # Note gF * bK = 0

        mask1 = face_to_tet[:,i]
        mask2 = loc_face_ind[:,i]
        b_mat = gK[:, i, ...] * bF[...,None]   

        # Contract with betaKF over last axis of 4
        # betaKF: (N_tet, 4)
        bubble_contr = b_mat * betaKF[mask1,mask2][:,None,None]
        
        val[:,i,...] =  grad_q0[mask1,None,:] + bubble_contr   #grad_q0[mask1,None,:] #+ bubble_contr  

    return val

def get_lap_morley_val(bF,bK,gF,gK,lapF,lapK,betaKF) :

    # lapF * bK[..., None]
    term1 = lapF * bK[..., None]

    # lapK[..., None] * bF
    term2 = lapK[..., None] * bF

    # 2 * dot(grad_bF_i, grad_bK)
    term3 = 2 * np.einsum('tknlj,tknlj->tknl', gF, gK[..., None, :])  # (N_tet, 12, N_pt, 4)

    # Combine
    b_vec = term1 + term2 + term3  # (N_tet, 12, N_pt, 4)

    # Contract with betaKF
    val = np.einsum('ti,tkni->tkn', betaKF, b_vec)  # (N_tet, 12, N_pt)

    return val



# -------------------------------------- A POSTERIORI ERROR ESTIMATOR --------------------------------------------

def get_conv_terms(K_inter,vv,rho,morley_inter_faces) :
    # Precompute common arrays
    num_faces = K_inter.num_faces
    grad_c_aux = vv[K_inter.face_ident_dual[:, 0]]   # (num_faces, 3)
    dual_id = K_inter.face_ident_dual[:, 1]             # (num_faces,)
    primal0 = K_inter.face_ident_primal[:, 0]
    primal1 = K_inter.face_ident_primal[:, 1]
    normals = K_inter.face_normals                        # (num_faces, 3)
    morley = morley_inter_faces                            # (num_faces,)

    # safe rho for log
    aux_rho_safe = np.maximum(rho, 1e-20)

    # Masks
    mask_dual_neg = dual_id < 0
    mask_dual_pos = ~mask_dual_neg

    val = np.zeros([num_faces, morley.shape[1]], dtype=grad_c_aux.dtype)

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
        
        # Compute F_E_over_areaE vectorized
        F = np.zeros_like(rho1)
        mask_zero = (rho1 < 1e-12) | (rho2 < 1e-12)
        mask_continuous = (~mask_zero) & (np.abs(rho1 - rho2) < 1e-12)
        mask_logmean = (~mask_zero) & (~mask_continuous)
        
        F[mask_continuous] = rho1[mask_continuous]
        F[mask_logmean] = (rho1[mask_logmean] - rho2[mask_logmean]) / \
                        (np.log(rho1_safe[mask_logmean]) - np.log(rho2_safe[mask_logmean]))
        
        # Compute aux using einsum
        grad_neg = grad_c_aux[mask_dual_neg]
        normals_neg = normals[mask_dual_neg]
        morley_neg = morley[mask_dual_neg]
        
        val[mask_dual_neg] = np.einsum('ij,ij->i', grad_neg, normals_neg)[:,np.newaxis] * (morley_neg - F[:,np.newaxis])

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
        val[mask_dual_pos] = jump_c[:,np.newaxis] * morley_pos

    return val

