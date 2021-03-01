# -*- coding: utf-8 -*-
import numpy as np
from dolfin import *
from time   import time, sleep
import pickle
from scipy import sparse
from scipy.sparse import linalg as sparseLinalg
from scipy import linalg as spla
from scipy.sparse import csr_matrix,coo_matrix
from petsc4py import PETSc
from tools import msh_to_xdmf

''' 
28.02.2021

Solve darcy equation using affine decomposition of the effective permeability tensor J.T * inv(K) * J / det(J)
affine decompostion is realized by seeing the tensor components as an additional dimension
forked from geometry_DEIM.py

'''
def make_mesh(file):
    msh_to_xdmf(file,"Meshes")

def getJacobian(P1, P2):

    x1 = P1[:,0]
    y1 = P1[:,1]
    z1 = P1[:,2]
    
    x2 = P2[:,0]
    y2 = P2[:,1]
    z2 = P2[:,2]

    dx = x1[1:] - x1[0]
    dy = y1[1:] - y1[0]
    dz = z1[1:] - z1[0]
    dX = x2[1:] - x2[0]
    dY = y2[1:] - y2[0]
    dZ = z2[1:] - z2[0]

    M = np.zeros([9,9])

    M[0,:] = np.array([dx[0],dy[0],dz[0],0,0,0,0,0,0])
    M[1,:] = np.array([0,0,0,dx[0],dy[0],dz[0],0,0,0])
    M[2,:] = np.array([0,0,0,0,0,0,dx[0],dy[0],dz[0]])
    M[3,:] = np.array([dx[1],dy[1],dz[1],0,0,0,0,0,0])
    M[4,:] = np.array([0,0,0,dx[1],dy[1],dz[1],0,0,0])
    M[5,:] = np.array([0,0,0,0,0,0,dx[1],dy[1],dz[1]])
    M[6,:] = np.array([dx[2],dy[2],dz[2],0,0,0,0,0,0])
    M[7,:] = np.array([0,0,0,dx[2],dy[2],dz[2],0,0,0])
    M[8,:] = np.array([0,0,0,0,0,0,dx[2],dy[2],dz[2]])

    b = np.array([dX[0],dY[0],dZ[0],dX[1],dY[1],dZ[1],dX[2],dY[2],dZ[2]])
    J = np.dot(np.linalg.inv(M),b)
    J = J.reshape([3,3])
    J = np.mat(J)

    return J, P1[0], P2[0]

def getJacobian2D(P1, P2):

    x1 = P1[:,0]
    y1 = P1[:,1]
    
    x2 = P2[:,0]
    y2 = P2[:,1]

    dx = x1[1:] - x1[0]
    dy = y1[1:] - y1[0]
    dX = x2[1:] - x2[0]
    dY = y2[1:] - y2[0]

    M = np.zeros([4,4])

    M[0,:] = np.array([ dx[0] , dy[0] , 0     , 0     ])
    M[1,:] = np.array([ 0     ,     0 , dx[0] , dy[0] ])
    M[2,:] = np.array([ dx[1] , dy[1] , 0     , 0     ])
    M[3,:] = np.array([ 0     ,     0 , dx[1] , dy[1] ])

    b = np.array([dX[0],dY[0],dX[1],dY[1]])
    J = np.dot(np.linalg.inv(M),b)
    J = J.reshape([2,2])

    return J

def norm(u): 
    return sqrt(dot(u,u))

def get_mesh():
    mesh_file = XDMFFile("Meshes/mesh.xdmf")
    mesh = Mesh()
    mesh_file.read(mesh)

    return mesh

def get_mesh(file):
    mesh_file = XDMFFile(file)
    mesh = Mesh()
    mesh_file.read(mesh)

    return mesh

def get_functionspace(mesh):
    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
    E   = MixedElement(BDM,DG)
    W   = FunctionSpace(mesh, E)

    return W

def mark_surfaces(file, mesh):
    subdomains_file = XDMFFile(file)
    subdomains_mesh = Mesh()
    subdomains_file.read(subdomains_mesh)

    subdomains = MeshValueCollection("size_t", mesh, 2)

    subdomains_file.read(subdomains)

    subdomains  = cpp.mesh.MeshFunctionSizet(mesh, subdomains)
    
    ds          = Measure('ds', domain=mesh, subdomain_data=subdomains)

    File("Output/sub_domains.pvd") << subdomains


    return ds, subdomains

def norm(u): 
    return sqrt(dot(u,u))

def get_reference_points():

    P = [np.array([0,0,0])]

    for i in range(6):
        phi = i*np.pi/3
        p1  = np.array([np.cos(phi),np.sin(phi),0.5])
        p2  = np.array([np.cos(phi),np.sin(phi),1.5])
        P.append(p1)
        P.append(p2)

    P.append([0,0,2])

    return P

def get_physical_points(P):
    
    r=.2
    U = [P[0]]

    for i in range(1,13):
        U.append(P[i] + np.random.random([3])*r)

    U.append(P[13])

    return U

def get_triangles(P,U):
    triangles = []
    triangles.append(Triangle(P[13], P[2], P[4],U[13], U[2], U[4]))
    triangles.append(Triangle(P[13], P[4], P[6],U[13], U[4], U[6]))
    triangles.append(Triangle(P[13], P[6], P[8],U[13], U[6], U[8]))
    triangles.append(Triangle(P[13], P[8], P[10],U[13], U[8], U[10]))
    triangles.append(Triangle(P[13], P[10], P[12],U[13], U[10], U[12]))
    triangles.append(Triangle(P[13], P[12], P[2],U[13], U[12], U[2]))
    triangles.append(Triangle(P[0], P[1], P[3],U[0], U[1], U[3]))
    triangles.append(Triangle(P[0], P[3], P[5],U[0], U[3], U[5]))
    triangles.append(Triangle(P[0], P[5], P[7],U[0], U[5], U[7]))
    triangles.append(Triangle(P[0], P[7], P[9],U[0], U[7], U[9]))
    triangles.append(Triangle(P[0], P[9], P[11],U[0], U[9], U[11]))
    triangles.append(Triangle(P[0], P[11], P[1],U[0], U[11], U[1]))
    
    return triangles

def get_triangle_expressions(triangles):
        
    triangle_expressions = []

    for tri in triangles:
        triangle_expressions.append(tri.solve())

    return triangle_expressions

def get_boundary_functions(triangle_expressions, triangles):

    boundary_functions = []

    for i in range(len(triangle_expressions)):
        g  = Boundary_function(triangle_expressions[i], triangles[i].P1, triangles[i].P2, triangles[i].P3,triangles[i].U1, triangles[i].U2, triangles[i].U3)
        boundary_functions.append(g)

    return boundary_functions

def get_interpolations(boundary_functions, V):

    interpolations = []

    for g in boundary_functions:
        g = interpolate(g, V)
        interpolations.append(g)
    
    return interpolations

def get_boundary_conditions(interpolations, V, subdomains):
    bcs = []
    for i in range(len(interpolations)):
        bc = DirichletBC(V, interpolations[i], subdomains, i+2)
        bcs.append(bc)

    return bcs

class Ortho(UserExpression):

    def __init__(self,K,X = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.X = X
        self.K = K

    def eval(self, value, x):
        if self.X != True: 
            p = x + self.X(x)
        else: p = x
        phi = np.arctan2(p[1],p[0])

        R = np.mat(  (( cos(phi) , -sin(phi) , 0 ),  
                      ( sin(phi) ,  cos(phi) , 0 ),  
                      ( 0        , 0         , 1 )))
        
        K = R*self.K*R.T

        value[0] = K[0,0]
        value[1] = K[0,1]
        value[2] = K[0,2]
        value[3] = K[1,0]
        value[4] = K[1,1]
        value[5] = K[1,2]
        value[6] = K[2,0]
        value[7] = K[2,1]
        value[8] = K[2,2]

    def value_shape(self):
        return (3,3)

class Boundary_function(UserExpression):
    def __init__(self, f, p1, p2, p3, u1, u2, u3):
        super().__init__() 
        self.f   = f
        self.v1  = p2 - p1  
        self.v2  = p3 - p1
        self.w1  = u2 - u1  
        self.w2  = u3 - u1  
        self.p1  = p1
        self.u1  = u1

        vT       = self.v2 - (np.dot(self.v1,self.v2) / np.dot(self.v1,self.v1)) * self.v1
        self.e_x = self.v1 / np.linalg.norm(self.v1)
        self.e_y = vT / np.linalg.norm(vT)
        self.e_z = np.cross(self.e_x, self.e_y)

        wT       = self.w2 - (np.dot(self.w1,self.w2) / np.dot(self.w1,self.w1)) * self.w1
        self.E_x = self.w1 / np.linalg.norm(self.w1)
        self.E_y = wT / np.linalg.norm(wT)
        self.E_z = np.cross(self.E_x, self.E_y)

    def value_shape(self): 
        return (3, )

    def eval(self, values, x):

        if abs(np.dot(self.e_z,x-self.p1)) < .0001: # check if x is on plane
            
            length = np.linalg.norm(x-self.p1) 
            if length > 0:
                phi     = self.get_angle(self.v1, x - self.p1).item(0) 
                x1      = cos(phi)*length # project 3D point onto 2D plane
                y1      = sin(phi)*length # project 3D point onto 2D plane

            else:
                x1 = 0
                y1 = 0
            
            FF    = self.f([x1,y1]) # get f(x): R² -> R²
            fx = FF[0]
            fy = FF[1]
            vals = (x1+fx)*self.E_x + (y1+fy)*self.E_y - x1*self.e_x - y1*self.e_y # project 2D function into 3D

            values[0] = vals[0]
            values[1] = vals[1]
            values[2] = vals[2]

    def get_angle(self, v1, v2):
    
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)

        cos = np.dot(v1,v2)
        cos = min(cos,1)
        cos = max(cos,-1)
        phi = np.arccos(cos)

        return phi

class Triangle:
    def __init__(self, P1, P2, P3, U1, U2, U3):
        self.U1  = U1
        self.U2  = U2
        self.U3  = U3
        self.P1  = P1
        self.P2  = P2
        self.P3  = P3
        self.U   = self.collapse_dimensionU()
        self.P   = self.collapse_dimensionP()
        self.J   = getJacobian2D(self.P,self.U)

    def collapse_dimensionU(self):

        v1 = self.U2 - self.U1
        v2 = self.U3 - self.U1

        p0 = np.array([0,0])
        p1 = np.array([np.linalg.norm(v1), 0])

        cos = np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))
        cos = min(cos,1)
        cos = max(cos,-1)
        alpha = np.arccos(cos)
        length = np.linalg.norm(v2)

        p2 = np.array([np.cos(alpha)*length, np.sin(alpha)*length])
        
        U = np.array([p0,p1,p2]).reshape([3,2])
        
        return U

    def collapse_dimensionP(self):

        v1 = self.P2 - self.P1
        v2 = self.P3 - self.P1

        p0 = np.array([0,0])
        p1 = np.array([np.linalg.norm(v1), 0])

        cos = np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))
        cos = min(cos,1)
        cos = max(cos,-1)
        alpha = np.arccos(cos)
        length = np.linalg.norm(v2)

        p2 = np.array([np.cos(alpha)*length, np.sin(alpha)*length])
        
        P = np.array([p0,p1,p2]).reshape([3,2])
        
        return P

    def solve(self):
        J = self.J - np.eye(2)

        g = Expression(('a*x[0] + b*x[1]', 'c*x[0] + d*x[1]'),
                        a  = Constant(J[0,0]),
                        b  = Constant(J[0,1]),
                        c  = Constant(J[1,0]),
                        d  = Constant(J[1,1]), degree=2)

        return g

def run_simulation():

    ##################################
    ########## GET JACOBIAN ##########
    ##################################

    P                    = get_reference_points()
    U                    = get_physical_points(P)
    triangles            = get_triangles(P,U)
    triangle_expressions = get_triangle_expressions(triangles)
    boundary_functions   = get_boundary_functions(triangle_expressions, triangles)

    mesh_lobule = get_mesh()

    E           = VectorElement('CG',tetrahedron,1)
    V_lobule    = FunctionSpace(mesh_lobule, E)
    u           = TrialFunction(V_lobule)
    v           = TestFunction(V_lobule)

    ds, subdomains = mark_surfaces(surface_file, mesh_lobule)
    interpolations = get_interpolations(boundary_functions, V_lobule)
    bcs            = get_boundary_conditions(interpolations, V_lobule, subdomains)

    a  = inner(grad(u), grad(v))*dx
    L  = dot(v,Constant((0,0,0)))*dx

    X   = Function(V_lobule)

    solve(a == L, X, bcs,solver_parameters={'linear_solver': 'mumps'})

    ### GET JACOBIAN ###

    E_J = TensorElement('DG',tetrahedron,0)
    V_J = FunctionSpace(mesh_lobule, E_J)
    I   = Constant(((1,0,0),(0,1,0), (0,0,1)))
    J   = project(grad(X)+I, V_J)

    ##################
    ### DARCY FLOW ###
    ##################

    W    = get_functionspace(mesh_lobule)
    u,p  = TrialFunctions(W)
    v,q  = TestFunctions(W)
    n    = FacetNormal(mesh_lobule)
    I    = np.eye(3)

    k_r, k_phi, k_z = 1,1,1

    E_K     = TensorElement('CG',tetrahedron,1)
    V_K     = FunctionSpace(mesh_lobule, E_K)
    K_ortho = np.mat([(k_r,0,0),(0,k_phi,0),(0,0,k_z)])
    K       = Ortho(K_ortho, X)
    K       = interpolate(K, V_K)

    ds, subdomains = mark_surfaces(surface_file, mesh_lobule)

    a  = (dot(inv(K)*J*u, J*v/det(J)) - div(v)*p - div(u)*q)*dx
    L  = Constant(0)*dot(n,v)*ds(14) - Constant(1)*dot(n,v)*ds(15)
    bc = DirichletBC(W.sub(0), Constant((0,0,0)), subdomains, 16) 

    w = Function(W)

    solve(a == L, w ,bc, solver_parameters = {'linear_solver': 'mumps'})

    return w

#######################
### DEIM FUNCTIONS ####
#######################

def get_tensor_field(mesh, T):
    N = len(mesh.cells())*4
    tensor_field = T.vector()[:].reshape(N,3,3)

    return tensor_field

def get_frobenius_norm(M):
    # validated
    frob_norm = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
            frob_norm[i] = np.sqrt(np.sum(np.power(M[i],2)))

    return frob_norm

def tensor_field_POD(mesh,Ks,n):
    S = []
    
    for K in Ks:
        t = get_tensor_field(mesh, K)
        t = t.reshape(t.size)
        S.append(t)

    S       = np.mat(S).T 
    U,Sig,Z = np.linalg.svd(S, full_matrices = False)
    U       = np.array(U)
    Q       = []
    MP      = []
    
    for b in range(n):
        Q.append( U[:,b].reshape([-1,9])/(max(abs(U[:,b]))))   
        MP.append(find_new_magic_point(Q,MP))

    return Q,MP

def find_new_magic_point(Q_tot,MP):
    Q = Q_tot[0:-1] 

    truesol = Q_tot[-1]
    approxsol = interpolate2(truesol,Q,MP)
    error = truesol-approxsol
    MPnew = np.unravel_index(abs(error).argmax(), error.shape)

    return MPnew

def interpolate4(solution,Q,MP):
    approxsol = np.zeros(solution.shape)
    B = np.zeros( (len(Q)*3,len(Q)*3) )
    F = np.zeros( (len(Q)*3, 3 ) )
    for r,i in enumerate(MP): 
        for c,q in enumerate(Q): 
            B[3*r:3*(r+1),3*c:3*(c+1)] = q[i] # value of r-th mode at position i equals entry of B
        F[3*r:3*(r+1),:] = solution[i]
    U = np.dot(np.linalg.inv(B),F)

    for i in range(len(Q)):
        u = U[3*i:3*(i+1),:]
        approxsol += np.matmul(Q[i],u)    
    return approxsol

def interpolate2(solution,Q,MP):
    approxsol = np.zeros(solution.shape)
    B = np.zeros( (len(Q),len(Q)) )
    F = np.zeros( (len(Q), 1 ) )
    for r,i in enumerate(MP): 
        for c,q in enumerate(Q): 
            B[r,c] = q[i] 
        F[r] = solution[i]
    U = np.dot(np.linalg.inv(B),F)
    for nr,u in enumerate(U):
        approxsol += Q[nr]*float(u) 
    return approxsol

def decompose_tensor_field(Q,MP,mesh, K):
    
    t = get_tensor_field(mesh, K)

    t = t.reshape([-1,9])

    B = np.zeros( (len(Q),len(Q)) )
    F = np.zeros( (len(Q), 1 ) )
    
    for r,(i,j) in enumerate(MP): 
        for c,q in enumerate(Q): 
            B[r,c] = q[i,j] 
        F[r] = t[i,j]

    U = np.dot(np.linalg.inv(B),F)
    U = U.reshape(U.size)

    return U

def decompose_tensor_field_old(Q,MP,mesh, K):

    C = []
    t = get_tensor_field(mesh, K)

    B = np.zeros( (len(Q)*3,len(Q)*3) )
    F = np.zeros( (len(Q)*3, 3 ) )
    for r,i in enumerate(MP): 
        for c,q in enumerate(Q): 
            B[3*r:3*(r+1),3*c:3*(c+1)] = q[i] # value of i-th mode at position x,y equals entry of B
        F[3*r:3*(r+1),:] = t[i]
    U = np.dot(np.linalg.inv(B),F)

    for i in range(len(Q)):
        c = U[3*i:3*(i+1),:]
        C.append(c)

    return C

def array_to_dolfin_function(array, mesh, space):
    u = Function(space)
    print(array.shape)
    print(len(u.vector()))
    u.vector()[:] = array.reshape(array.size)

    return u

def assemble_tensor_field(Q, C):

    T1 = np.zeros_like(Q[0][0])
    T2 = np.zeros_like(Q[0][0])
    T3 = np.zeros_like(Q[0][0])
    T4 = np.zeros_like(Q[0][0])
    T5 = np.zeros_like(Q[0][0])
    T6 = np.zeros_like(Q[0][0])

    T = [T1, T2, T3, T4, T5, T6]
    
    for i in range(6):
        for j in range(len(Q[0])):
            T[i] += C[i][j] * Q[i][j]

    tensor_field = np.zeros([Q[0][0].shape[0],3,3])
    
    tensor_field[:,0,0] = T[0]    
    tensor_field[:,0,1] = T[1]    
    tensor_field[:,1,0] = T[1]    
    tensor_field[:,0,2] = T[2]    
    tensor_field[:,2,0] = T[2]    
    tensor_field[:,1,1] = T[3]
    tensor_field[:,1,2] = T[4]
    tensor_field[:,2,1] = T[4]
    tensor_field[:,2,2] = T[5]
    

    return tensor_field 

def assemble_weighted_mode_old(Q_i, C_i):

    T1 = np.zeros_like(Q_i[0])
    T2 = np.zeros_like(Q_i[0])
    T3 = np.zeros_like(Q_i[0])
    T4 = np.zeros_like(Q_i[0])
    T5 = np.zeros_like(Q_i[0])
    T6 = np.zeros_like(Q_i[0])

    T = [T1, T2, T3, T4, T5, T6]
    
    for i in range(6):
            T[i] += C_i[i] * Q_i[i]

    tensor_field = np.zeros([Q_i[0].shape[0],3,3])
    
    tensor_field[:,0,0] = T[0]    
    tensor_field[:,0,1] = T[1]    
    tensor_field[:,1,0] = T[1]    
    tensor_field[:,0,2] = T[2]    
    tensor_field[:,2,0] = T[2]    
    tensor_field[:,1,1] = T[3]
    tensor_field[:,1,2] = T[4]
    tensor_field[:,2,1] = T[4]
    tensor_field[:,2,2] = T[5]
    
    return tensor_field 

def assemble_weighted_modes_old2(Q_i, C_i):
    pass

def assemble_weighted_modes(Q, C):


    # print(len(Q)) # Liste der 6 Komponenten, wobei diese per Liste unterteilt ist in n Modes
    # print(len(Q[0])) # n Modes für eine tensor komponente 
    # print(Q[0][0].shape) # array über feld für mode einer komponente

    TFs = []
    
    for j in range(len(Q[0])):
        T = [np.zeros_like(Q[0][0]) for i in range(6)]
        for i in range(6):
            T[i] = C[i][j] * Q[i][j] # i-te komponente, j-te mode

        tf = np.zeros([Q[0][0].shape[0],3,3])
            
        tf[:,0,0] = T[0]    
        tf[:,0,1] = T[1]    
        tf[:,1,0] = T[1]    
        tf[:,0,2] = T[2]    
        tf[:,2,0] = T[2]    
        tf[:,1,1] = T[3]
        tf[:,1,2] = T[4]
        tf[:,2,1] = T[4]
        tf[:,2,2] = T[5]

        TFs.append(tf)
    

    return TFs

def assemble_modes(Q):

    # print(len(Q)) # Liste der 6 Komponenten, wobei diese per Liste unterteilt ist in n Modes
    # print(len(Q[0])) # n Modes für eine tensor komponente 
    # print(Q[0][0].shape) # array über feld für mode einer komponente

    TFs = []
    
    for j in range(len(Q[0])):
        T = [np.zeros_like(Q[0][0]) for i in range(6)]
        for i in range(6):
            T[i] = Q[i][j] # i-te komponente, j-te mode

        tf = np.zeros([Q[0][0].shape[0],3,3])
            
        tf[:,0,0] = T[0]    
        tf[:,0,1] = T[1]    
        tf[:,1,0] = T[1]    
        tf[:,0,2] = T[2]    
        tf[:,2,0] = T[2]    
        tf[:,1,1] = T[3]
        tf[:,1,2] = T[4]
        tf[:,2,1] = T[4]
        tf[:,2,2] = T[5]

        TFs.append(tf)
    

    return TFs

def assemble_weights(C):

    TFs = []
    
    for j in range(len(C[0])):
        T = [np.zeros_like(C[0][0]) for i in range(6)]
        for i in range(6):
            T[i] = C[i][j] # i-te komponente, j-te mode

        tf = np.zeros([3,3])
            
        tf[0,0] = T[0]    
        tf[0,1] = T[1]    
        tf[1,0] = T[1]    
        tf[0,2] = T[2]    
        tf[2,0] = T[2]    
        tf[1,1] = T[3]
        tf[1,2] = T[4]
        tf[2,1] = T[4]
        tf[2,2] = T[5]

        TFs.append(tf)
    

    return TFs

def load_dolfin_functions(file, space, n = -1):
    
    data = []

    with open(file, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass

    if n == -1: n = len(data)
    
    functions = []

    for i in range(n):
        u = Function(space)
        u.vector()[:] = data[i]
        functions.append(u)

    return functions

def get_decomposed_matrices(U,modes,mesh,space):
    
    print("get decomposed matrices...")

    matrices = []

    I11 = Constant(((1,0,0),(0,0,0),(0,0,0)))
    I22 = Constant(((0,0,0),(0,1,0),(0,0,0)))
    I33 = Constant(((0,0,0),(0,0,0),(0,1,0)))
    I12 = Constant(((0,1,0),(1,0,0),(0,0,0)))
    I13 = Constant(((0,0,1),(0,0,0),(1,0,0)))
    I23 = Constant(((0,0,0),(0,0,1),(0,1,0)))

    for i in range(len(modes)):
        mode = modes[i]
        K    = array_to_dolfin_function(mode, mesh, space)# (modes[i], mesh_1x, V_1x)
        K.set_allow_extrapolation(True)
        K = project(K, DG1_2x)

        a1   = (dot(elem_mult(I11,K)*u,v))*dx 
        a1   = assemble(a1)
        a1   = DolfinToScipy(a1)
        a1_r = U.transpose().dot(a1)
        a1_r = a1_r.dot(U)

        a2   = (dot(elem_mult(I12,K)*u,v))*dx 
        a2   = assemble(a2)
        a2   = DolfinToScipy(a2)
        a2_r = U.transpose().dot(a2)
        a2_r = a2_r.dot(U)

        a3   = (dot(elem_mult(I13,K)*u,v))*dx 
        a3   = assemble(a3)
        a3   = DolfinToScipy(a3)
        a3_r = U.transpose().dot(a3)
        a3_r = a3_r.dot(U)

        a4   = (dot(elem_mult(I22,K)*u,v))*dx 
        a4   = assemble(a4)
        a4   = DolfinToScipy(a4)
        a4_r = U.transpose().dot(a4)
        a4_r = a4_r.dot(U)

        a5   = (dot(elem_mult(I23,K)*u,v))*dx 
        a5   = assemble(a5)
        a5   = DolfinToScipy(a5)
        a5_r = U.transpose().dot(a5)
        a5_r = a5_r.dot(U)

        a6   = (dot(elem_mult(I33,K)*u,v))*dx 
        a6   = assemble(a6)
        a6   = DolfinToScipy(a6)
        a6_r = U.transpose().dot(a6)
        a6_r = a6_r.dot(U)

        a7 = -(+ div(v)*p + div(u)*q)*dx
        a7   = assemble(a7)
        a7   = DolfinToScipy(a7)
        a7_r = U.transpose().dot(a7)
        a7_r = a7_r.dot(U)

        matrices.append([a1_r,a2_r,a3_r,a4_r,a5_r,a6_r,a7_r])

    return matrices

def get_decomposed_matrices2(U,modes,mesh,space):
    
    print("get decomposed matrices...")

    matrices = []
    for i in range(len(modes)):
        mode = modes[i]
        K    = array_to_dolfin_function(mode, mesh, space)# (modes[i], mesh_1x, V_1x)
        K.set_allow_extrapolation(True)
        K = project(K, DG1_2x)

        a1   = (dot(K*u,v))*dx 
        a1   = assemble(a1)
        a1   = DolfinToScipy(a1)
        a1_r = U.transpose().dot(a1)
        a1_r = a1_r.dot(U)

        a2 = -(+ div(v)*p + div(u)*q)*dx
        a2   = assemble(a2)
        a2   = DolfinToScipy(a2)
        a2_r = U.transpose().dot(a2)
        a2_r = a2_r.dot(U)

        matrices.append([a1_r,a2_r])
    return matrices

def DolfinToScipy(M,shape=None):
    M_ = as_backend_type(M).mat()
    M_csr = csr_matrix(M_.getValuesCSR()[::-1],shape=M_.size)
    if shape != None:
        M_csr.resize(shape)
    return M_csr

def load_raw_data(file, n = -1):
    
    data = []

    with open(file, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass

    if n == -1: n = len(data)

    return data[:n]

def load_modes(file):
    with open(file, 'rb') as fr:
        U = pickle.load(fr)

    return U

def NumpyToDolfin(M):
    U = numpyToScipy(M)
    U = ScipyToPETSc(U)
    U = PETScToDolfin(U)
    return U

def numpyToScipy(M):
    return sparse.csr_matrix(M)

def PETScToDolfin(M):
    return Matrix(M)

def ScipyToPETSc(M):
    M = M.tocsr()
    IM = M.indptr
    JM = M.indices
    DM = M.data
    shape = M.shape
    M_ = PETSc.Mat().createAIJWithArrays(shape,(IM,JM,DM))
    return PETScMatrix(M_)

def assemble_tensor_field(Q, C):

    T = np.zeros_like(Q[0])
    
    for i in range(len(Q)):
        T += C[i] * Q[i]

    T = T.reshape([-1,3,3])

    return T

#############
### FILES ###
#############

surface_file_1x        = "Meshes/1x/subdomains.xdmf"
file_1x                = "Meshes/1x/mesh.xdmf"
surface_file_2x        = "Meshes/2x/subdomains.xdmf"
file_2x                = "Meshes/2x/mesh.xdmf"
solution_modes         = 'Data/solution_modes.p'
permeability_snapshots = 'Data/permeability_snapshots.p'
t1                     = time()

##################################
##################################
##################################

mesh_1x = get_mesh(file_1x)
mesh_2x = get_mesh(file_2x)
E       = TensorElement('DG',tetrahedron,1)
DG1_1x  = FunctionSpace(mesh_1x, E)
CG1_1x  = TensorFunctionSpace(mesh_1x, 'CG', 1, shape = (3,3))
DG1_2x  = FunctionSpace(mesh_2x, E)
V_div   = FunctionSpace(mesh_2x, 'CG', 1)
Ks      = load_dolfin_functions(permeability_snapshots, DG1_1x,20)

############
### DEIM ###
############

n_tensor_modes   = 5
n_solution_modes = 8
tensor_field     = get_tensor_field(mesh_1x,Ks[0])
Q,MP             = tensor_field_POD(mesh_1x, Ks, n_tensor_modes)
C                = decompose_tensor_field(Q, MP, mesh_1x, Ks[0])
approx           = assemble_tensor_field(Q,C)

###############################
### VARIATIONAL FORMULATION ###
###############################

W              = get_functionspace(mesh_2x)
u,p            = TrialFunctions(W)
v,q            = TestFunctions(W)
n              = FacetNormal(W)
ds, subdomains = mark_surfaces(surface_file_2x, mesh_2x)
U              = load_modes(solution_modes)
U_             = U
U              = U[:,:n_solution_modes]
U              = numpyToScipy(U)
M              = get_decomposed_matrices2(U, Q, mesh_1x, DG1_1x)
A              = csr_matrix((n_solution_modes, n_solution_modes), dtype=np.float)

for i in range(n_tensor_modes):
    c = C[i]
    a1,a2 = M[i]
    a = c*a1
    A += a

A += a2

L   = Constant(0)*dot(n,v)*ds(14) - Constant(1)*dot(n,v)*ds(15)
L   = assemble(L)

L   = L.get_local()
L   = numpyToScipy(L).transpose()
L_r = U.transpose().dot(L)

w_r = sparseLinalg.spsolve(A, L_r)

w_approx        = U.dot(w_r)
w_r             = Function(W)
w_r.vector()[:] = list(w_approx)
u_r, p_r        = w_r.split()

File("Output/reduced_p.pvd")<<p_r

error = tensor_field - approx
error = np.abs(error)
error = error.reshape(error.size)
mean_error = np.mean(error)

print("mean error =", mean_error)

t2 = time() - t1
t2 = round(t2, 3)

print("time =",t2,"s")



