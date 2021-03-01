# -*- coding: utf-8 -*-
import numpy as np
from dolfin import *
from time   import time
import pickle
from time import sleep


'''

17.02.2021

generates snapshots of the solution. 
Jacobian is loaded from permeability_snapshots.p
Used for validation POD on the darcy flow.

forked from job2.py

'''

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
        U.append(P[i] )
    s = np.random.random()*r
    U.append(P[13] + np.array([0,0,s])) 

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

        if len(data[i])==len(u.vector()):
            u.vector()[:] = data[i]
            functions.append(u)

        else:
            print("Array size mismatch.")
            print(str(len(data[i])) + " vs " + str(len(u.vector())))

    return functions

def run_simulation(Ks, mesh):
    
    ##################
    ### DARCY FLOW ###
    ##################

    W              = get_functionspace(mesh)
    DG1            = TensorFunctionSpace(mesh, 'DG',1)
    # K              = Ks[np.random.randint(0,len(Ks)-1)]
    K              = Ks[0]
    K.set_allow_extrapolation(True)
    K              = project(K,DG1)
    u,p            = TrialFunctions(W)
    v,q            = TestFunctions(W)
    n              = FacetNormal(W)
    ds, subdomains = mark_surfaces(surface_file, mesh)

    a  = (dot(K*u,v)- div(v)*p - div(u)*q)*dx # don't forget that K needs to be piecewise constant
    L  = Constant(0)*dot(n,v)*ds(14) - Constant(1)*dot(n,v)*ds(15)
    bc = DirichletBC(W.sub(0), Constant((0,0,0)), subdomains, 16) 
    w  = Function(W)

    solve(a == L, w ,bc, solver_parameters = {'linear_solver': 'mumps'})

    return w

#############
### FILES ###
#############

file_1x      = "Meshes/round_1x/mesh.xdmf"
surface_file = "Meshes/round_1x/subdomains.xdmf"
file_2x      = "Meshes/round_1x/mesh.xdmf"
mesh_1x      = get_mesh(file_1x)
mesh_2x      = get_mesh(file_2x)
solutions    = []
n            = 1
t1           = time()
DG1          = TensorFunctionSpace(mesh_1x, 'DG',1)
Ks           = load_dolfin_functions('Data/permeability_snapshots.p', DG1)



for i in range(n):
    w = run_simulation(Ks, mesh_2x)
    u,p = w.split()
    with open('Data/solution_snaptshots.p', 'ab+') as fp:
        pickle.dump(w.vector()[:],fp)


print(time()-t1)