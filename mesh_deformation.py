import numpy as np
from dolfin import *
from time   import time
import pickle
from time import sleep

'''

28.02.2021

Peforms deformation of mesh, by moving up to 14 nodes.
Deformation is done solving Poisson Equation R³ -> R³.

'''

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

def mark_surfaces(subdomains_file, sublines_file, mesh):
    
    subdomains_file = XDMFFile(subdomains_file)
    subdomains_mesh = Mesh()
    subdomains_file.read(subdomains_mesh)
    subdomains      = MeshValueCollection("size_t", mesh, 2)
    subdomains_file.read(subdomains)
    subdomains      = cpp.mesh.MeshFunctionSizet(mesh, subdomains)
    
    sublines_file = XDMFFile(sublines_file)
    sublines      = MeshValueCollection("size_t", mesh, 1)
    sublines_file.read(sublines)
    sublines      = cpp.mesh.MeshFunctionSizet(mesh, sublines)

    return subdomains,sublines

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
    
    r=.1
    U = [P[0]]
    U.append(P[1] + np.array([1,0,0])) 

    for i in range(2,13):
        s = np.random.random(3)*.0
        s[2]=0
        U.append(P[i]  + s)
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

def get_boundary_function_3D(p1_,p2_,p1,p2):

    x1  = p1[0]
    y1  = p1[1]
    z1  = p1[2]

    x2  = p2[0]
    y2  = p2[1]
    z2  = p2[2]

    x1_ = p1_[0]
    y1_ = p1_[1]
    z1_ = p1_[2]

    x2_ = p2_[0]
    y2_ = p2_[1]
    z2_ = p2_[2]

    S = sqrt((x2_ - x1_)**2 + (y2_ - y1_)**2 + (z2_ - z1_)**2)

    g = Expression(( ' x1-x[0]+ (  pow(pow(x[0] - x1_,2) + pow(x[1] - y1_,2) + pow(x[2] - z1_,2),.5)    )/S*(x2-x1)',
                     ' y1-x[1]+ (  pow(pow(x[0] - x1_,2) + pow(x[1] - y1_,2) + pow(x[2] - z1_,2),.5)    )/S*(y2-y1)', 
                     ' z1-x[2]+ (  pow(pow(x[0] - x1_,2) + pow(x[1] - y1_,2) + pow(x[2] - z1_,2),.5)    )/S*(z2-z1)'), 
                        x1  = Constant(x1),
                        x2  = Constant(x2),
               
                        y1  = Constant(y1),
                        y2  = Constant(y2),
               
                        z1  = Constant(z1),
                        z2  = Constant(z2),
               
                        x1_ = Constant(x1_),
                        y1_ = Constant(y1_),
                        z1_ = Constant(z1_),
               
                        S   = Constant(S),
                        degree=2)

    return g

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

class PointBC(SubDomain):
    def __init__(self, points):
        SubDomain.__init__(self)
        self.points = points

    def inside(self, x, on_boundary):
        for p in self.points:
            if ((x[0]-p[0])**2+(x[1]-p[1])**2+(x[2]-p[2])**2)<.001:
                return True

        return False

def get_line_points(mesh,sublines, n):
    line = np.where(sublines.array() == n)
    edges_ = edges(mesh)
    edges_ = list(edges_)
    edges_ = np.array(edges_)
    points = edges_[line]

    P = []

    for edge in points:
        p=edge.entities(0)
        P.append(p[0])
        P.append(p[1])

    P = list(set(P))
    
    coos = []

    for p in P:
        c = mesh.coordinates()[p]
        coos.append(c)

    return coos

def get_line_bc(mesh,gs,V, sublines,n1, n2):


    P = get_line_points(mesh, sublines, n1)

    p1_ = np.array([P[0][0], P[0][1], .5])
    p2_ = np.array([P[0][0], P[0][1], 1.5])
    p1 =  gs[n2+6](p1_) + p1_ # unten letzte
    p2 =  gs[n2](p2_) + p2_ # oben letzte
    
    g = get_boundary_function_3D(p1_, p2_, p1, p2)
    pbc = PointBC(P)
    bc = DirichletBC(V, g, pbc,method='pointwise')

    return bc

def get_line_bcs(mesh, interpolations, space, sublines, line_tags):
    bcs = []

    for (i,j) in line_tags:
        bc = get_line_bc(mesh, interpolations, space, sublines, i, j)
        bcs.append(bc)

    return bcs

def move_nodes(mesh, X):
    for i in range(len(mesh.coordinates()[:])):
        p = mesh.coordinates()[i]
        mesh.coordinates()[i] += X(p)

    return mesh

def get_deformation(mesh):

    P                    = get_reference_points()
    U                    = get_physical_points(P)
    triangles            = get_triangles(P,U)
    triangle_expressions = get_triangle_expressions(triangles)
    boundary_functions   = get_boundary_functions(triangle_expressions, triangles)
    line_tags            = [[17,5], [18,0], [19,0], [20,1], [21,1], [22,2], [23,2], [24,3], [25,3], [26,4], [27,4], [28,5]]
    subdomains,sublines  = mark_surfaces(surface_file, lines_file, mesh)

    E           = VectorElement('CG',tetrahedron,1)
    V           = FunctionSpace(mesh, E)
    u           = TrialFunction(V)
    v           = TestFunction(V)

    interpolations      = get_interpolations(boundary_functions, V)
    bcs                 = get_boundary_conditions(interpolations, V, subdomains)
    bc                  = get_line_bcs(mesh,interpolations,V, sublines, line_tags)
    bcs                 = bcs + bc

    ##############################
    ### SOLVE POISSON EQUATION ###
    ##############################

    a  = inner(grad(u), grad(v))*dx
    L  = dot(v,Constant((0,0,0)))*dx
    X  = Function(V)

    solve(a == L, X, bcs,solver_parameters={'linear_solver': 'mumps'})

    return X

########################
########################
########################

file_square  = "Meshes/1x/mesh.xdmf"
file_round   = "Meshes/round_2x/mesh.xdmf"
surface_file = "Meshes/1x/subdomains.xdmf"
lines_file   = "Meshes/1x/sublines.xdmf"
mesh_square  = get_mesh(file_square)
mesh_round   = get_mesh(file_round)
solutions    = []
t1           = time()

X = get_deformation(mesh_square)
X.set_allow_extrapolation(True)

mesh_stretched = move_nodes(mesh_round, X)
    
File("Output/stretched_mesh.pvd") << mesh_stretched

print("Elapsed time =",round(time()-t1,2), "s")
