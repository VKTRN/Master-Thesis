import numpy as np
from dolfin import *
from time   import sleep, time
from numpy  import array
from tools  import *

t1 = time()

def get_mesh(file):
    msh_to_xdmf(file,"Meshes")
    mesh_file = XDMFFile("Meshes/mesh.xdmf")
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

def mark_volumes(file):
    subvolumes_file = XDMFFile(file)
    subvolumes_mesh = Mesh()
    subvolumes_file.read(subvolumes_mesh)
    subvolumes = MeshValueCollection("size_t", mesh, 3)
    subvolumes_file.read(subvolumes)
    subvolumes = cpp.mesh.MeshFunctionSizet(mesh, subvolumes)
    File("Output/sub_volumes.pvd") << subvolumes

    dx = Measure('dx', domain = mesh, subdomain_data=subvolumes)

    return dx, subvolumes

def get_tetras_ref():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 1.5]

    c0 = [cos(0*pi/3), sin(0*pi/3), 0]
    d0 = [cos(0*pi/3), sin(0*pi/3), 1]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]
    d1 = [cos(1*pi/3), sin(1*pi/3), 1]
    c2 = [cos(2*pi/3), sin(2*pi/3), 0]
    d2 = [cos(2*pi/3), sin(2*pi/3), 1]
    c3 = [cos(3*pi/3), sin(3*pi/3), 0]
    d3 = [cos(3*pi/3), sin(3*pi/3), 1]
    c4 = [cos(4*pi/3), sin(4*pi/3), 0]
    d4 = [cos(4*pi/3), sin(4*pi/3), 1]
    c5 = [cos(5*pi/3), sin(5*pi/3), 0]
    d5 = [cos(5*pi/3), sin(5*pi/3), 1]


    tetras.append(array([a,b,c0,c1])) # mitte 1
    tetras.append(array([b,d0,d1,c1])) # mitte 2
    tetras.append(array([b,c0,c1,d0])) # mitte 3

    tetras.append(array([a,b,c1,c2])) # mitte 1
    tetras.append(array([b,d1,d2,c2])) # mitte 2
    tetras.append(array([b,c1,c2,d1])) # mitte 3

    tetras.append(array([a,b,c2,c3])) # mitte 1
    tetras.append(array([b,d2,d3,c3])) # mitte 2
    tetras.append(array([b,c2,c3,d2])) # mitte 3

    tetras.append(array([a,b,c3,c4])) # mitte 1
    tetras.append(array([b,d3,d4,c4])) # mitte 2
    tetras.append(array([b,c3,c4,d3])) # mitte 3

    tetras.append(array([a,b,c4,c5])) # mitte 1
    tetras.append(array([b,d4,d5,c5])) # mitte 2
    tetras.append(array([b,c4,c5,d4])) # mitte 3

    tetras.append(array([a,b,c5,c0])) # mitte 1
    tetras.append(array([b,d5,d0,c0])) # mitte 2
    tetras.append(array([b,c5,c0,d5])) # mitte 3

    return tetras

def get_cubes_ref():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 1.5]
    c0 = [cos(0*pi/3), sin(0*pi/3), 0]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]

    tetras.append(array([a,b,c0,c1])) 
    tetras.append(array([a,b,c0,c1])) 

    return tetras

def move_a_node_cube():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 1.5]
    c0 = [cos(0*pi/3)*1.5, sin(0*pi/3)*1.5, 0]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]

    tetras.append(array([a,b,c0,c1])) 
    tetras.append(array([a,b,c0,c1])) 

    return tetras

def get_cell_ref():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 1.5]
    c0 = [cos(0*pi/3), sin(0*pi/3), 0]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]

    tetras.append(array([a,b,c0,c1])) 

    return tetras

def move_a_node_cell():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 1.5]
    c0 = [cos(0*pi/3)*1.5, sin(0*pi/3)*1.5, 0]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]

    tetras.append(array([a,b,c0,c1])) 

    return tetras

def get_part_ref():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 1.5]

    c0 = [cos(0*pi/3), sin(0*pi/3), 0]
    d0 = [cos(0*pi/3), sin(0*pi/3), 1]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]
    d1 = [cos(1*pi/3), sin(1*pi/3), 1]

    tetras.append(array([a,b,c0,c1])) # mitte 1
    tetras.append(array([b,d0,d1,c1])) # mitte 2
    tetras.append(array([b,c0,c1,d0])) # mitte 3

    return tetras

def move_a_node_part():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 2]

    c0 = [cos(0*pi/3), sin(0*pi/3), 0]
    d0 = [cos(0*pi/3), sin(0*pi/3), 1]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]
    d1 = [cos(1*pi/3), sin(1*pi/3), 1]

    tetras.append(array([a,b,c0,c1])) # mitte 1
    tetras.append(array([b,d0,d1,c1])) # mitte 2
    tetras.append(array([b,c0,c1,d0])) # mitte 3

    return tetras

def move_a_node():
    tetras = []

    a = [0, 0, -.5]
    b = [0, 0, 2]

    c0 = [cos(0*pi/3), sin(0*pi/3), 0]
    d0 = [cos(0*pi/3), sin(0*pi/3), 1]
    c1 = [cos(1*pi/3), sin(1*pi/3), 0]
    d1 = [cos(1*pi/3), sin(1*pi/3), 1]
    c2 = [cos(2*pi/3), sin(2*pi/3), 0]
    d2 = [cos(2*pi/3), sin(2*pi/3), 1]
    c3 = [cos(3*pi/3), sin(3*pi/3), 0]
    d3 = [cos(3*pi/3), sin(3*pi/3), 1]
    c4 = [cos(4*pi/3), sin(4*pi/3), 0]
    d4 = [cos(4*pi/3), sin(4*pi/3), 1]
    c5 = [cos(5*pi/3), sin(5*pi/3), 0]
    d5 = [cos(5*pi/3), sin(5*pi/3), 1]


    tetras.append(array([a,b,c0,c1])) # mitte 1
    tetras.append(array([b,d0,d1,c1])) # mitte 2
    tetras.append(array([b,c0,c1,d0])) # mitte 3

    tetras.append(array([a,b,c1,c2])) # mitte 1
    tetras.append(array([b,d1,d2,c2])) # mitte 2
    tetras.append(array([b,c1,c2,d1])) # mitte 3

    tetras.append(array([a,b,c2,c3])) # mitte 1
    tetras.append(array([b,d2,d3,c3])) # mitte 2
    tetras.append(array([b,c2,c3,d2])) # mitte 3

    tetras.append(array([a,b,c3,c4])) # mitte 1
    tetras.append(array([b,d3,d4,c4])) # mitte 2
    tetras.append(array([b,c3,c4,d3])) # mitte 3

    tetras.append(array([a,b,c4,c5])) # mitte 1
    tetras.append(array([b,d4,d5,c5])) # mitte 2
    tetras.append(array([b,c4,c5,d4])) # mitte 3

    tetras.append(array([a,b,c5,c0])) # mitte 1
    tetras.append(array([b,d5,d0,c0])) # mitte 2
    tetras.append(array([b,c5,c0,d5])) # mitte 3

    return tetras

def get_domain_map(mesh, tetras_ref, subvolumes):

    domain_map = {}

    for i in range(len(tetras_ref)):
        dof = np.argmin(np.linalg.norm(mesh.coordinates() - np.sum(tetras_ref[i],0)/4, axis=1))
        for j in range(len(mesh.cells())):
            if dof in mesh.cells()[j]: 
                domain_map[i] = subvolumes.array()[j]
                break

    domain_map = {v: k for k, v in domain_map.items()}

    return domain_map

def get_jacobians(tetras_ref, tetras_phys, domain_map):
    J  = []
    R1 = []
    R2 = []

    for i in range(len(tetras_ref)):
        j,r1,r2 = getJacobian(tetras_ref[domain_map[i+1]],tetras_phys[domain_map[i+1]])
        J.append(j)
        R1.append(r1)
        R2.append(r2)
    return J,R1,R2

def get_bilinear_form_ref(Js,r1, r2, K_ortho,u,v,dx):
    
    K = Ortho(K_ortho,Js[0],r1[0],r2[0])
    J = Constant(Js[0])
    a = (dot(inv(K)*J*u, J*v/det(J)) - div(v)*p - div(u)*q)*dx(1)
    
    for i in range(1,len(Js)):
        K = Ortho(K_ortho,Js[i],r1[i],r2[i])
        J = Constant(Js[i])
        a += (dot(inv(K)*J*u, J*v/det(J)) - div(v)*p - div(u)*q)*dx(i+1)

    return a

def get_linear_form(p_in, p_out, n, v, ds):

    L  = Constant(p_out)*dot(n,v)*ds(6) - Constant(p_in)*dot(n,v)*ds(5)

    # for i in range(32,38):
    #     L -= Constant(p_in)*dot(n,v)*ds(i) 

    return L

def stretch_mesh(mesh, subvolumes,r1, r2):

    processed_nodes = []

    for i in range(len(r1)):
        start = list(subvolumes.array()).index(i+1)
        end   = len(list(subvolumes.array()))-list(subvolumes.array())[::-1].index(i+1)
        cells = mesh.cells()[start:end]
        cells = cells.reshape([-1,])
        nodes = np.unique(cells)

        nodes = list(set(nodes) - set(processed_nodes))

        coos = mesh.coordinates()[nodes]
        coos_new = np.zeros_like(coos)

        for j in range(coos.shape[0]):
            coos_new[j] = np.dot(J[i], coos[j] - r1[i]) + r2[i] 
            
        mesh.coordinates()[nodes] = coos_new
        processed_nodes += nodes

    return mesh

def norm(u): 
    return sqrt(dot(u,u))

class Ortho2(UserExpression):
    def eval(self, value, x):


        A = np.array([0,0,1.5])
        B = np.array([0,0,-.5])
        C = np.array([x[0], x[1], x[2]])
        a = C-A
        b = B-A
        
        f = np.dot(a,b)/np.dot(b,b)


        v1 = A + b*f - C
        v1 = v1 / np.linalg.norm(v1)
        v3 = b / np.linalg.norm(b)
        v2 = np.cross(v1,v3)


        A = np.mat(A)
        B = np.mat(B)
        C = np.mat(C)

        k1=1
        k2=1
        k3=1
        
        K = k1*np.outer(v1,v1) + k2*np.outer(v2,v2) + k3*np.outer(v3,v3)

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

class Ortho(UserExpression):

    def __init__(self,K,J,r1,r2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.J = J
        self.K = K
        self.r1 = r1
        self.r2 = r2

    def eval(self, value, x):
        x = np.dot(self.J,x - self.r1) + self.r2
        x = np.squeeze(np.array(x))
        phi = np.arctan2(x[1],x[0])

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

R = Expression(     (( 'cos(atan2(x[1],  x[0]))' , '-sin(atan2(x[1], x[0]))' , '0' ),  
                     ( 'sin(atan2(x[1],  x[0]))' , ' cos(atan2(x[1], x[0]))' , '0' ),  
                     ( '0'                       , '0'                       , '1' )), degree=0)

##################
### PARAMETERS ###
##################

p_in  = 1
p_out = 0

k_r   = 2.0
k_phi = 0.1
k_z   = 1.0

K_ortho = np.mat([(k_r,0,0),(0,k_phi,0),(0,0,k_z)])

mesh_file    = "square_lobule_part.msh"
surface_file = "Meshes/subdomains.xdmf"
file         = "Meshes/mesh.xdmf"

########################
### REFERENCE DOMAIN ###
########################

mesh = get_mesh(mesh_file)
W    = get_functionspace(mesh)
u,p  = TrialFunctions(W)
v,q  = TestFunctions(W)
n    = FacetNormal(mesh)

ds, subdomains = mark_surfaces(surface_file, mesh)
dx, subvolumes = mark_volumes(file)

tetras_ref  = get_part_ref()
tetras_phys = move_a_node_part()
domain_map  = get_domain_map(mesh, tetras_ref, subvolumes)
J,r1,r2     = get_jacobians(tetras_ref, tetras_phys, domain_map)

a     = get_bilinear_form_ref(J,r1, r2, K_ortho, u, v, dx)
L     = get_linear_form(p_in, p_out, n, v, ds)
bc    = DirichletBC(W.sub(0), Constant((0,0,0)), subdomains, 4) 
w_ref = Function(W)

solve(a == L, w_ref ,bc, solver_parameters = {'linear_solver': 'mumps'})

#######################
### PHYSICAL DOMAIN ###
#######################

mesh_phys = stretch_mesh(mesh, subvolumes, r1,r2)
W         = get_functionspace(mesh_phys)
u,p       = TrialFunctions(W)
v,q       = TestFunctions(W)
n         = FacetNormal(W)
I         = np.eye(3)

sub_domains_phys            = MeshFunction('size_t', mesh_phys, 2)
sub_domains_phys.array()[:] = subdomains.array()
ds                          = Measure('ds', domain=mesh_phys, subdomain_data=sub_domains_phys)

K       = Ortho(K_ortho,I, np.array([0,0,0]), np.array([0,0,0]))
a       = (dot(inv(K)*u,v) - div(v)*p - div(u)*q)*dx
L       = get_linear_form(p_in, p_out, n, v, ds)
bc      = DirichletBC(W.sub(0), Constant((0,0,0)), sub_domains_phys, 4) 
w_phys  = Function(W)

solve(a == L, w_phys, bc, solver_parameters={'linear_solver': 'mumps'})



####################
### SAVE RESULTS ###
####################

u_ref,p_ref   = w_ref.split()
u_phys,p_phys = w_phys.split()

mesh                = get_mesh(mesh_file)
W                   = get_functionspace(mesh)
w_error             = Function(W)
u_error, p_error    = w_error.split()
u_error.vector()[:] = u_phys.vector() - u_ref.vector()
p_error.vector()[:] = p_phys.vector() - p_ref.vector()
u_error             = project(norm(u_error))

File("Output/pressure_ref.pvd")     << p_ref
File("Output/flow_ref.pvd")         << u_ref
File("Output/mesh_ref.pvd")         << mesh
File("Output/flow_phys.pvd")        << u_phys
File("Output/pressure_phys.pvd")    << p_phys
File("Output/mesh_phys.pvd")        << mesh_phys
File("Output/pressure_error.pvd")   << p_error
File("Output/sub_volumes_phys.pvd") << subvolumes
File("Output/flow_error.pvd")       << u_error

print(time()-t1)