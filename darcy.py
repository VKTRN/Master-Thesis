from dolfin import *
from tools import *

### MESH & FUNCTION SPACE ###

msh_file = "lobule.msh"

msh_to_xdmf(msh_file,"Meshes")

mesh_file = XDMFFile("Meshes/mesh.xdmf")
mesh = Mesh()
mesh_file.read(mesh)

Q      = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
B      = FiniteElement("Bubble",   mesh.ufl_cell(), 4)
V      = VectorElement(NodalEnrichedElement(Q, B))
E      = MixedElement(V,Q)
W      = FunctionSpace(mesh, E)

n       = FacetNormal(mesh)
u,p     = TrialFunctions(W)
v,q     = TestFunctions(W)

### SUB-DOMAINS ###

subdomains_file = XDMFFile("Meshes/subdomains.xdmf")
subdomains_mesh = Mesh()
subdomains_file.read(subdomains_mesh)

subdomains = MeshValueCollection("size_t", mesh, 2)
subdomains_file.read(subdomains)

subdomains  = cpp.mesh.MeshFunctionSizet(mesh, subdomains)
ds          = Measure('ds', domain=mesh, subdomain_data=subdomains)

### VARIATIONAL FORMULATION ###

a  =  (dot(u,v) - div(v)*p - div(u)*q)*dx
L  = -Constant(3)*dot(n,v)*ds(1) + Constant(7)*dot(n,v)*ds(2)

bc = DirichletBC(W.sub(0), Constant((0,0,0)), subdomains, 3) 
w  = Function(W)

solve(a == L, w,bc, solver_parameters={'linear_solver': 'mumps'})

file = File("Output/flow.pvd")
file << w.split()[0]

file = File("Output/mesh.pvd")
file << mesh

file = File("Output/pressure.pvd")
file << w.split()[1]

file = File("Output/subdomains.pvd")
file << subdomains