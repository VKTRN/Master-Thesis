from dolfin import *

class Bottom(SubDomain):
    def inside(self,x, on_boundary):
        return x[2]<0.001 and on_boundary and (.2<x[0]<.8) and (.2<x[1]<.8)

class Top(SubDomain):
    def inside(self,x, on_boundary):
        return x[2]>.99 and on_boundary

def wall(x, on_boundary):
        return  ((x[0]<=.3 or x[0]>=.7) or (x[1]<=.3 or x[1]>=.7))  and on_boundary

### MESH & FUNCTION SPACE ###

mesh    = UnitCubeMesh(20,20,20)

Qe      = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Be      = FiniteElement("Bubble",   mesh.ufl_cell(), 4)
Ve      = VectorElement(NodalEnrichedElement(Qe, Be))
element = MixedElement(Ve,Qe)
W       = FunctionSpace(mesh, element)

n       = FacetNormal(mesh)
u,p     = TrialFunctions(W)
v,q     = TestFunctions(W)

### SUB-DOMAINS ###

bottom = Bottom()
top    = Top()

sub_domains = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
sub_domains.set_all(0)

bottom.mark(sub_domains, 1)
top.mark(sub_domains, 2)

ds = Measure('ds', domain=mesh, subdomain_data=sub_domains)

### BOUNDARY CONDITION ###

bc = DirichletBC(W.sub(0), Constant((0,0,0)), wall) # no slip top

### VARIATIONAL FORMULATION ###

a =  (dot(u,v) - div(v)*p - div(u)*q)*dx
L = -Constant(1)*dot(n,v)*ds(1) + Constant(2)*dot(n,v)*ds(2)

w = Function(W)

solve(a == L, w, bc, solver_parameters={'linear_solver': 'mumps'})

file = File("output/flow.pvd")
file << w.split()[0]

file = File("output/pressure.pvd")
file << w.split()[1]

file = File("output/subdomains.pvd")
file << sub_domains