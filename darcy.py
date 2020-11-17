from dolfin import *
from matplotlib import pyplot as plt
import numpy as np

def bottom(x, on_boundary):
    return x[2]<0.001 and on_boundary and (.2<x[0]<.8) and (.2<x[1]<.8)

def top(x, on_boundary):
    return x[2]>.99 and on_boundary

def wall(x, on_boundary):
    return  ((x[0]<=.3 or x[0]>=.7) or (x[1]<=.3 or x[1]>=.7))  and on_boundary

mesh     = UnitCubeMesh(10, 10, 10)

BDM      = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG       = FiniteElement("DG",  mesh.ufl_cell(), 0)
E        = MixedElement(BDM, DG)
W        = FunctionSpace(mesh, E)

n        = FacetNormal(mesh)
u,p      = TrialFunctions(W)
v,q      = TestFunctions(W)

bottomBC = DirichletBC(W.sub(0), Constant((0,0,1)), bottom)
wallBC   = DirichletBC(W.sub(0), Constant((0,0,0)), wall) 
bcs      = [wallBC, bottomBC]

a        = (dot(u,v)  - div(v)*p - div(u)*q)*dx
L        = dot(Constant((0,0,0)), v)*dx
w        = Function(W)

solve(a == L, w, bcs, solver_parameters={'linear_solver': 'mumps'})

file = File("output/darcy.pvd")
file << w.split()[0]