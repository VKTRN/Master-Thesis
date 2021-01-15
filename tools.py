import meshio
from dolfin import *
import numpy as np
from time import sleep

def msh_to_xdmf(file, path):
    msh = meshio.read(path + "/" + file)
    triangle_cells = []
    tetra_cells    = []
    triangle_data  = []
    tetra_data     = []

    for cell in msh.cells:
        #print(cell)
        if cell.type == "triangle":
            if len(triangle_cells)==0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])
        elif  cell.type == "tetra":
            if len(tetra_cells)==0:
                tetra_cells = cell.data
            else:
                tetra_cells = np.vstack([tetra_cells, cell.data])

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "tetra":
            tetra_data = msh.cell_data_dict["gmsh:physical"][key]

    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells}, cell_data={"name_to_read":[tetra_data]})
    triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells}, cell_data={"name_to_read":[triangle_data]})

    meshio.write(path + "/" + "mesh.xdmf", tetra_mesh)
    meshio.write(path + "/" + "subdomains.xdmf", triangle_mesh)
    
def msh_to_xdmf2D(file, path):
    msh = meshio.read(path + "/" + file)
    print("_______")
    print(msh.points)
    print("_______")

    line_cells = []
    triangle_cells    = []
    line_data  = []
    triangle_data     = []

    for cell in msh.cells:
        if cell.type == "line":
            if len(line_cells)==0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])
        elif  cell.type == "triangle":
            if len(triangle_cells)==0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            line_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]

    triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells}, cell_data={"name_to_read":[triangle_data]})
    line_mesh = meshio.Mesh(points=msh.points, cells={"line": line_cells}, cell_data={"name_to_read":[line_data]})

    meshio.write(path + "/" + "mesh.xdmf", triangle_mesh)
    meshio.write(path + "/" + "subdomains.xdmf", line_mesh)

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