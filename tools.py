import meshio
from dolfin import *
import numpy as np

def msh_to_xdmf(file, path):
    msh = meshio.read(path + "/" + file)
    triangle_cells = []
    tetra_cells    = []
    triangle_data  = []
    tetra_data     = []

    for cell in msh.cells:
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





