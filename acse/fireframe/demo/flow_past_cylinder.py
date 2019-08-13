import sys
sys.path.append("..")
from PDESystem import *
from PDESubsystem import *
from pdeforms import navier_stokes

class pde_solver(PDESystem):
    def __init__(self, comp, mesh, parameters):
        PDESystem.__init__(self, comp, mesh, parameters)

    def setup_bcs(self):
        x, y = fd.SpatialCoordinate(self.mesh)

        bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (1, 4)), # top-bottom and cylinder
          fd.DirichletBC(self.V['u'], ((4.0*1.5*y*(0.41 - y) / 0.41**2) ,0), 2)] # inflow
        bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 3)]  # outflow

        self.bc['u'][0] = [bcu, None, None, None,'fixed']
        self.bc['p'] = [[bcp, None, None, None, 'fixed']]

    def setup_constants(self):
        self.constants = {
            'k' : fd.Constant(self.prm['dt']),
            'n' : fd.FacetNormal(self.mesh),
            'f' : fd.Constant((0.0, 0.0)),
            'nu' : fd.Constant(0.001)
        }

if __name__ == '__main__':
    solver_parameters = recursive_update(solver_parameters,
    {
    'space': {'u': fd.VectorFunctionSpace},
    'degree': {'u': 2},
    'ksp_type': {'u': 'gmres', 'p': 'gmres'},
    'subsystem_class' : {'up' : navier_stokes},
    'precond': {'u': 'sor', 'p':'sor'},
    'dt' : 0.001,
    'T' :1
    }
    )

    mesh = fd.Mesh("../../meshes/flow_past_cylinder.msh")
    solver = pde_solver([['u', 'p']], mesh, solver_parameters)
    solver.setup_constants()
    solver.define(['u', 'p', 'u'], 'up')
    solver.setup_bcs()
    solver.solve()
