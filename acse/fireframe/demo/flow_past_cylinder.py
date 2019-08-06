import sys
sys.path.append("..")
from PDESystem import *
from PDESubsystem import *
from pdeforms import navier_stokes

class navier_stokes(PDESubsystem):
    
    def form1(self, u_trl, u_tst, u_n, p_n, **kwargs):
        n = fd.FacetNormal(self.mesh)
        f = fd.Constant((0.0, 0.0))
        u_mid = 0.5 * (u_n + u_trl)
        nu = fd.Constant(0.001)
        k = fd.Constant(self.prm['dt'])
        
        def sigma(u, p):
            return 2*nu*fd.sym(fd.nabla_grad(u)) - p*fd.Identity(len(u))
        
        Form = fd.inner((u_trl - u_n)/k, u_tst) * fd.dx \
        + fd.inner(fd.dot(u_n, fd.nabla_grad(u_mid)), u_tst) * fd.dx \
        + fd.inner(sigma(u_mid, p_n), fd.sym(fd.nabla_grad(u_tst))) * fd.dx \
        + fd.inner(p_n * n, u_tst) * fd.ds \
        - fd.inner(nu * fd.dot(fd.nabla_grad(u_mid), n), u_tst) * fd.ds \
        - fd.inner(f, u_tst) * fd.dx
        return Form
    
    def form2(self, p_trl, p_tst, p_n, u_, **kwargs):
        k = fd.Constant(self.prm['dt'])
        
        Form = fd.inner(fd.nabla_grad(p_trl), fd.nabla_grad(p_tst)) * fd.dx \
        - fd.inner(fd.nabla_grad(p_n), fd.nabla_grad(p_tst)) * fd.dx \
        + (1/k) * fd.inner(fd.div(u_), p_tst) * fd.dx
        
        return Form
    
    def form3(self, u_trl, u_tst, u_, p_n, p_, **kwargs):
        k = fd.Constant(self.prm['dt'])
        
        Form = fd.inner(u_trl, u_tst) * fd.dx \
        - fd.inner(u_, u_tst) * fd.dx \
        + k * fd.inner(fd.nabla_grad(p_ - p_n), u_tst) * fd.dx
        
        return Form

class pde_solver(PDESystem):
    def __init__(self, comp, mesh, parameters):
        PDESystem.__init__(self, comp, mesh, parameters)

    def setup_bcs(self):
        x, y = fd.SpatialCoordinate(self.mesh)
        
        bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (1, 4)), # top-bottom and cylinder
          fd.DirichletBC(self.V['u'], ((4.0*1.5*y*(0.41 - y) / 0.41**2) ,0), 2)] # inflow
        bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 3)]  # outflow
        
        self.bc['up'] = [bcu, bcp, None]
        
    def define(self, name, var_seq, bcs, forms_cnt):
        self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, name, bcs, forms_cnt)

if __name__ == '__main__':
    solver_parameters  = copy.deepcopy(default_solver_parameters)

    solver_parameters = recursive_update(solver_parameters, 
    {
    'space': {'u': fd.VectorFunctionSpace},
    'degree': {'u': 2},
    'linear_solver': {'u': 'gmres', 'p': 'gmres'},
    'subsystem_class' : {'up' : navier_stokes},
    'precond': {'u': 'sor', 'p':'sor'}}
    )

    mesh = fd.Mesh("../../../meshes/flow_past_cylinder.msh")
    solver = pde_solver([['u', 'p']], mesh, solver_parameters)
    solver.setup_bcs()
    solver.setup_constants()
    solver.define(['u', 'p', 'u'], 'up')
    solver.solve()
    
