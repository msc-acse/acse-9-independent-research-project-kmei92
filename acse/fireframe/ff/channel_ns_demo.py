from fireframe import *

class navier_stokes_solver(PDESystem):
    def __init__(self, mesh, parameters):
        PDESystem.__init__(self, [['u', 'p']], mesh, parameters)
        
        self.setup_bcs()
        self.define()
        
    def setup_bcs(self):
        x, y = fd.SpatialCoordinate(self.mesh)
        
        self.bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (1, 4)), # top-bottom and cylinder
          fd.DirichletBC(self.V['u'], ((4.0*1.5*y*(0.41 - y) / 0.41**2) ,0), 2)] # inflow
        self.bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 3)]  # outflow
        
    def define(self):
        self.pdesubsystems['up'] = navier_stokes(vars(self), ['u', 'p', 'u'], [self.bcu, self.bcp, None], forms_cnt=3)


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
    
    
if __name__ == '__main__':
	solver_parameters  = copy.deepcopy(default_solver_parameters)

	solver_parameters = recursive_update(solver_parameters, 
	{
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'linear_solver': {'up': 'gmres'},
	'precond': {'up': 'sor'}}
	)
	mesh = fd.Mesh("../../meshes/flow_past_cylinder.msh")
	solver = navier_stokes_solver(mesh, solver_parameters)
	solver.solve()

	fig1 = plt.figure(figsize=(12, 4))
	ax1 = fig1.add_subplot(111)
	fd.plot(mesh, axes=ax1)
	fig1.show()
	plt.pause(3)

	fig2 = plt.figure(figsize=(16, 4))
	ax2 = fig2.add_subplot(111)
	ax2.set_xlabel('$x$', fontsize=16)
	ax2.set_ylabel('$y$', fontsize=16)
	ax2.set_title('FEM Navier-Stokes - channel flow - pressure', fontsize=16)
	fd.plot(solver.pdesubsystems['up'].form_args['p_'],axes=ax2)
	ax2.axis('equal')
	fig2.show()
	plt.pause(3)

	fig3 = plt.figure(figsize=(16, 4))
	ax3 = fig3.add_subplot(111)
	ax3.set_xlabel('$x$', fontsize=16)
	ax3.set_ylabel('$y$', fontsize=16)
	ax3.set_title('FEM Navier-Stokes - channel flow - velocity', fontsize=16)
	fd.plot(solver.pdesubsystems['up'].form_args['u_'],axes=ax3)
	ax3.axis('equal');
	fig3.show()
	plt.pause(3)
    
