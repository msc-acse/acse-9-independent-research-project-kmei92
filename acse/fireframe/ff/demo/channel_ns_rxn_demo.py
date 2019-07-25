import sys
sys.path.append("..")
from channel_ns_demo import navier_stokes
from PDESystem import *
from PDESubsystem import *

class pde_solver(PDESystem):
	def __init__(self, comp, mesh, parameters):
		PDESystem.__init__(self, comp, mesh, parameters)

	def setup_bcs(self):
		x, y = fd.SpatialCoordinate(self.mesh)

		bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (1, 4)), # top-bottom and cylinder
		  fd.DirichletBC(self.V['u'], ((4.0*1.5*y*(0.41 - y) / 0.41**2) ,0), 2)] # inflow
		bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 3)]  # outflow

		self.bc['up'] = [bcu, bcp, None]
		self.bc['c'] = [None]

	def define(self, name, var_seq, bcs, forms_cnt):
		self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, name, bcs, forms_cnt)

class reactions(PDESubsystem):

	def form1(self, c_1, c_n1, c_tst1, c_2, c_n2, c_tst2, c_3, c_n3, c_tst3, u_, **kwargs):
		eps = fd.Constant(0.01)
		K = fd.Constant(10.0)
		k = fd.Constant(self.prm['dt'])

		x, y = fd.SpatialCoordinate(self.mesh)
		f_1 = fd.conditional(pow(x-0.1, 2)+pow(y-0.1,2)<0.05*0.05, 0.1, 0)
		f_2 = fd.conditional(pow(x-0.1, 2)+pow(y-0.3,2)<0.05*0.05, 0.1, 0)
		f_3 = fd.Constant(0.0)

		Form = ((c_1 - c_n1) / k)*c_tst1*fd.dx \
		+ ((c_2 - c_n2) / k)*c_tst2*fd.dx \
		+ ((c_3 - c_n3) / k)*c_tst3*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_1)), c_tst1)*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_2)), c_tst2)*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_3)), c_tst3)*fd.dx \
		+ eps*fd.dot(fd.grad(c_1), fd.grad(c_tst1))*fd.dx \
		+ eps*fd.dot(fd.grad(c_2), fd.grad(c_tst2))*fd.dx \
		+ eps*fd.dot(fd.grad(c_3), fd.grad(c_tst3))*fd.dx \
		+ K*c_1*c_2*c_tst1*fd.dx  \
		+ K*c_1*c_2*c_tst2*fd.dx  \
		- K*c_1*c_2*c_tst3*fd.dx \
		+ K*c_3*c_tst3*fd.dx \
		- f_1*c_tst1*fd.dx \
		- f_2*c_tst2*fd.dx \
		- f_3*c_tst3*fd.dx

		return Form

if __name__ == '__main__':
	solver_parameters  = copy.deepcopy(default_solver_parameters)

	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'linear_solver': {'up': 'gmres'},
	'subsystem_class' : {'up' : navier_stokes},
	'precond': {'up': 'sor'}}
	)

	mesh = fd.Mesh("../../../meshes/cylinder.msh")
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)

	solver_parameters = recursive_update(solver_parameters,
	{
	    'space': {'c': fd.MixedFunctionSpace},
	    'degree': {'c': 1},
	    'order' : {'c' : 3},
	    'linear_solver': {'c': 'gmres'},
	    'subsystem_class' : {'c' : reactions},
	    'precond': {'c': 'sor'}
	})

	solver.add_subsystem('c', solver_parameters)
	solver.setup_bcs()
	solver.define('up', ['u', 'p', 'u'], solver.bc['up'], 3)
	solver.define('c', ['c'], solver.bc['c'], 1)
	solver.solve()

	#fig1 = plt.figure(figsize=(12, 4))
	#ax1 = fig1.add_subplot(111)
	#fd.plot(mesh, axes=ax1)
	#plt.pause(3)

	#fig2 = plt.figure(figsize=(16, 4))
	#ax2 = fig2.add_subplot(111)
	#ax2.set_xlabel('$x$', fontsize=16)
	#ax2.set_ylabel('$y$', fontsize=16)
	#ax2.set_title('FEM Navier-Stokes - channel flow - pressure', fontsize=16)
	#fd.plot(solver.form_args['p_'],axes=ax2)
	#ax2.axis('equal')
	#plt.pause(3)

	#fig3 = plt.figure(figsize=(16, 4))
	#ax3 = fig3.add_subplot(111)
	#ax3.set_xlabel('$x$', fontsize=16)
	#ax3.set_ylabel('$y$', fontsize=16)
	#ax3.set_title('FEM Navier-Stokes - channel flow - velocity', fontsize=16)
	#fd.plot(solver.form_args['u_'],axes=ax3)
	#ax3.axis('equal');
	#plt.pause(3)

	c1, c2, c3 = solver.form_args['c_'].split()

	fig4 = plt.figure(figsize=(16, 2.5))
	ax4 = fig4.add_subplot(111)
	ax4.set_xlabel('$x$', fontsize=16)
	ax4.set_ylabel('$y$', fontsize=16)
	ax4.set_title('FEM Navier-Stokes - channel flow - conc1', fontsize=16)
	fd.plot(c1,axes=ax4)
	ax4.axis('equal')
	plt.pause(3)

	fig5 = plt.figure(figsize=(16, 2.5))
	ax5 = fig5.add_subplot(111)
	ax5.set_xlabel('$x$', fontsize=16)
	ax5.set_ylabel('$y$', fontsize=16)
	ax5.set_title('FEM Navier-Stokes - channel flow - conc2', fontsize=16)
	fd.plot(c2,axes=ax5)
	ax5.axis('equal')
	plt.pause(3)

	fig6 = plt.figure(figsize=(16, 2.5))
	ax6 = fig6.add_subplot(111)
	ax6.set_xlabel('$x$', fontsize=16)
	ax6.set_ylabel('$y$', fontsize=16)
	ax6.set_title('FEM Navier-Stokes - channel flow - conc3', fontsize=16)
	fd.plot(c3,axes=ax6)
	ax6.axis('equal')
	plt.pause(3)
