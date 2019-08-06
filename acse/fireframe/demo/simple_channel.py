import sys
sys.path.append("..")
from PDESystem import *
from PDESubsystem import *
from navier_stokes import navier_stokes
import numpy.linalg as nl
import numpy as np


class pde_solver(PDESystem):
	def __init__(self, comp, mesh, parameters):
		PDESystem.__init__(self, comp, mesh, parameters)

	def setup_bcs(self):
		x, y = fd.SpatialCoordinate(self.mesh)

		bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0, 0)), (3, 4))] # no slip on top and botom
		bcp = [fd.DirichletBC(self.V['p'], fd.Constant(8), 1), # inflow
				fd.DirichletBC(self.V['p'], fd.Constant(0), 2)]  # outflow

		self.bc['up'] = [bcu, bcp, None]

	def setup_constants(self):
		self.constants = {
			'k' : fd.Constant(self.prm['dt']),
			'n' : fd.FacetNormal(self.mesh),
			'f' : fd.Constant((0.0, 0.0)),
			'nu' : fd.Constant(1),
		}

	def define(self, var_seq, name):
		self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, name, self.constants,
														self.bc[name])

if __name__ == '__main__':
	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'linear_solver': {'u': 'gmres', 'p' :'gmres'},
	'subsystem_class' : {'up' : navier_stokes},
	'precond': {'u': 'sor', 'p': 'sor'},
	'dt' : 0.01,
	'T' : 10
	})

	mesh = fd.UnitSquareMesh(8, 8)
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver.setup_bcs()
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up')
	solver.solve()

	#compare with exact Poiseuille flow solution
	x, y = fd.SpatialCoordinate(mesh)
	u_exact = fd.interpolate(fd.as_vector((4*y*(1-y), 0)), solver.V['u'])

	#compute L2 error
	error_l2 = nl.norm(solver.form_args['u_'].dat.data - u_exact.dat.data) / np.sqrt(u_exact.dat.data.shape)
	print(error_l2)
	# error_L2 = fd.errornorm(u_exact, solver.form_args['u_'], 'L2') / fd.sqrt(mesh.num_vertices())
	# print(error_L2)

	# error_L2 = fd.errornorm(u_exact, solver.form_args['u_'], 'L2') / fd.sqrt(mesh.num_vertices())
	# print(error_L2)
	# fig1 = plt.figure(figsize=(16, 4))
	# ax1 = fig1.add_subplot(111)
	# ax1.set_xlabel('$x$', fontsize=16)
	# ax1.set_ylabel('$y$', fontsize=16)
	# ax1.set_title('FEM Navier-Stokes - channel flow - flow exact', fontsize=16)
	# fd.plot(u_exact,axes=ax1)
	# ax1.axis('equal')
	# plt.pause(3)
	#
	# # fig2 = plt.figure(figsize=(16, 4))
	# # ax2 = fig2.add_subplot(111)
	# # ax2.set_xlabel('$x$', fontsize=16)
	# # ax2.set_ylabel('$y$', fontsize=16)
	# # ax2.set_title('FEM Navier-Stokes - channel flow - pressure', fontsize=16)
	# # fd.plot(solver.form_args['p_'],axes=ax2)
	# # ax2.axis('equal')
	# # plt.pause(3)
	# #
	# fig3 = plt.figure(figsize=(16, 4))
	# ax3 = fig3.add_subplot(111)
	# ax3.set_xlabel('$x$', fontsize=16)
	# ax3.set_ylabel('$y$', fontsize=16)
	# ax3.set_title('FEM Navier-Stokes - channel flow - velocity', fontsize=16)
	# fd.plot(solver.form_args['u_'],axes=ax3)
	# ax3.axis('equal');
	# plt.pause(3)
