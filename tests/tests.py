import sys
sys.path.append("..")
from acse.fireframe.PDESystem import *
from acse.fireframe.PDESubsystem import *
from acse.fireframe.pdeforms import *
import firedrake as fd



def test_parameters():
	solver_parameters  = copy.deepcopy(default_solver_parameters)

	params = ['space', 'degree', 'ksp_type', 'subsystem_class', 'precond', 'dt', 'T']

	diction = {
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'ksp_type': {'u': 'gmres', 'p' :'gmres'},
	'subsystem_class' : {'up' : navier_stokes},
	'precond': {'u': 'sor', 'p': 'sor'},
	'dt' : 0.01,
	'T' : 10
	}

	solver_parameters = recursive_update(solver_parameters, diction)

	for par in params:
		assert (solver_parameters[par] is not None)

def test_load_mesh():

	# flow_past_cylinder
	mesh1 = fd.Mesh("acse/meshes/flow_past_cylinder.msh")
	# high resolution cylinder
	mesh2 = fd.Mesh("acse/meshes/cylinder.msh")
	assert(mesh1 is not None)
	assert(mesh2 is not None)

	for i in range(1, 11):
		mesh = fd.Mesh("acse/meshes/step%d.msh" % i)
		assert(mesh is not None)

def test_flow_demo():
	solver_parameters  = copy.deepcopy(default_solver_parameters)
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

	mesh = fd.Mesh("acse/meshes/flow_past_cylinder.msh")
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up')
	solver.setup_bcs()

def test_channel_demo():
	solver_parameters  = copy.deepcopy(default_solver_parameters)
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
			x, y = fd.SpatialCoordinate(self.mesh)
			self.constants = {
				'k' : fd.Constant(self.prm['dt']),
				'n' : fd.FacetNormal(self.mesh),
				'f' : fd.Constant((0.0, 0.0)),
				'nu' : fd.Constant(0.001),
				'eps' : fd.Constant(0.01),
				'K' : fd.Constant(10.0),
				'f_1' : fd.conditional(pow(x-0.1, 2)+pow(y-0.1,2)<0.05*0.05, 0.1, 0),
				'f_2' : fd.conditional(pow(x-0.1, 2)+pow(y-0.3,2)<0.05*0.05, 0.1, 0),
				'f_3' : fd.Constant(0.0)
			}

	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'ksp_type': {'u': 'gmres', 'p': 'gmres'},
	'subsystem_class' : {'up' : navier_stokes},
	'precond': {'u': 'sor', 'p' : 'sor'},
	'dt' : 0.0005,
	 'T' : 1.0 }
	)

	mesh = fd.Mesh("acse/meshes/cylinder.msh")
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
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up')
	solver.define(['c'], 'c')
	solver.setup_bcs()

def test_radio_transport():
	solver_parameters  = copy.deepcopy(default_solver_parameters)
	class pde_solver(PDESystem):
		def __init__(self, comp, mesh, parameters):
			PDESystem.__init__(self, comp, mesh, parameters)

		def setup_bcs(self):
			x, y = fd.SpatialCoordinate(self.mesh)

			bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (10, 12)), # top-bottom and cylinder
			  fd.DirichletBC(self.V['u'], ((1.0*(y - 1)*(2 - y))/(0.5**2) ,0), 9)] # inflow
			bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 11)]  # outflow

			self.bc['u'][0] = [bcu, None, None, None,'fixed']
			self.bc['p'] = [[bcp, None, None, None, 'fixed']]

		def setup_constants(self):
			self.constants = {
				'k' : fd.Constant(self.prm['dt']),
				'Kd' : fd.Constant(0.01),
				'k1' : fd.Constant(0.5),
				'k2' : fd.Constant(0.01),
				'lamd1' : fd.Constant(1.5),
				'lamd2' : fd.Constant(0.),
				'rho_s' : fd.Constant(1.),
				'L' :  fd.Constant(1.),
				'phi' : fd.Constant(0.3),
				'f' : fd.Constant(1.),
				'n' : fd.FacetNormal(self.mesh),
				'f' : fd.Constant((0.0, 0.0)),
				'nu' : fd.Constant(0.001),
				'frac' : fd.Constant(1.)
			}

	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace, 'cs': fd.MixedFunctionSpace, 'cd' : fd.MixedFunctionSpace, 'as' : fd.MixedFunctionSpace},
	'degree': {'u': 2, 'p': 1, 'cs': 1, 'cd' : 1, 'as' : 1},
	'order' : {'u': 1, 'p': 1, 'cs' : 2, 'cd' : 2, 'as' : 2, 'cdcsas' : 2},
	'ksp_type': {'u': 'gmres', 'p': 'gmres', 'cs': 'gmres', 'cd': 'gmres', 'as': 'gmres'},
	'subsystem_class' : {'up': navier_stokes, 'cdcsas' : radio_transport},
	'precond': {'u': 'sor', 'p' : 'sor', 'cs': 'sor', 'cd': 'sor', 'as': 'sor'},
	'T': 1.0
	}
	)

	#define mesh
	mesh = fd.Mesh("acse/meshes/step3.msh")

	# add subsystems
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver.add_subsystem(['cd', 'cs', 'as'], solver_parameters)

	#setup system and define subsystems
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up')
	solver.define(['cd', 'cs', 'as'], 'cdcsas')
	solver.setup_bcs()


	#setup initial condition
	x, y = fd.SpatialCoordinate(mesh)
	c = fd.conditional(pow(x-1, 2)+pow(y-1.5,2)<0.05*0.05, 10, 0)
	solver.setup_initial('cd_n', c, mixedspace=True, index=0)
