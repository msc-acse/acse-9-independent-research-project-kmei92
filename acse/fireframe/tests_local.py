"""
author : Keer Mei
email: keer.mei18@imperial.ac.uk
github username: kmei92
"""

from PDESystem import *
from PDESubsystem import *
from pdeforms import *
import firedrake as fd



def test_parameters():
	solver_parameters  = copy.deepcopy(default_solver_parameters)

	params = ['space', 'degree', 'family',  'ksp_type', 'precond', 'dt', 'T']

	diction = {
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'ksp_type': {'u': 'gmres', 'p' :'gmres'},
	'precond': {'u': 'sor', 'p': 'sor'},
	'dt' : 0.01,
	'T' : 10
	}

	solver_parameters = recursive_update(solver_parameters, diction)

	for par in params:
		assert (solver_parameters[par] is not None)

def test_load_mesh():

	# flow_past_cylinder
	mesh1 = fd.Mesh("meshes/flow_past_cylinder.msh")
	# high resolution cylinder
	mesh2 = fd.Mesh("meshes/cylinder.msh")
	assert(mesh1 is not None)
	assert(mesh2 is not None)

	for i in range(1, 11):
		mesh = fd.Mesh("meshes/step%d.msh" % i)
		assert(mesh is not None)

	for i in range(1, 6):
		mesh = fd.Mesh("meshes/cylinder%d.msh" % i)
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
			self.constants.update({
				'deltat' : fd.Constant(self.prm['dt']),
				'n' : fd.FacetNormal(self.mesh),
				'f' : fd.Constant((0.0, 0.0)),
				'nu' : fd.Constant(0.001)
			})

	# update the parameters
	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'ksp_type': {'u': 'gmres', 'p': 'gmres'},
	'precond': {'u': 'sor', 'p':'sor'},
	'dt' : 0.01,
	'T' :0.5
	}
	)

	mesh = fd.Mesh("meshes/flow_past_cylinder.msh")
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up', navier_stokes)
	solver.setup_bcs()

def test_rxn_demo():
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
				'deltat' : fd.Constant(self.prm['dt']),
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
	'space': {'u': fd.VectorFunctionSpace, 'c' : fd.MixedFunctionSpace},
	'degree': {'u': 2 },
	'order' : {'c' : 3},
	'ksp_type': {'u': 'gmres', 'p': 'gmres', 'c':'gmres'},
	'precond': {'u': 'sor', 'p' : 'sor', 'c':'sor'},
	'dt' : 0.0005,
	 'T' : 1.0 }
	)

	mesh = fd.Mesh("meshes/cylinder.msh")
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver.add_subsystem('c', solver_parameters)
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up', navier_stokes)
	solver.define(['c'], 'c', reactions)
	solver.setup_bcs()

	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace, 'c1': fd.FunctionSpace, 'c2' :fd.FunctionSpace, 'c3':fd.FunctionSpace},
	'degree': {'u': 2},
	'ksp_type': {'u': 'gmres', 'p': 'gmres', 'c1': 'gmres','c2': 'gmres','c3': 'gmres'},
	'subsystem_class' : {'up' : navier_stokes, 'c1c2c3' : reactions_uncoupled}, #also a new system is declared
	'precond': {'u': 'sor', 'p' : 'sor', 'c1': 'sor','c2': 'sor','c3': 'sor'},
	'dt' : 0.0005,
	 'T' : 1.0 }
	)
	# test uncoupled
	solver2 = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver2.add_subsystem(['c1', 'c2', 'c3'], solver_parameters)
	solver2.setup_constants()
	solver2.define(['u', 'p', 'u'], 'up', navier_stokes)
	solver2.define(['c1', 'c2', 'c3'], 'c1c2c3', reactions_uncoupled)
	solver2.setup_bcs()

def test_radio_transport():
	solver_parameters  = copy.deepcopy(default_solver_parameters)
	class pde_solver(PDESystem):
		def __init__(self, comp, mesh, parameters):
			PDESystem.__init__(self, comp, mesh, parameters)

		def setup_bcs(self):
			x, y = fd.SpatialCoordinate(self.mesh)

			bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (10, 12)), # top-bottom
			  fd.DirichletBC(self.V['u'], ((1.0*(y - 1)*(2 - y))/(0.5**2) ,0), 9)] # inflow
			bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 11)]  # outflow


			self.bc['u'][0] = [bcu, None, None, None,'fixed']
			self.bc['p'] = [[bcp, None, None, None, 'fixed']]

		def setup_constants(self):
			x, y = fd.SpatialCoordinate(self.mesh)
			self.constants = {
				'deltat' : fd.Constant(self.prm['dt']),
				'Kd' : fd.Constant(0.01),
				'k1' : fd.Constant(0.005),
				'k2' : fd.Constant(0.00005),
				'lamd1' : fd.Constant(0.000005),
				'lamd2' : fd.Constant(0.0),
				'rho_s' : fd.Constant(1.),
				'L' :  fd.Constant(1.),
				'phi' : fd.Constant(0.3),
				'n' : fd.FacetNormal(self.mesh),
				'f' : fd.Constant((0.0, 0.0)),
				'nu' : fd.Constant(0.001),
				'frac' : fd.Constant(1.),
				'source1' : fd.conditional(pow(x-1, 2)+pow(y-1.5,2)<0.25*0.25, 10.0, 0)
			}

	# update the parameters
	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace, 'c': fd.MixedFunctionSpace, 'd' : fd.MixedFunctionSpace},
	'degree': {'u': 2},
	'order' : {'c': 3, 'd':3},
	'ksp_type': {'u': 'gmres', 'p': 'gmres',  'c': 'gmres', 'd':'gmres'},
	'precond': {'u': 'sor', 'p' : 'sor', 'c': 'sor', 'd':'gmres'},
	'dt' : 0.05,
	'T' : 5.0}
	)

	#load mesh
	mesh = fd.Mesh("meshes/step10.msh")
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	solver.add_subsystem(['c', 'd'], solver_parameters)
	solver.setup_constants()
	solver.define(['u', 'p', 'u'], 'up', navier_stokes)
	solver.define(['c', 'd'], 'cd', radio_transport_coupled)
	solver.setup_bcs()

def test_dx():
	solver_parameters  = copy.deepcopy(default_solver_parameters)
	class pde_solver(PDESystem):
		def __init__(self, comp, mesh, parameters):
			PDESystem.__init__(self, comp, mesh, parameters)

		def setup_bcs(self):
			x, y = fd.SpatialCoordinate(self.mesh)
			c0 = fd.exp(x*y*self.t)

			bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (10, 12)), # top-bottom and cylinder
			  fd.DirichletBC(self.V['u'], ((1.0*(y - 1)*(2 - y))/(0.5**2) ,0), 9)] # inflow
			bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 11)]  # outflow
			bcc1 = [fd.DirichletBC(self.V['c'].sub(0), c0, 'on_boundary')]

			self.bc['u'][0] = [bcu, None, None, None,'fixed']
			self.bc['p'] = [[bcp, None, None, None, 'fixed']]
			self.bc['c'][0] = [bcc1, c0, 'on_boundary', 0, 'update']

		def setup_constants(self):
			x, y = fd.SpatialCoordinate(self.mesh)

			self.constants = {
				'deltat' : fd.Constant(self.prm['dt']),
				'Kd' : fd.Constant(0.01),
				'k1' : fd.Constant(0.005),
				'k2' : fd.Constant(0.00005),
				'lamd1' : fd.Constant(0.000005),
				'lamd2' : fd.Constant(0.0),
				'rho_s' : fd.Constant(1.),
				'L' :  fd.Constant(1.),
				'phi' : fd.Constant(0.3),
				'n' : fd.FacetNormal(self.mesh),
				'f' : fd.Constant((0.0, 0.0)),
				'nu' : fd.Constant(0.001),
				'frac' : fd.Constant(1.),
			}
	# update the parameters
	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace, 'c': fd.MixedFunctionSpace, 'd' : fd.MixedFunctionSpace},
	'degree': {'u': 2},
	'order' : {'c': 3, 'd':3},
	'ksp_type': {'u': 'gmres', 'p': 'gmres',  'c': 'gmres', 'd':'gmres'},
	'precond': {'u': 'sor', 'p' : 'sor', 'c': 'sor', 'd':'gmres'},
	'dt' : 0.001,
	'T' : 0.1})

	#load mesh
	mesh = fd.Mesh("meshes/step1.msh")

	# add subsystems for navier stokes and radio_transport
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	# solver.add_subsystem(['cd', 'cs', 'as'], solver_parameters)
	solver.add_subsystem(['c', 'd'], solver_parameters)
	#setup constants
	solver.setup_constants()
	# define subsystems and variable sequence
	solver.define(['u', 'p', 'u'], 'up', navier_stokes)
	solver.define(['c', 'd'], 'cd', radio_transport_coupled_mms)
	# setup boundary conditions
	solver.setup_bcs()

	x, y, t = sy.symbols(('x', 'y', 't'))
	expr = sy.exp(x*y*t)
	meshes = [fd.Mesh("meshes/step%d.msh" % i) for i in range(1, 2)]
	solver.test_mms('c', expr, spatial=True, f_dict={"exp":fd.exp}, meshes=meshes, plot=False, index=0)

def test_dt():
	solver_parameters  = copy.deepcopy(default_solver_parameters)
	class pde_solver(PDESystem):
		def __init__(self, comp, mesh, parameters):
			PDESystem.__init__(self, comp, mesh, parameters)

		def setup_bcs(self):
			x, y = fd.SpatialCoordinate(self.mesh)
			c0 = fd.exp(x*y*self.t)

			bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (10, 12)), # top-bottom and cylinder
			  fd.DirichletBC(self.V['u'], ((1.0*(y - 1)*(2 - y))/(0.5**2) ,0), 9)] # inflow
			bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 11)]  # outflow
			bcc1 = [fd.DirichletBC(self.V['c'].sub(0), c0, 'on_boundary')]

			self.bc['u'][0] = [bcu, None, None, None,'fixed']
			self.bc['p'] = [[bcp, None, None, None, 'fixed']]
			self.bc['c'][0] = [bcc1, c0, 'on_boundary', 0, 'update']

		def setup_constants(self):
			x, y = fd.SpatialCoordinate(self.mesh)

			self.constants = {
				'deltat' : fd.Constant(self.prm['dt']),
				'Kd' : fd.Constant(0.01),
				'k1' : fd.Constant(0.005),
				'k2' : fd.Constant(0.00005),
				'lamd1' : fd.Constant(0.000005),
				'lamd2' : fd.Constant(0.0),
				'rho_s' : fd.Constant(1.),
				'L' :  fd.Constant(1.),
				'phi' : fd.Constant(0.3),
				'n' : fd.FacetNormal(self.mesh),
				'f' : fd.Constant((0.0, 0.0)),
				'nu' : fd.Constant(0.001),
				'frac' : fd.Constant(1.),
			}

	solver_parameters = recursive_update(solver_parameters,
	{
	'space': {'u': fd.VectorFunctionSpace, 'c': fd.MixedFunctionSpace, 'd' : fd.MixedFunctionSpace},
	'degree': {'u': 2},
	'order' : {'c': 3, 'd':3},
	'ksp_type': {'u': 'gmres', 'p': 'gmres',  'c': 'gmres', 'd':'gmres'},
	'precond': {'u': 'sor', 'p' : 'sor', 'c': 'sor', 'd':'gmres'},
	'dt' : 0.01,
	'T' : 0.1})

	#load mesh
	mesh = fd.Mesh("meshes/step1.msh")
	deltat = [0.01 / (2**i) for i in range(1)]

	# add subsystems for navier stokes and radio_transport
	solver = pde_solver([['u', 'p']], mesh, solver_parameters)
	# solver.add_subsystem(['cd', 'cs', 'as'], solver_parameters)
	solver.add_subsystem(['c', 'd'], solver_parameters)
	#setup constants
	solver.setup_constants()
	# define subsystems and variable sequence
	solver.define(['u', 'p', 'u'], 'up', navier_stokes)
	solver.define(['c', 'd'], 'cd', radio_transport_coupled_mms)
	# setup boundary conditions
	solver.setup_bcs()

	x, y, t = sy.symbols(('x', 'y', 't'))
	expr = sy.exp(x*y*t)
	solver.test_mms('c', expr, temporal=True, f_dict={"exp":fd.exp}, dt_list=deltat, plot=False, index=0)
