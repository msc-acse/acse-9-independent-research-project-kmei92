from collections import defaultdict
import copy
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


"""
The default_solver_parameters is a dictionary adapted from Mikael Mortensen on July, 2019.
Source code can be found here:
https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py
"""
default_solver_parameters = {
	'degree': defaultdict(lambda: 1), # default function space dimension
	'family': defaultdict(lambda: 'CG'), # default trial/test functions
	'linear_solver': defaultdict(lambda: ''),
	'order' : defaultdict(lambda: 1), # default order of function spaces
	'space' : defaultdict(lambda: fd.FunctionSpace), # default functionspace
	'subsystem_class': defaultdict(lambda: None), # Users MUST SPECIFY what class object (ex. navier stokes, reactions) to use for each subsystem
	'T': 10., # End time for simulation,
	'dt': 0.001  # timestep,
}


def recursive_update(dst, src):
	"""
	This function was obtained from Mikael Mortensen on July, 2019.
	Source code can be found here:
	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

	Description:
	Update dict dst with items from src deeply ("deep update").
	"""
	for key, val in src.items():
		if key in dst and isinstance(val, dict) and isinstance(dst[key], dict):
			dst[key] = recursive_update(dst[key], val)
		else:
			dst[key] = val

	return dst

class PDESystem:
	"""
	This class was adapted from Mikael Mortensen on July, 2019.
	Source code can be found here:
	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

	Description
	The PDESystem class contains all of the problem information:
	-mesh
	-solver parameters
	-subsystems
	-time (initial and end)
	-all variables to be solved for

	This class is responsible for setting up all of the required function spaces, specified by the user.

	"""
	def __init__(self, system_composition, mesh, parameters):      # removed problem(mesh)
		self.system_composition = system_composition         # Total system comp.
		self.system_names = []                               # Compounds solved for
		self.names = []                                      # All components
		self.mesh = mesh
		self.prm = parameters
		self.subsystem = self.prm['subsystem_class']

		self.t0 = 0                                           # start time
		self.tend = self.prm['T']                                         # simulation end time
		self.dt = self.prm['dt']                                           # time step

		self.pdesubsystems = dict((name, None) for name in self.system_names)
		self.bc = dict((name,   []) for name in self.system_names)

		self.setup_system()

	def setup_system(self):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function calls on other member functions of the PDESystem class to set up function spaces,
		the subsystems (depending on how many coupled systems of PDEs there are) and creates the form_args
		(the form_args are the variables that the PDESystem will solve for)
		"""
		for sub_system in self.system_composition:
			system_name = ''.join(sub_system)
			self.system_names.append(system_name)
			for n in sub_system:       # Run over all individual components
				self.names.append(n)

		self.define_function_spaces(self.mesh, self.prm['degree'], self.prm['space'],
		self.prm['order'], self.prm['family']) # removed cons
		self.setup_subsystems(self.prm['order'])
		self.setup_form_args(self.prm['order'])

	def define_function_spaces(self, mesh, degree, space, order, family):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function creates function spaces for each of the subsystems
		There is a check on whether the subsystem contains a MixedFunctionSpace
		"""
		V = {}
		for name in self.names:
			if order[name] > 1:
				total_space = []
				single_space = fd.FunctionSpace(mesh, family[name], degree[name])
				total_space = [single_space for i in range(order[name])]
				V.update(dict([(name, space[name](total_space))]))
			else:
				V.update(dict([(name, space[name](mesh, family[name], degree[name]))]))
		self.V = V

	def setup_subsystems(self, order):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function creates all of the trial and test functions to be used the the in the
		variational form. It is dependent on the variable names given to the PDESystem
		"""
		V, sys_comp = self.V, self.system_composition

		q = {}
		v = {}

		for sub_sys in sys_comp:
			for name in sub_sys:
				if order[name] > 1:
					q.update(dict( (name+'_trl%i'%(num), fd.TrialFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
					v.update(dict( (name+'_tst%i'%(num), fd.TestFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
				else:
					q.update(dict( [ (name+'_trl', fd.TrialFunction(V[name])) ] ) )
					v.update(dict( [ (name+'_tst', fd.TestFunction(V[name])) ] ) )

		self.qt, self.vt = q, v

	def setup_form_args(self, order):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function creates all of the trial and test functions to be used the the in the
		variational form. It is dependent on the variable names given to the PDESystem
		"""
		V, sys_comp = self.V, self.system_composition
		form_args = {}
		for sub_sys in sys_comp:
			for name in sub_sys:
				if order[name] > 1:
					form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
					form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
					form_args.update(dict( (name+'_%i'%(num), form_args[name+'_'][num-1]) for num in range(1, order[name]+1))) # gets individual components
					form_args.update(dict( (name+'_n%i'%(num), form_args[name+'_n'][num-1]) for num in range(1, order[name]+1))) # gets individual components
				else:
					form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
					form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current

		form_args.update(self.qt)
		form_args.update(self.vt)

		self.form_args = form_args

	def solve(self):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function calls on each of the available pdesubsystems and solve for their respective variables
		"""
		solvers = []
		sub_vars = []
		for name in self.system_names:
			for solver in self.pdesubsystems[name].solvers:
				solvers.append(solver)
			for var in self.pdesubsystems[name].vars:
				sub_vars.append(var)

		print('solvers: ', solvers)
		print('sub_vars: ', sub_vars)

		t0 = self.t0
		tend = self.tend
		dt = self.dt
		print('solving from master PDESystem')
		while t0 < tend:
			for i in range(len(solvers)):
				solvers[i].solve()

			for var in sub_vars:
				self.form_args[var+'_n'].assign(self.form_args[var+'_'])

			t0 += dt
			if( np.abs( t0 - np.round(t0,decimals=0) ) < 1.e-8):
				print('time = {0:.3f}'.format(t0))

	def add_subsystem(self, composition, parameters):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function allows users to create a new subsystem of PDEs with different variables within the overall
		PDESystem class. The function automatically calls on setup_systems() function to generate the new function
		spaces and form args
		"""

		if not isinstance(composition, list):
			temp = []
			temp.append(composition)
			composition = temp

		self.system_composition.append(composition)
		self.prm = recursive_update(self.prm, parameters)
		self.system_names = []
		self.names = []
		self.setup_system()
