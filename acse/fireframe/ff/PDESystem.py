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
	'ksp_type': defaultdict(lambda: ''),
	'precond': defaultdict(lambda: ''),
	'order' : defaultdict(lambda: 1), # default order of function spaces
	'space' : defaultdict(lambda: fd.FunctionSpace), # default functionspace
	'subsystem_class': defaultdict(lambda: None), # Users MUST SPECIFY what class object (ex. navier stokes, reactions) to use for each subsystem
	'T': 10., # End time for simulation,
	'dt': 0.001  # timestep,
}
solver_parameters  = copy.deepcopy(default_solver_parameters)



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
		self.constants = {}

		self.tstart = 0                                           			# start time
		self.t = fd.Constant(self.tstart)									# create a time variable
		self.tend = self.prm['T']                                         	# simulation end time
		self.dt = self.prm['dt']                                           	# time step

		self.pdesubsystems = dict((name, None) for name in self.system_names)
		self.bc = {}
		self.bc_expr = [[None, None]]
		self.var_seq = []

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

		self.bc.update(dict((name,   dict((name, [[None], [None]]) for name in self.names)) for name in self.system_names)) # setup boundary condition lists for each subsystem
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
		form_args = {'t': self.t} # create the time variable first
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

	def solve(self, time_update=False):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function calls on each of the available pdesubsystems and solve for their respective variables
		"""
		forms = []
		solvers = []
		''''sub_vars = []''''
		for name in self.system_names:
			'''for solver in self.pdesubsystems[name].solvers:
				solvers.append(solver)'''
			'''for var in self.pdesubsystems[name].vars:
				sub_vars.append(var)'''
			for key, form in self.pdesubsystems[name].F.items():
				forms.append(form)

		boundaries = [None] * len(self.var_seq) # create how many boundaries
		set1 = set(self.var_seq)
		abacus = dict.fromkeys(set1, 0)
		counter = 0

		for var in self.var_seq:
			for system in self.bc:
				if var in abacus and len(self.bc[system][var][0]) > abacus[var]:
					abacus[var] += 1
					boundaries[counter] = self.bc[syste][var][0][abacus[var]]
				counter += 1
		# print('solvers: ', solvers)
		# print('sub_vars: ', sub_vars)

		tstart = self.tstart
		tend = self.tend
		dt = self.dt
		print('solving from master PDESystem')
		print(self.bc['T']['T'])

		if time_update:
			while tstart < tend:
				# solve current timestep
				'''for i in range(len(solvers)):
					solvers[i].solve()'''
				# solve current timestep variables
				for i, var in enumerate(self.var_seq):
					fd.solve(forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i])
					if i < len(self.bc_expr):
						print('updated boundaries for %s' %var)
						# update boundary conditions
						boundaries[i] = [fd.DirichletBC(self.V[var], self.bc_expr[i][0], self.bc_expr[i][1])]
					'''counter = 0
					for system in self.bc:
						# print(counter)
						# print(self.bc_expr[0][1])
						# print(self.bc[system][var][1])
						print(self.bc[system][var][1])
						if var in self.bc[system] and self.bc[system][var][1][counter] == 'update':
							# print('updating %s' % var)
							self.bc[system][var][0] = [fd.DirichletBC(self.V[var], self.bc_expr[counter][0], self.bc_expr[counter][1])]
							counter += 1'''
				# assign next timestep variables
				for var in self.var_seq:
					self.form_args[var+'_n'].assign(self.form_args[var+'_'])
				# time step
				tstart += dt
				self.t.assign(tstart)

				#print progress
				if( np.abs( tstart - np.round(tstart,decimals=0) ) < 1.e-8):
					print('time = {0:.3f}'.format(tstart))
		else:
			while tstart < tend:
				# solve current timestep
				for i in range(len(solvers)):
					solvers[i].solve()
				# assign next timestep variables
				for var in sub_vars:
					self.form_args[var+'_n'].assign(self.form_args[var+'_'])


				# time step
				tstart += dt

				#print progress
				if( np.abs( tstart - np.round(tstart,decimals=0) ) < 1.e-8):
					print('time = {0:.3f}'.format(tstart))

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

		for sub_system in composition:
			system_name = ''.join(sub_system)
			self.bc.update(dict([(system_name,   dict((name, [[None], [None]]) for name in composition))])) # setup boundary condition lists for each subsystem
			self.system_names.extend(system_name)
			for n in sub_system:       # Run over all individual components
				self.names.extend(n)

		self.bc.update(dict((name,   dict((name, [[None], [None]]) for name in self.names)) for name in self.system_names)) # setup boundary condition lists for each subsystem
		self.define_function_spaces(self.mesh, self.prm['degree'], self.prm['space'],
		self.prm['order'], self.prm['family']) # removed cons
		self.setup_subsystems(self.prm['order'])
		self.setup_form_args(self.prm['order'])

	def setup_initial(self, var, expression, mixedspace=False, **kwargs):
		"""
		**kwargs
		index: the specific subspace of the MixedFunctionSpace to apply the initial condition
		"""
		if not mixedspace:
			self.form_args[var].interpolate(expression)
		else:
			split = self.form_args[var].split()
			split[kwargs['index']].interpolate(expression)

	def setup_constants(self, dictionary):
		"""
		params: dictionary - keys with the same name as the arguments used in the variational forms,
		values can be firedrake Constants, Conditionals, or other firedrake classes (ex. FacetNormal)
		"""
		self.constants.update(dictionary)

	def define(self, var_seq, name):
		self.var_seq.extend(var_seq)
		# try:
		# 	self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, name, self.constants, self.bc[name]) # index 0 for the list of boundary conditions
		# except:
		# 	print("self.subsystem[%s] is not defined" % (name))
		# print(self.bc[name])
		self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, name, self.constants, self.bc[name]) # index 0 for the list of boundary conditions
