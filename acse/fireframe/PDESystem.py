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

		self.setup_function_spaces(self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family']) # removed cons
		self.setup_trial_test(self.prm['order'])
		self.setup_form_args(self.prm['order'])

	def setup_function_spaces(self, mesh, degree, space, order, family):
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

	def update_function_spaces(self, name_list, mesh, degree, space, order, family):

		for name in name_list:
			if order[name] > 1:
				total_space = []
				single_space = fd.FunctionSpace(mesh, family[name], degree[name])
				total_space = [single_space for i in range(order[name])]
				self.V.update(dict([(name, space[name](total_space))]))
			else:
				self.V.update(dict([(name, space[name](mesh, family[name], degree[name]))]))

	def setup_trial_test(self, order):
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

	def update_trial_test(self, name_list, order):
		V = self.V
		for name in name_list:
			if order[name] > 1:
				self.qt.update(dict( (name+'_trl%i'%(num), fd.TrialFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
				self.vt.update(dict( (name+'_tst%i'%(num), fd.TestFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
			else:
				self.qt.update(dict( [ (name+'_trl', fd.TrialFunction(V[name])) ] ) )
				self.vt.update(dict( [ (name+'_tst', fd.TestFunction(V[name])) ] ) )

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

	def update_form_args(self, name_list, order):
		V = self.V
		for name in name_list:
			if order[name] > 1:
				self.form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				self.form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
				self.form_args.update(dict( (name+'_%i'%(num), self.form_args[name+'_'][num-1]) for num in range(1, order[name]+1))) # gets individual components
				self.form_args.update(dict( (name+'_n%i'%(num), self.form_args[name+'_n'][num-1]) for num in range(1, order[name]+1))) # gets individual components
			else:
				self.form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				self.form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current

		self.form_args.update(self.qt)
		self.form_args.update(self.vt)

	def obtain_forms(self):
		forms, a, L = [], [], []
		for subsystem in self.pdesubsystems:
			for key, form in self.pdesubsystems[subsystem].F.items():
				forms.append(form)
			for key, left in self.pdesubsystems[subsystem].a.items():
				a.append(left)
			for key, right in self.pdesubsystems[subsystem].L.items():
				L.append(right)

		self.forms, self.a, self.L = forms, a, L

	def solve(self, time_update=False):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function calls on each of the available pdesubsystems and solve for their respective variables
		"""

		self.obtain_forms()

		boundaries = [] # create how many boundaries
		for var in self.var_seq:
			boundaries.extend([None]*self.prm['order'][var])

		abacus = dict.fromkeys(set(self.var_seq), 0)
		# print(boundaries)
		# print(abacus)
		# print(self.bc['cd'])
		counter = 0
		# print(self.var_seq)
		for var in self.var_seq:
			for i in range(self.prm['order'][var]):
				# print(abacus[var], len(self.bc[var]))
				if len(self.bc[var]) > abacus[var]:
					# print(var, counter)
					boundaries[counter] = self.bc[var][abacus[var]][0]
					abacus[var] += 1
					counter += 1

		# print(boundaries)
		# print('solvers: ', solvers)
		# print('sub_vars: ', sub_vars)

		tstart = self.tstart
		tend = self.tend
		dt = self.dt

		if time_update:
			while tstart < tend:
				# solve current timestep variables
				abacus = dict.fromkeys(set(self.var_seq), 0)
				for i, var in enumerate(self.var_seq):

					try:
						if self.prm['order'][var] > 1: # if mixedfunctionspace
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i])
						elif var in self.prm['ksp_type'] and var not in self.prm['precond']:
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var]})
						elif var not in self.prm['ksp_type'] and var in self.prm['precond']:
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'pc_type': self.prm['precond'][var]})
						else:
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var], 'pc_type': self.prm['precond'][var]})
					except:
						if self.prm['order'][var] > 1: # if mixedfunctionspace
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i])
						elif var in self.prm['ksp_type'] and var not in self.prm['precond']:
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var]})
						elif var not in self.prm['ksp_type'] and var in self.prm['precond']:
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'pc_type': self.prm['precond'][var]})
						else:
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var], 'pc_type': self.prm['precond'][var]})

					# check if boundary conditions need to be updated
					if self.bc[var][abacus[var]][-1] == 'update':
						# print('updated boundaries for %s' %var)
						if self.prm['order'][var] > 1:
							boundaries[i] = [fd.DirichletBC(self.V[var].sub(self.bc[var][abacus[var]][3]), self.bc[var][abacus[var]][1], self.bc[var][abacus[var]][2])]
						else:
							boundaries[i] = [fd.DirichletBC(self.V[var], self.bc[var][abacus[var]][1], self.bc[var][abacus[var]][2])]

					abacus[var] += 1
					# assign next timestep variables
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
				for i, var in enumerate(self.var_seq):
					try:
						if self.prm['order'][var] > 1: # if mixedfunctionspace
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i])
						elif var in self.prm['ksp_type'] and var not in self.prm['precond']:
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var]})
						elif var not in self.prm['ksp_type'] and var in self.prm['precond']:
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'pc_type': self.prm['precond'][var]})
						else:
							fd.solve(self.forms[i] == 0, self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var], 'pc_type': self.prm['precond'][var]})
					except:
						if self.prm['order'][var] > 1: # if mixedfunctionspace
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i])
						elif var in self.prm['ksp_type'] and var not in self.prm['precond']:
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var]})
						elif var not in self.prm['ksp_type'] and var in self.prm['precond']:
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'pc_type': self.prm['precond'][var]})
						else:
							fd.solve(self.a[i] == self.L[i], self.form_args[var+'_'], bcs=boundaries[i], solver_parameters={'ksp_type': self.prm['linear_solver'][var], 'pc_type': self.prm['precond'][var]})

				# assign next timestep variables
				for var in self.var_seq:
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

		:arg composition: `list`
		"""

		if not isinstance(composition, list):
			temp = []
			temp.append(composition)
			composition = temp

		self.system_composition.append(composition)
		self.prm = recursive_update(self.prm, parameters)

		system_name = ''.join(composition)
		self.system_names.extend(system_name)
		for n in composition:       # Run over all individual components
			self.names.extend(n)

		self.update_function_spaces(composition, self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family'])
		self.update_trial_test(composition, self.prm['order'])
		self.update_form_args(composition, self.prm['order'])

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
		self.bc.update(dict((name, [[None, None, None, None, None]] * self.prm['order'][name] * self.var_seq.count(name)) for name in self.var_seq)) # setup boundary condition lists for each subsystem
		# print(self.bc)
		# try:
		# 	self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, name, self.constants, self.bc[name]) # index 0 for the list of boundary conditions
		# except:
		# 	print("self.subsystem[%s] is not defined" % (name))
		# print(self.bc[name])
		self.pdesubsystems[name] = self.subsystem[name](vars(self), var_seq, self.constants) # index 0 for the list of boundary conditions
