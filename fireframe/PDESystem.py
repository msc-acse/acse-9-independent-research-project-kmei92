"""
author : Keer Mei
email: keer.mei18@imperial.ac.uk
github username: kmei92
"""
from collections import defaultdict
import copy
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy


"""
The default_solver_parameters is a dictionary adapted from Mikael Mortensen on July, 2019.
Source code can be found here:
https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py
"""
default_solver_parameters = {
	'degree': defaultdict(lambda: 1), # default function space dimension
	'family': defaultdict(lambda: 'CG'), # default trial/test functions
	'ksp_type': defaultdict(lambda: ''), # specifies iterative method for solving matrices
	'precond': defaultdict(lambda: ''), # specifies matrix preconditioners
	'order' : defaultdict(lambda: 1), # default order of function spaces
	'space' : defaultdict(lambda: fd.FunctionSpace), # default functionspace
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

	Description:
	The PDESystem class represents a complete set of Partial Differential Equations.
	A PDESystem can include a single set of PDEs, for example the Chorin projection
	scheme for the navier-stokes equation. Or, the PDESystem can contain a coupled
	set of PDEs, for example a radionuclide problem coupled with the Chorin projection
	equations.

	:param system_composition: a list of list. i.e. [['u', 'p']] for navier-stokes
	:type system_composition: `list` of `list`
	Example:
		[['u', 'p']] a system with two variables, u and p.
		[['up']] a system with one variable, up. Most likely a MixedFunctionSpace.
		[['u'], ['p']] two coupled systems with single variable.


	:param mesh: a firedrake mesh object
	:type mesh: `fd.Mesh object`

	:param parameters: see default_sovler_parameters
	:type parameters: `dictionary` of `dictionary`

	:attribute names: a list of variables. i.e. ['u', 'p']
	:attribute system_names: a list of system names. i.e. ['up', 'cdcsas']
	:attribute subsystem: a dictionary of available subclasses. key = system_name, value = PDESubsystem object. i.e. subsystem['up'] = navier_stokes
	:attribute constants: a dictionary of constants. To be passed together with form_args. used in the forms of PDESubsystem objects.
	:attribute tstart: (int or float) default 0. Start time of problem
	:attribute t: firedrake Constant object. Time variable of the problem
	:attribute tend: (int or float) parameters['T'] value. End time of problem
	:attribute dt: (int or float) parameters['dt']. Time step value of problem
	"""
	def __init__(self, system_composition, mesh, parameters):
		self.system_composition = system_composition   	#ex. [['u', 'p']]
		self.system_names = []                         	#ex. ['up', 'c']
		self.names = []                                	#ex. ['u', 'p', 'c']
		self.mesh = mesh								#fd.Mesh object
		self.prm = parameters							#solver or default_solver_parameters
		self.constants = {}

		self.tstart = 0                                 # start time
		self.t = fd.Constant(self.tstart)				# create a time variable
		self.tend = self.prm['T']                       # simulation end time
		self.dt = self.prm['dt']                        # time step

		# initialize an empty dictionary to keep track of PDESubsystems
		self.pdesubsystems = dict((name, None) for name in self.system_names)
		# initialize empty boundary conditions
		self.bc = {}
		self.var_seq = []								#ex. ['u', 'p', 'c']

		# creates function spaces, trial, test, and functions
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

		:returns self.names: a list of variables
		:rtype: `list`

		:returns self.system_names: a list of system variables
		:rtype: `list`
		"""
		# keep track of each subsystem
		for sub_system in self.system_composition:
			# create a system name from all variables in subsystem
			system_name = ''.join(sub_system)
			self.system_names.append(system_name)
			# track individual, UNIQUE names into self.names list
			for n in sub_system:       # Run over all individual components
				self.names.append(n)
		# create function spaces
		self.setup_function_spaces(self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family']) # removed cons
		# create trial and test functions
		self.setup_trial_test(self.prm['order'], self.prm['space'])
		# create arguments to be passed into PDESubsystems
		self.setup_form_args(self.prm['order'], self.prm['space'])

	def setup_function_spaces(self, mesh, degree, space, order, family):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function creates function spaces for each of the subsystems
		There is a check on whether the subsystem contains a MixedFunctionSpace

		:params mesh: A Firedrake mesh object
		:type mesh: `fd.Mesh object`0

		:params degree: A dictionary with keys of variable names and values of integers. i.e. {'u' : 1, 'p' : 1}
		:type degree: `dictionary`

		:params space: A dictionary with keys of variable names and values of firedrake function space methods. i.e. {'u' : fd.VectorFunctionSpace}
		:type space: `dictionary`

		:params order: A dictionary with keys of variable names and values of integers. i.e. {'u' : 1, 'p' : 1}
		:type order: `dictionary`

		:params family: A dictionary with keys of variable names and values of Firedrake finite element familys. i.e. {'u' : 'DG', 'p' : 'CG'}
		:type family: `dictionary`

		:returns self.V: A dictionary with keys of variable names and values of Firedrake function space objects. i.e. {'u': fd.VectorFunctionSpace, 'p': fd.FunctionSpace, 'cd' : fd.MixedFunctionSpace}
		:rtype self.V: `dictionary`
		"""
		# empty dictionary
		V = {}
		# for each variable
		for name in self.names:
			# order > 1 should be MixedFunctionSpace of generic function spaces
			if order[name] > 1:
				# create individual fd.FunctionSpaces multipled by total order#
				total_space = []
				single_space = fd.FunctionSpace(mesh, family[name], degree[name])
				total_space = [single_space for i in range(order[name])]
				V.update(dict([(name, space[name](total_space))]))
			# special scenario for user specified MixedLists
			elif isinstance(space[name], list) :
				total_space = [0] * len(space[name])
				if isinstance(degree[name], list) and isinstance(family[name], list):
					# check that the user has input a correct list
					assert len(degree[name]) == len(space[name])
					assert len(family[name]) == len(space[name])
					for i in range(len(space[name])):
						total_space[i] = space[name][i](mesh, family[name][i], degree[name][i])
				elif isinstance(degree[name], list):
					# check that the user has input a correct list
					assert len(degree[name]) == len(space[name])
					for i in range(len(space[name])):
						total_space[i] = space[name][i](mesh, family[name], degree[name][i])
				elif isinstance(family[name], list):
					# check that the user has input a correct list
					assert len(family[name]) == len(space[name])
					for i in range(len(space[name])):
						total_space[i] = space[name][i](mesh, family[name][i], degree[name])
				V.update(dict([(name, fd.MixedFunctionSpace(total_space))]))
			# use the parameters['space'] value to determine type of function space
			else:
				V.update(dict([(name, space[name](mesh, family[name], degree[name]))]))
		# initialize new attribute self.V to track all created function spaces
		self.V = V

	def update_function_spaces(self, name_list, mesh, degree, space, order, family):
		"""
		Description:
		This function updates function spaces when a new subsystem is added to
		the PDESystem.
		There is a check on whether the subsystem contains a MixedFunctionSpace

		:params mesh: A Firedrake mesh object
		:type mesh: `fd.Mesh object`

		:params degree: A dictionary with keys of variable names and values of integers. i.e. {'u' : 1, 'p' : 1}
		:type degree: `dictionary`

		:params space: A dictionary with keys of variable names and values of firedrake function space methods. i.e. {'u' : fd.VectorFunctionSpace}
		:type space: `dictionary`

		:params order: A dictionary with keys of variable names and values of integers. i.e. {'u' : 1, 'p' : 1}
		:type order: `dictionary`

		:params family: A dictionary with keys of variable names and values of Firedrake finite element familys. i.e. {'u' : 'DG', 'p' : 'CG'}
		:type family: `dictionary`

		:returns self.V: A dictionary with keys of variable names and values of Firedrake function space objects. i.e. {'u': fd.VectorFunctionSpace, 'p': fd.FunctionSpace, 'cd' : fd.MixedFunctionSpace}
		:rtype self.V: `dictionary`
		"""
		# for each variable in new system
		for name in name_list:
			# order > 1 should be MixedFunctionSpace
			if order[name] > 1:
				# create individual fd.FunctionSpaces multipled by total order#
				total_space = []
				single_space = fd.FunctionSpace(mesh, family[name], degree[name])
				total_space = [single_space for i in range(order[name])]
				self.V.update(dict([(name, space[name](total_space))]))
			# special scenario for MixedLists
			elif isinstance(space[name], list):
				total_space = [0] * len(space[name])
				if isinstance(degree[name], list) and isinstance(family[name], list):
					# check that the user has input a correct list
					assert len(degree[name]) == len(space[name])
					assert len(family[name]) == len(space[name])
					for i in range(len(space[name])):
						total_space[i] = space[name][i](mesh, family[name][i], degree[name][i])
				elif isinstance(degree[name], list):
					# check that the user has input a correct list
					assert len(degree[name]) == len(space[name])
					for i in range(len(space[name])):
						total_space[i] = space[name][i](mesh, family[name], degree[name][i])
				elif isinstance(family[name], list):
					# check that the user has input a correct list
					assert len(family[name]) == len(space[name])
					for i in range(len(space[name])):
						total_space[i] = space[name][i](mesh, family[name][i], degree[name])
				self.V.update(dict([(name, fd.MixedFunctionSpace(total_space))]))
			# use the parameters['space'] value to determine type of function space
			else:
				self.V.update(dict([(name, space[name](mesh, family[name], degree[name]))]))

	def setup_trial_test(self, order, space):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function creates all of the trial and test functions to be used the the in the
		variational form. It is dependent on the variable names given to the PDESystem

		:params order: A dictionary with keys of variable names and values of integers. i.e. {'c' : 3}
		:type order: `dictionary`

		:returns self.qt: A dictionary with keys of variable names + suffix '_trl' and Firedrake TrialFunction or TrialFunctions objects. i.e. {'u_trl', fd.TrialFunction}
		:rtype self.qt: `dictionary`

		:returns self.vt: A dictionary with keys of variable names + suffix '_tst' and Firedrake TestFunction or TestFunctions objects. i.e. {'u_trl', fd.TestFunction}
		:rtype self.vt: `dictionary`
		"""
		V, names = self.V, self.names
		# initialize two new dictionarys to track trial and test functions
		q = {}
		v = {}

		# for each variable
		for name in names:
			# if MixedFunctionSpace use indexing to extract subcomponents
			if order[name] > 1:
				q.update(dict( (name+'_trl%i'%(num), fd.TrialFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
				v.update(dict( (name+'_tst%i'%(num), fd.TestFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
			# if special MixedLists
			elif isinstance(space[name], list):
				q.update(dict( (name+'_trl%i'%(num), fd.TrialFunctions(V[name])[num-1]) for num in range(1, len(space[name])+1)))
				v.update(dict( (name+'_tst%i'%(num), fd.TestFunctions(V[name])[num-1]) for num in range(1, len(space[name])+1)))
			# create Trial and Test
			else:
				q.update(dict( [ (name+'_trl', fd.TrialFunction(V[name])) ] ) )
				v.update(dict( [ (name+'_tst', fd.TestFunction(V[name])) ] ) )
		# create new attribute self.qt and self.vt for trial and test functions
		self.qt, self.vt = q, v

	def update_trial_test(self, name_list, order, space):
		"""
		Description:
		This function updates self.qt and self.vt when a new subsystem is added.
		It is dependent on the variable names given to the PDESystem.

		:params name_list: A list of variables. i.e. ['cd', 'cs', 'as']
		:type name_list: `list`

		:params order: A dictionary with keys of variable names and values of integers. i.e. {'c' : 3}
		:type order: `dictionary`

		:returns self.qt: A dictionary with keys of variable names + suffix '_trl' and Firedrake TrialFunction or TrialFunctions objects. i.e. {'u_trl', fd.TrialFunction}
		:rtype self.qt: `dictionary`

		:returns self.vt: A dictionary with keys of variable names + suffix '_tst' and Firedrake TestFunction or TestFunctions objects. i.e. {'u_trl', fd.TestFunction}
		:rtype self.vt: `dictionary`
		"""

		V = self.V
		# for each new variable
		for name in name_list:
			# if MixedFunctionSpace use indexing to extract subcomponents
			if order[name] > 1:
				self.qt.update(dict( (name+'_trl%i'%(num), fd.TrialFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
				self.vt.update(dict( (name+'_tst%i'%(num), fd.TestFunctions(V[name])[num-1]) for num in range(1, order[name]+1)))
			# if special MixedLists
			elif isinstance(space[name], list):
				self.qt.update(dict( (name+'_trl%i'%(num), fd.TrialFunctions(V[name])[num-1]) for num in range(1,  len(space[name])+1)))
				self.vt.update(dict( (name+'_tst%i'%(num), fd.TestFunctions(V[name])[num-1]) for num in range(1,  len(space[name])+1)))
			# create Trial and Test
			else:
				self.qt.update(dict( [ (name+'_trl', fd.TrialFunction(V[name])) ] ) )
				self.vt.update(dict( [ (name+'_tst', fd.TestFunction(V[name])) ] ) )

	def setup_form_args(self, order, space):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function creates all of the current and future iterations of variables
		Functions to be used the the in the variational form.  It also retrieves
		self.qt and self.vt and combines all Functions, TrialFunctions and
		TestFunctions into one dictioanry.
		It is dependent on the variable names given to the PDESystem.

		:params order: A dictionary with keys of variable names and values of integers. i.e. {'c' : 3}
		:type order: `dictionary`

		:returns self.form_args: A dictionary with keys of variable names + suffix '_trl' and Firedrake TrialFunction or TrialFunctions objects. i.e. {'u_': fd.Function, 'u_n': fd.Function, 'u_trl':fd.TrialFunction, 'u_tst' : fd.TestFunction}
		:rtype self.form_args: `dictionary`
		"""
		V, names = self.V, self.names
		form_args = {'t': self.t} # create the time variable first
		# for each variable in system
		for name in names:
			# if MixedFunctionSpace, extract the individual components
			if order[name] > 1:
				# create current iteration '_n' and future iteration '_' functions
				form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
				form_args.update(dict( (name+'_%i'%(num), form_args[name+'_'][num-1]) for num in range(1, order[name]+1))) # gets individual components
				form_args.update(dict( (name+'_n%i'%(num), form_args[name+'_n'][num-1]) for num in range(1, order[name]+1))) # gets individual components
			# if special MixedLists
			elif isinstance(space[name], list):
				# create current iteration '_n' and future iteration '_' functions
				form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
				form_args.update(dict( (name+'_%i'%(num), form_args[name+'_'].split()[num-1]) for num in range(1, len(space[name])+1))) # gets individual components
				form_args.update(dict( (name+'_n%i'%(num), form_args[name+'_n'].split()[num-1]) for num in range(1, len(space[name])+1))) # gets individual components
			# create current iteration '_n' and future iteration '_' functions
			else:
				form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
		# store all functions into a dictionary form_args
		form_args.update(self.qt)
		form_args.update(self.vt)

		# create new attribute self.form_args to store all function dictionaries
		self.form_args = form_args

	def update_form_args(self, name_list, order, space):
		"""
		Description:
		This function updates self.form_args dictionary when a new subsystem
		is added to the PDESystem.

		:params order: A dictionary with keys of variable names and values of integers. i.e. {'c' : 3}
		:type order: `dictionary`

		:returns self.form_args: A dictionary with keys of variable names + suffix '_trl' and Firedrake TrialFunction or TrialFunctions objects. i.e. {'u_': fd.Function, 'u_n': fd.Function, 'u_trl':fd.TrialFunction, 'u_tst' : fd.TestFunction}
		:rtype self.form_args: `dictionary`
		"""
		V = self.V
		# for each new variable
		for name in name_list:
			# if MixedFunctionSpace, extract the individual components
			if order[name] > 1:
				# create current iteration '_n' and future iteration '_' functions
				self.form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				self.form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
				self.form_args.update(dict( (name+'_%i'%(num), self.form_args[name+'_'][num-1]) for num in range(1, order[name]+1))) # gets individual components
				self.form_args.update(dict( (name+'_n%i'%(num), self.form_args[name+'_n'][num-1]) for num in range(1, order[name]+1))) # gets individual components
			# if special MixedLists
			elif isinstance(space[name], list):
				# create current iteration '_n' and future iteration '_' functions
				self.form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				self.form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current
				self.form_args.update(dict( (name+'_%i'%(num), form_args[name+'_'].split()[num-1]) for num in range(1, len(space[name])+1))) # gets individual components
				self.form_args.update(dict( (name+'_n%i'%(num), form_args[name+'_n'].split()[num-1]) for num in range(1, len(space[name])+1))) # gets individual components
			# create current iteration '_n' and future iteration '_' functions
			else:
				self.form_args.update(dict( [ (name + '_', fd.Function(V[name])) ] ) ) # next iteration
				self.form_args.update(dict( [ (name + '_n', fd.Function(V[name])) ] ) ) # current

		self.form_args.update(self.qt)
		self.form_args.update(self.vt)

	def obtain_forms(self):
		"""
		This function retrieves the variational forms, LHS, and RHS
		from the PDESubsystem objects. This function should is called in solve.
		The solve function should only be called once each subsystem has been
		defined!

		:returns forms: A list of UFL expressions of the variational forms of the underlying PDEs.
		:rtypes forms: `list`

		:returns a: A list of UFL expressions of the LHS of the variational forms.
		:rtypes a: `list`

		:returns L: A list of UFL expressions of the RHS of the underlying variational forms.
		:rtypes L: `list`

		:returns linear_solve: A list of strings to be called using eval() in the solve function. i.e. fd.solve(a == L)
		:rtypes linear_solve: `list`

		:returns nonlinear_solve: A list of strings to be called using eval() in the solve function. i.e. fd.solve(F==0)
		:rtypes nonlinear_solve: `list`
		"""
		forms, a, L = [], [], []
		# for each subsystem
		for subsystem in self.pdesubsystems:
			# retrieve variational form
			for key, form in self.pdesubsystems[subsystem].F.items():
				forms.append(form)
			# retrieve LHS
			for key, left in self.pdesubsystems[subsystem].a.items():
				a.append(left)
			# retrive RHS
			for key, right in self.pdesubsystems[subsystem].L.items():
				L.append(right)

		# create new attributes, forms, a, L
		self.forms, self.a, self.L = forms, a, L

		# use string formatting to setup solvers. This method has the advantage
		# of not relying on in time evaluation of parameters to determine
		# adequate solve function
		linear_solve = []
		nonlinear_solve = []
		# for each variable
		for i, var in enumerate(self.var_seq):
			if len(self.V[var]) > 1 : # if mixedfunctionspace
				linear_solve.append("fd.solve(self.a[%d] == self.L[%d], self.form_args[%r], bcs=boundaries[%d])" % (i, i, var+'_', i))
				nonlinear_solve.append("fd.solve(self.forms[%d] == 0, self.form_args[%r], bcs=boundaries[%d])" % (i, var+'_', i))
			# if neither are specified
			elif var not in self.prm['ksp_type'] and var not in self.prm['precond']:
				linear_solve.append("fd.solve(self.a[%d] == self.L[%d], self.form_args[%r], bcs=boundaries[%d])" % (i, i, var+'_', i))
				nonlinear_solve.append("fd.solve(self.forms[%d] == 0, self.form_args[%r], bcs=boundaries[%d])" % (i, var+'_', i))
			# if users have specified a preconditioner but not an iterative method
			elif var in self.prm['ksp_type'] and var not in self.prm['precond']:
				linear_solve.append("fd.solve(self.a[%d] == self.L[%d], self.form_args[%r], bcs=boundaries[%d], solver_parameters={'ksp_type': self.prm['ksp_type'][%r]})" % (i, i, var+'_', i, var))
				nonlinear_solve.append("fd.solve(self.forms[%d] == 0, self.form_args[%r], bcs=boundaries[%d], solver_parameters={'ksp_type': self.prm['ksp_type'][%r]})" % (i, var+'_', i, var))
			# if users have specified an iterative method but not a preconditioner
			elif var not in self.prm['ksp_type'] and var in self.prm['precond']:
				linear_solve.append("fd.solve(self.a[%d] == self.L[%d], self.form_args[%r], bcs=boundaries[%d], solver_parameters={'pc_type': self.prm['precond'][%r]})" % (i, i, var+'_', i, var))
				nonlinear_solve.append("fd.solve(self.forms[%d] == 0, self.form_args[%r], bcs=boundaries[%d], solver_parameters={'pc_type': self.prm['precond'][%r]})" % (i, var+'_', i, var))
			# if both are specified
			else:
				linear_solve.append("fd.solve(self.a[%d] == self.L[%d], self.form_args[%r], bcs=boundaries[%d], solver_parameters={'ksp_type': self.prm['ksp_type'][%r], 'pc_type': self.prm['precond'][%r]})" %(i, i, var+'_', i, var, var))
				nonlinear_solve.append("fd.solve(self.forms[%d] == 0, self.form_args[%r], bcs=boundaries[%d], solver_parameters={'ksp_type': self.prm['ksp_type'][%r], 'pc_type': self.prm['precond'][%r]})" %(i, var+'_', i, var, var))

		self.linear_solve = linear_solve
		self.nonlinear_solve = nonlinear_solve

	def solve(self, time_update=False, save_vars=[]):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESystem.py

		Description:
		This function calls on each of the available pdesubsystems and solve for their respective variables

		:params time_update: A switch to determine whether this problem has time dependent boundary conditions and variables. select True if yes.
		:type time_update: `bool`

		:returns: None. Solution functions are stored in and updated in the self.form_args attribute. This solve function only updates the values in self.form_args.
		:rtype: None
		"""
		# call on the obtain form function to retrieve forms from PDESubsystems
		self.obtain_forms()

		# for saving, create different files for each variable specified
		outfiles = dict.fromkeys(save_vars, None)
		for var in save_vars:
			outfiles[var] = fd.File("outputs/%s.pvd" % var)
			# output the initial variable data
			outfiles[var].write(self.form_args[var+'_n'])

		# create boundaries list to be used when solving
		boundaries = [] # create how many boundaries

		abacus = dict.fromkeys(self.names, 0)
		# this loop will take the specified boundaries dictionary and add
		# relatvant boundaries for each variable in the order that they are solved
		# the abacus keeps track of repeated variables
		for i, var in enumerate(self.var_seq):
			# create a boundary condition for every form
			boundaries.append(self.bc[var][abacus[var]][0])
			abacus[var] += 1
			# for all subsequent subspaces, instead of adding a new list,
			# merge together all of the boundary conditions
			if len(self.V[var]) > 1:
				for j in range(1, len(self.V[var])):
					boundaries[i].extend(self.bc[var][abacus[var]][0])
					abacus[var]+= 1
		tstart = self.tstart
		tend = self.tend
		dt = self.dt
		# if there is a time updated condition and boundaries need to be updated
		if time_update:
			while tstart < tend:
				# solve current timestep variables
				# use the abacus function again in order to keep track of
				# repeated variables and boundary conditions
				abacus = dict.fromkeys(set(self.var_seq), 0)
				for i, var in enumerate(self.var_seq):
					# first try the linear solve methods
					try:
						eval(self.linear_solve[i])
					# if failed, try the nonlinear solve methods
					except:
						eval(self.nonlinear_solve[i])
					# check if boundary conditions need to be updated
					for j in range(len(self.V[var])):
						index = abacus[var] * len(self.V[var]) + j
						# check to see if boundaries need to be updated
						if self.bc[var][index][-1] == 'update':
							# check to see how many different boundaries need to be applied
							if isinstance(self.bc[var][index][1], list):
								# check to see user has input correct boundary specs
								if len(self.V[var]) > 1:
									assert len(self.bc[var][index][1]) == len(self.bc[var][index][2]) and len(self.bc[var][index][2]) == len(self.bc[var][index][3])
									boundaries[i] = list(fd.DirichletBC(self.V[var].sub(self.bc[var][index][3][k]), self.bc[var][index][1][k], self.bc[var][index][2][k]) for k in range(len(self.bc[var][index][2])))
								else:
									assert len(self.bc[var][index][1]) == len(self.bc[var][index][2])
									boundaries[i] = list(fd.DirichletBC(self.V[var], self.bc[var][index][1][k], self.bc[var][index][2][k]) for k in range(len(self.bc[var][index][2])))
							else:
								# if a MixedFunctionSpace, check to see which subspaces
								# require a boundary condition and update
								if len(self.V[var]) > 1 and len(boundaries[i]) > 1:
									boundaries[i][j] = fd.DirichletBC(self.V[var].sub(self.bc[var][index][3]), self.bc[var][index][1], self.bc[var][index][2])
								elif len(self.V[var]) > 1 and len(boundaries[i]) == 1:
									boundaries[i] = [fd.DirichletBC(self.V[var].sub(self.bc[var][index][3]), self.bc[var][index][1], self.bc[var][index][2])]
								else:
									boundaries[i] = [fd.DirichletBC(self.V[var], self.bc[var][index][1], self.bc[var][index][2])]
					abacus[var] += 1
				for var in self.var_seq:
					# assign next timestep variables
					self.form_args[var+'_n'].assign(self.form_args[var+'_'])
					# write current timestep variables
					if var in save_vars:
						if abacus[var] == self.var_seq.count(var):
							outfiles[var].write(self.form_args[var+'_n'])
				# increment time step
				tstart += dt
				# assign the time variable to the new value
				self.t.assign(tstart)
				if( np.abs( tstart - np.round(tstart,decimals=0) ) < 1.e-8):
					print('time = {0:.3f}'.format(tstart))
		# if there is no need to update boundary conditions
		else:
			while tstart < tend:
				# solve current timestep variables
				# use abacus to check for repeated variables
				abacus = dict.fromkeys(set(self.var_seq), 0)
				for i, var in enumerate(self.var_seq):
					# try the linear solve methods
					try:
						eval(self.linear_solve[i])
					# try nonlinear solve methods
					except:
						eval(self.nonlinear_solve[i])
					abacus[var] += 1
				# assign next timestep variables
				for var in self.var_seq:
					self.form_args[var+'_n'].assign(self.form_args[var+'_'])
					# write current timestep variables
					if var in save_vars:
						if abacus[var] == self.var_seq.count(var):
							outfiles[var].write(self.form_args[var+'_n'])
				# increment time step
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
		PDESystem class. The function automatically calls a series of update functions to generate the new function
		spaces and form args

		:params composition: A list of variable names to be added to the PDESystem. i.e. ['cd', 'cs', 'as']
		:type composition: `list`.

		:params parameters: A dictionary of parameters with keys of the new composition to be added. i.e. {'family' : {'cd' : }, ...}
		:type parameters: `dictionary`
		"""
		# check to see if the new system variables added are in a list and if not,
		# make a list
		# relevant if users only wish to pass in one variable like 'c'
		if not isinstance(composition, list):
			temp = []
			temp.append(composition)
			composition = temp

		# add the new variables
		self.system_composition.append(composition)
		# update the parameters dictionary with the new variables and values
		self.prm = recursive_update(self.prm, parameters)

		# update the system names, and the variables
		system_name = ''.join(composition)
		self.system_names.append(system_name)
		for n in composition:       # Run over all individual components
			self.names.append(n)

		# update the function spaces
		self.update_function_spaces(composition, self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family'])
		# create new variable trial and test functions
		self.update_trial_test(composition, self.prm['order'], self.prm['space'])
		# create new variable current and next iteration functions
		self.update_form_args(composition, self.prm['order'], self.prm['space'])

	def setup_initial(self, var, expression, mixedspace=False, **kwargs):
		"""
		This function interpolates an expression onto a variable. Used for setting
		up initalized Functions.

		:params var: variable name. The key to the Function to be interpolated. i.e. 'cd'
		:type var: `str`

		:params expression: a Firedrake expression. NOTE, this is not the same as a firedrake.Expression object. expression here can include firedrake Constants(), or firedrake conditionals(). i.e. fd.exp(x*y*t)
		:type expression: `see description`

		:params mixedspace: a switch to determine whether the variable is a MixedFunctionSpace or Taylorhood.
		:type mixedspace: `bool`

		:kwargs:
			index: `int`. the specific subspace of the MixedFunctionSpace to
			apply the initial condition. i.e. index = 0 for the first component
			of the MixedFunctionSpace
		"""
		if not mixedspace:
			self.form_args[var].interpolate(expression)
		else:
			split = self.form_args[var].split()
			split[kwargs['index']].interpolate(expression)

	def setup_constants(self, dictionary):
		"""
		This function sets up a constants dictionary to be passed together with
		self.form_args. The constants are dependent on the variable names
		expressed in the forms function of PDESubsystems.

		:params dictionary: keys with the same name as the arguments used in the variational forms, values can be firedrake Constants, Conditionals, or other firedrake classes (ex. FacetNormal) i.e. {'k' : fd.Constant(self.prm['dt'])}
		:type dictionary: `dictionary`

		:returns constants: a dictionary containing all of the constants.
		:rtype constants: `dictionary`
		"""
		self.constants.update(dictionary)

	def define(self, var_seq, name, subsystem):
		"""
		This function specifies PDESubsystem objects for each individual subsystem
		inside the overall PDESystem. This function must be called before setting
		up boundary conditions as this function creates the boundary conditions
		dictionary.

		:params var_seq: A list of the variable sequence that a subsystem should solve for. The length of this list should be equivalent to the number of forms in the PDESubsystem specified for this subsystem. i.e. ['u', 'p', 'u'] for the Chorin projection scheme.
		:type var_seq: `list`

		:params name: The name of the subsystem that is defined. i.e. 'up'
		:type name: `str`

		:return self.bc: A dictionary of all of the variables in the var_seq.
		:rtype self.bc: `dictionary`

		:return self.pdesystems: a dictionary of
		:rtype self.pdesystems: `dictionary`
		"""
		# the variable sequqnece is the sequence that the PDESubsystems forms
		# are designed to solve for
		self.var_seq.extend(var_seq)
		# create boundary conditions for each variable.
		# if repeated variables exist, the dictionary will simply override the previous
		# NOTE: the VALUES in the dictionary contain 5 different solver_parameters
		# index 0 should be fd.DirichletBC objects
		# index 1 should be expressions, such as fd.exp(x*y*t) or conditionals
		# such as fd.Conditional()
		# index 2 refers to the boundary which to apply the condition. ex. 'on_boundary'
		# index 3 should be either 'fixed' or 'update'. This tells the solver whether to
		# update the boundary conditin
		# index 4 tells the solver which indexed subspace this boundary condition should
		# be applied to. ex. 0 for the first index of a MixedFunctionSpace
		for var in var_seq:
			if isinstance(self.prm['space'][name], list):
				self.bc.update(dict([(var, [[[], None, None, None, None]] * len(self.prm['space'][name]))]))
			else:
				self.bc.update(dict([(var, [[[], None, None, None, None]] * self.prm['order'][var] * self.var_seq.count(var))]))
		# initialize the subsystem dictionary with PDESubsystem objects
		self.pdesubsystems[name] = subsystem(vars(self), var_seq)

	def test_mms(self, var, expr, spatial=False, temporal=False, f_dict = {}, dt_list=[], meshes=[], plot=False, index=None):
		"""
		Description
		This function allows users to test a PDESystem's variable by asking for a manufactured
		analytical solution. This function can either be set to test against spatial or temporal
		discretization.

		:params var: the variable that will be tested in the MMS. ex. 'cd'
		:type var: `char` or `str`

		:params expr: an expression used to express the analytical manufactured "solution"
		:type expr: `sympy expression / equation`

		:params spatial: a boolean switch to determine if the mms test is against dt
		:type spatial: `bool`

		:params temporal: a booolean switch to determine if the mms test is against delta x
		:type temporal: `bool`

		:params f_dict: this dictionary converts sympy functions into firedrake functions. Keys are strings representing sympy expressions and values are firedrake functions ex. {'exp' : fd.exp}
		:type f_dict: `dictionary`

		:params dt_list: a list of delta t values to be used in the MMS test
		:type dt_list: `list`

		:params meshes: a list of firedrake.Mesh objects
		:type meshes: `list`

		:params plot: a boolean switch to allow plotting of the convergence graphs
		:type plot: `bool`

		:params index: an integer signifiying which subspace of a MixedFunctionSpace that the variable exists in
		:type index: `int`
		"""

		# if users select error vs. dx
		if spatial:
			self.dx_array = []
			self.error = []
			self.dx_test(var, expr, meshes, f_dict, index)
			if plot:
				plot_error(self.dx_array, self.error, 'x')
		# if users select error vs. dt
		elif temporal:
			self.dt_array = []
			self.error = []
			self.dt_test(var, expr, dt_list, f_dict, index)
			if plot:
				plot_error(self.dt_array, self.error, 't')
		# no tests specified
		else:
			print("no convergence criteria specified")

	def dx_test(self, var, expr, meshes, f_dict, index):
		"""
		Description:

		This function calculates the l2 norm between the exact mms solution and
		the numerical solution against incremental values of dx.

		:params var: the variable that will be tested in the MMS. ex. 'cd'
		:type var: `char` or `str`

		:params expr: an expression used to express the analytical manufactured "solution"
		:type expr: `sympy expression / equation`

		:params f_dict: this dictionary converts sympy functions into firedrake functions. Keys are strings representing sympy expressions and values are firedrake functions ex. {'exp' : fd.exp}
		:type f_dict: `dictionary`

		:params meshes: a list of firedrake.Mesh objects
		:type meshes: `list`

		:params index: an integer signifiying which subspace of a MixedFunctionSpace that the variable exists in
		:type index: `int`
		"""
		# extract the 'str' or 'char' values of the symbol Symbols objects
		# create a sorted list of these values, as they represent keys for Firedrake
		# objects
		symbols = expr.free_symbols
		keys = list(map(str, list(symbols)))
		keys.sort()

		# create a function from the sympy expression
		function = sy.lambdify(list(symbols), expr, f_dict)

		# for each mesh specified in the list
		for mesh in meshes:
			# update the PDESystem's mesh value
			self.mesh = mesh
			# get the dx value of the mesh
			self.dx_array.append(get_dx(mesh))
			# obtain the x, y, and/or z coordinates of the mesh
			coordinate = fd.SpatialCoordinate(self.mesh)
			# create a list of these objects
			ini_args = list(coordinate) # create for initial
			fin_args = list(coordinate) # create for final
			an_args = list(coordinate) # create for analytical

			if 't' in keys:
				# add the time variables into the list
				ini_args.insert(0, fd.Constant(self.tstart))
				fin_args.insert(0, fd.Constant(self.prm['T']))
			# set initial time
			self.t = fd.Constant(self.tstart)
			an_args.insert(0, self.t)

			# create a dictionary to be passed into the lambdified sympy function
			ini_expr = dict(zip(keys, ini_args))
			fin_expr = dict(zip(keys, fin_args))
			analytical_expr = function(**dict(zip(keys, an_args)))

			# reinitialize the function spaces and functions for each different mesh
			self.setup_function_spaces(self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family'])
			self.setup_trial_test(self.prm['order'], self.prm['space'])
			self.setup_form_args(self.prm['order'], self.prm['space'])
			self.form_args.update(dict([('analytical', analytical_expr)]))
			self.setup_constants()
			# reinitialize the PDESubsystems with new mesh arguments
			for name in self.system_names:
				self.pdesubsystems[name].mesh = mesh
				self.pdesubsystems[name].get_form(self.form_args, self.constants)
			# interpolate the initial sympy function
			if index is None:
				self.form_args[var+'_n'].interpolate(function(**ini_expr))
			else:
				self.form_args[var+'_n'].split()[index].interpolate(function(**ini_expr))
			# set up boundary conditions
			self.setup_bcs()
			# solve and obtain error
			self.solve(time_update=True)
			if index is None:
				solution = fd.interpolate(function(**fin_expr), self.V[var])
				self.error.append(fd.errornorm(solution, self.form_args[var+'_n']))
			else:
				solution = fd.interpolate(function(**fin_expr), self.V[var][index])
				self.error.append(fd.errornorm(solution, self.form_args[var+'_n'].split()[index]))


	def dt_test(self, var, expr, dt_list, f_dict, index):
		"""
		Description:

		This function calculates the l2 norm between the exact mms solution and
		the numerical solution against incremental values of dt.

		:params var: the variable that will be tested in the MMS. ex. 'cd'
		:type var: `char` or `str`

		:params expr: an expression used to express the analytical manufactured "solution"
		:type expr: `sympy expression / equation`

		:params f_dict: this dictionary converts sympy functions into firedrake functions. Keys are strings representing sympy expressions and values are firedrake functions ex. {'exp' : fd.exp}
		:type f_dict: `dictionary`

		:params dt_list: a list of delta t values
		:type dt_list: `list`

		:params index: an integer signifiying which subspace of a MixedFunctionSpace that the variable exists in
		:type index: `int`
		"""
		# extract the 'str' or 'char' values of the symbol Symbols objects
		# create a sorted list of these values, as they represent keys for Firedrake
		# objects

		symbols = expr.free_symbols
		keys = list(map(str, list(symbols)))
		keys.sort()

		# create a function from the sympy expression
		function = sy.lambdify(list(symbols), expr, f_dict)

		# obtain the x, y, and/or z coordinates of the mesh
		coordinate = fd.SpatialCoordinate(self.mesh)
		# create a list of these objects
		ini_args = list(coordinate)
		fin_args = list(coordinate)
		an_args = list(coordinate)

		if 't' in keys:
			# add the time variables into the list
			ini_args.insert(0, fd.Constant(self.tstart))
			fin_args.insert(0, fd.Constant(self.prm['T']))

		# create a dictionary to be passed into the lambdified sympy function
		ini_expr = dict(zip(keys, ini_args))
		fin_expr = dict(zip(keys, fin_args))

		old_dt = self.dt
		# for each timestep value
		for deltat in dt_list:
			self.dt_array.append(deltat)
			self.dt = deltat
			# reinitialize the parameters with the new time step
			self.prm = recursive_update(self.prm, {'dt':deltat})
			self.t = fd.Constant(self.tstart)
			# reinitialize the function spaces and functions for each different mesh
			self.setup_function_spaces(self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family'])
			self.setup_trial_test(self.prm['order'], self.prm['space'])
			self.setup_form_args(self.prm['order'], self.prm['space'])

			an_args = list(coordinate)
			an_args.insert(0, self.t)
			analytical_expr = function(**dict(zip(keys, an_args)))

			self.form_args.update(dict([('analytical', analytical_expr)]))
			self.setup_constants()
			for name in self.system_names:
				self.pdesubsystems[name].get_form(self.form_args, self.constants)
			# interpolate the initial sympy function
			if index is None:
				self.form_args[var+'_n'].interpolate(function(**ini_expr))
			else:
				self.form_args[var+'_n'].split()[index].interpolate(function(**ini_expr))
			# set up boundary conditions
			self.setup_bcs()
			# solve and obtain error
			self.solve(time_update=True)
			if index is None:
				solution = fd.interpolate(function(**fin_expr), self.V[var])
				self.error.append(fd.errornorm(solution, self.form_args[var+'_n']))
			else:
				solution = fd.interpolate(function(**fin_expr), self.V[var][index])
				self.error.append(fd.errornorm(solution, self.form_args[var+'_n'].split()[index]))

	def view_args(self):
		"""
		This function prints all of the form_args keys, or names of the trial, test, and functions created in the current solver
		"""
		print(self.form_args.keys())

	def refresh(self):
		self.setup_function_spaces(self.mesh, self.prm['degree'], self.prm['space'], self.prm['order'], self.prm['family'])
		self.setup_trial_test(self.prm['order'], self.prm['space'])
		self.setup_form_args(self.prm['order'], self.prm['space'])

def get_dx(mesh):
	"""
	Description:
	This function returns the average delta x values of a firedrake.Mesh
	It makes a simplyfying assumption that finite elements are equilateral
	triangles.
	"""
	DG0 = fd.FunctionSpace(mesh, 'DG', 0)
	b = fd.Function(DG0).interpolate(fd.CellVolume(mesh))
	mean = b.dat.data.mean()

	return 2*np.sqrt(mean)

def plot_error(x, y, var):
	"""
	Description:
	A simple matplotlib wrapper to plot the logarithmic curves of error vs
	dt or dx. Fits a linear line to the data points to determine convergence rate.
	"""
	fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
	# plot error map
	ax1.loglog(x, y, 'o', label='fd_norm', color='k', markerfacecolor='None')

	ax1.set_xlabel('$\Delta %s$' %var, fontsize=14)
	ax1.set_ylabel('$l_{2}$  norm', fontsize=14)
	ax1.set_title('Error convergence graph vs. $\Delta %s$' %var, fontsize=14)
	ax1.legend(loc='best', fontsize=14)

	# line fit
	start_fit = 0
	line_fit = np.polyfit(np.log(x[start_fit:]), np.log(y[start_fit:]), 1)

	ax1.loglog(x, np.exp(line_fit[1]) * x**(line_fit[0]), 'k-', label = 'slope: {:.2f}'.format(line_fit[0]))
	ax1.legend(loc='best', fontsize=14)
