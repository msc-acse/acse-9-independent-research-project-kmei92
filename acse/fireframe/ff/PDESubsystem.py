import firedrake as fd

class PDESubsystem:
	"""
	This class was adapted from Mikael Mortensen on July, 2019.
	Source code can be found here:
	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

	Description:
	"""
	def __init__(self, solver_namespace, var_sequence, name, constants, bcs):
		self.solver_namespace = solver_namespace
		self.vars = var_sequence
		self.mesh = solver_namespace['mesh']
		self.prm = solver_namespace['prm']
		self.name = name
		self.constants = constants
		# print('subsystem_bcs', bcs)
		self.system_bcs = bcs
		# print('system_bcs', self.system_bcs)
		self.setup_base()

	def setup_base(self):
		self.query_args()
		self.get_form(self.form_args, self.constants)
		# self.setup_prob()
		# self.setup_solver()

	def query_args(self):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

		Extract the form_args that have been supplied by the system.
		Obtain the solution functions that are being solved for in the problems.
		Obtain the old solution functions that will be replaced after solving at each timestep.
		"""
		form_args = self.solver_namespace['form_args']
		self.form_args = form_args

		#boundary set up
		# self.bcs = {}
		# for var, boundaries in self.system_bcs.items():
		# 	self.bcs.update([(var, boundaries[0])])
		# print('bcs', self.bcs)


	def get_form(self, form_args, constants):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py


		"""
		x = range(1, len(self.vars)+1)
		self.F = dict.fromkeys(x, None)
		self.a = dict.fromkeys(x, None)
		self.L = dict.fromkeys(x, None)
		i = 1 # need this index to create corresponding forms
		for name in self.vars:
			if self.prm['order'][name] > 1:
				self.F[i] = eval('self.form%d(**form_args, **constants)'%(i))
				# print('self.form%d(**form_args)'%(i))
				i+=1
			else:
				self.F[i] = eval('self.form%d(**form_args, **constants)'%(i))
				self.a[i], self.L[i] = fd.system(self.F[i])
				i+=1

	def setup_prob(self):
		self.problems = [] # create a list of problems to be solved
		bounds = self.bcs # create a copy of the boundary conditions
		print(bounds)
		i = 1 # need this index to create corresponding problems
		for name in self.vars:
			print(name)
			if bounds[name]: # checks if there are boundary conditions associated with this variable
				if self.prm['order'][name] > 1: # for nonlinear problems
					# print(name+'_')
					self.problems.append(fd.NonlinearVariationalProblem(self.F[i], self.form_args[name+'_'], bcs=bounds[name][0]))
					i += 1
					# bounds[name].pop(0)
				else:
					# print('setting up linear problem')
					try:
						self.problems.append(fd.LinearVariationalProblem(self.a[i], self.L[i], self.form_args[name+'_'], bcs=bounds[name][0]))
						i += 1
					except:
						self.problems.append(fd.NonlinearVariationalProblem(self.F[i], self.form_args[name+'_'], bcs=bounds[name][0]))
					# bounds[name].pop(0)
			else:
				if self.prm['order'][name] > 1: # for nonlinear problems
					# print(name+'_')
					self.problems.append(fd.NonlinearVariationalProblem(self.F[i], self.form_args[name+'_']))
					i += 1
				else:
					# print('setting up linear problem')
					self.problems.append(fd.LinearVariationalProblem(self.a[i], self.L[i], self.form_args[name+'_']))
					i += 1

	def setup_solver(self):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py


		"""
		self.solvers = [] # create a list
		i = 0 # need this index to create corresponding
		for name in self.vars:
			# print(i)
			if self.prm['order'][name] > 1:
				self.solvers.append(fd.NonlinearVariationalSolver(self.problems[i]))
				i += 1
			elif name in self.prm['ksp_type'] and name not in self.prm['precond']: # if only iterative methods are specified
				self.solvers.append(fd.LinearVariationalSolver(self.problems[i], solver_parameters={'ksp_type': self.prm['linear_solver'][name]}))
				i += 1
			elif name not in self.prm['ksp_type'] and name in self.prm['precond']: # if only preconditioners are specified
				self.solvers.append(fd.LinearVariationalSolver(self.problems[i], solver_parameters={'pc_type': self.prm['precond'][name]}))
				i += 1
			else: # use default settings
				self.solvers.append(fd.LinearVariationalSolver(self.problems[i]))

# """
# This class is no longer used
# class PDESubsystem(PDESubsystembase):
#
# 	This class was adapted from Mikael Mortensen on July, 2019.
# 	Source code can be found here:
# 	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py
#
# 	Description:
# 	Base class for most PDESubSystems
#
# 	def __init__(self, solver_namespace, var_sequence, name, constants, bcs):
# 			PDESubsystembase.__init__(self, solver_namespace, var_sequence, name, constants, bcs)
# """
