"""
author : Keer Mei
email: keer.mei18@imperial.ac.uk
github username: kmei92
"""

import firedrake as fd

class PDESubsystem:
	"""
	This class was adapted from Mikael Mortensen on July, 2019.
	Source code can be found here:
	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

	Description:
	The PDESubsystem represents a single set of stand alone Partial Differential equations.
	This set of PDEs should be able to be solved without coupling to any other set of PDEs.
	For example, the Chorin projection scheme of the navier stokes equation.

	:param solver_namespace: a dictionary of all vars from the PDESystem object.
	:type solver_namespace: `dictionary`

	:param var_sequence: a list of the variables to be solved. i.e. ['u', 'p', 'u'] for the navier stokes equation
	:type var_sequence: `list`
	"""
	def __init__(self, solver_namespace, var_sequence):
		self.solver_namespace = solver_namespace 	#PDEsystem's vars(self)
		self.vars = var_sequence 					#ex. ['u','p','u']
		self.mesh = solver_namespace['mesh']
		self.prm = solver_namespace['prm']
		self.constants = solver_namespace['constants']

		self.setup_base()

	def setup_base(self):
		"""
		Description:
		This function retrieves the form_args from the PDESystem and extracts
		the functions and trial functions from form)args
		"""
		self.query_args()
		self.get_form(self.form_args, self.constants)


	def query_args(self):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

		Description:
		Extracts the form_args that have been supplied by the system.

		:returns self.form_args: a dictionary of all functions and Constants created from the PDESystem
		:rtype self.form_args: `dictionary`
		"""
		form_args = self.solver_namespace['form_args']
		self.form_args = form_args

	def get_form(self, form_args, constants):
		"""
		This function was adapted from Mikael Mortensen on July, 2019.
		Source code can be found here:
		https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

		Description:
		For each variable in the var_seq, extract the variational form, the LHS
		and the RHS associated with the PDE responsible for solving that variable.

		:params form_args: a dictionary of trial, test, and functions
		:type form_args: `dictionary`

		:params constants: a dictionary of constants and firedrake parameters such as conditionals, coordinates, FacetNormals, etc...
		:type constants: `dictionary`
		"""
		# for variable in the sequence extract their forms
		# the sequence starts at 1 because forms should be labelled as
		# form1, form2, form3, etc....
		x = range(1, len(self.vars)+1)
		self.F = dict.fromkeys(x, None)
		self.a = dict.fromkeys(x, None)
		self.L = dict.fromkeys(x, None)
		i = 1
		for name in self.vars:
			if self.prm['order'][name] > 1:
				self.F[i] = eval('self.form%d(**form_args, **constants)'%(i))
				i += 1
			else:
				self.F[i] = eval('self.form%d(**form_args, **constants)'%(i))
				self.a[i], self.L[i] = fd.system(self.F[i])
				i += 1
