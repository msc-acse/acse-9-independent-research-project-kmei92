import firedrake as fd

class PDESubsystembase:
"""
This class was adapted from Mikael Mortensen on July, 2019.
Source code can be found here:
https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

Description:
"""
	def __init__(self, solver_namespace, var_sequence, name, bcs, forms_cnt):        
		self.solver_namespace = solver_namespace
		self.vars = var_sequence
		self.mesh = solver_namespace['mesh']
		self.prm = solver_namespace['prm']
		self.name = name
		self.bcs = bcs
		self.forms_cnt = forms_cnt

		self.setup_base()
        
	def setup_base(self):
		self.query_args()
		self.get_form(self.form_args)
		self.setup_prob()
		self.setup_solver()
        
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

	def get_form(self, form_args):
        """
	This function was adapted from Mikael Mortensen on July, 2019.
	Source code can be found here:
	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

	
        """
		x = range(1, self.forms_cnt+1)
		self.F = dict.fromkeys(x, None)
		self.a = dict.fromkeys(x, None)
		self.L = dict.fromkeys(x, None)
		
		if self.prm['order'][self.name] > 1:
		    for i in range(1, self.forms_cnt+1):
		        self.F[i] = eval('self.form%d(**form_args)'%(i))
		        print('self.form%d(**form_args)'%(i))
		else:
		    for i in range(1, self.forms_cnt+1):
		        self.F[i] = eval('self.form%d(**form_args)'%(i))
		        self.a[i], self.L[i] = fd.system(self.F[i])
        
	def setup_prob(self):
		self.problems = [] # create a list
		if self.prm['order'][self.name] > 1: # for nonlinear problems
		    for i in range(1, self.forms_cnt+1):
		        print(self.vars[i-1]+'_')
		        self.problems.append(fd.NonlinearVariationalProblem(self.F[i], self.form_args[self.vars[i-1]+'_'], 
		                                                            bcs=self.bcs[i-1]))
		else:
		    for i in range(1, self.forms_cnt+1):
		        self.problems.append(fd.LinearVariationalProblem(self.a[i], self.L[i], # solve for the fd.Function
		                                                 self.form_args[self.vars[i-1]+'_'],
		                                                 bcs=self.bcs[i-1]))
            
	def setup_solver(self):
        """
	This function was adapted from Mikael Mortensen on July, 2019.
	Source code can be found here:
	https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

	
        """
		self.solvers = [] # create a list
		if self.prm['order'][self.name] > 1:
		     for i in range(self.forms_cnt):
		        self.solvers.append(fd.NonlinearVariationalSolver(self.problems[i]))
		else:
		    for i in range(self.forms_cnt):
		        self.solvers.append(fd.LinearVariationalSolver(self.problems[i], 
		                                            solver_parameters=
		                                                   {'ksp_type': self.prm['linear_solver'][self.name],
		                                                               'pc_type': self.prm['precond'][self.name]}))

class PDESubsystem(PDESubsystembase):
"""

This class was adapted from Mikael Mortensen on July, 2019.
Source code can be found here:
https://bitbucket.org/simula_cbc/cbcpdesys/src/master/cbc/pdesys/PDESubSystems.py

Description:
Base class for most PDESubSystems
"""
	def __init__(self, solver_namespace, name, sub_system, bcs, forms_cnt):
        	PDESubsystembase.__init__(self, solver_namespace, name, sub_system, bcs, forms_cnt)
        

