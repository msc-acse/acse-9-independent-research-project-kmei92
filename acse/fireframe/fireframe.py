__author__ = "Keer Mei <keer.mei18@imperial.ac.uk>"
__date__ "2019-__-__"
__copyright__ = ""
__license__ = ""

from collections import defaultdict
import copy
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

default_solver_parameters = {
    'degree': defaultdict(lambda: 1), # default function space dimension
    'family': defaultdict(lambda: 'CG'), # default trial/test functions
    'space' : defaultdict(lambda: fd.FunctionSpace), # default functionspace
    'pdesubsystem': defaultdict(lambda: 1),
    'linear_solver': defaultdict(lambda: 'gmres'),
    'T': 1., # End time for simulation,
    'dt': 0.001  # timestep,
}

def recursive_update(dst, src):
    """Update dict dst with items from src deeply ("deep update")."""
    for key, val in src.items():
        if key in dst and isinstance(val, dict) and isinstance(dst[key], dict):
            dst[key] = recursive_update(dst[key], val)
        else:
            dst[key] = val
    return dst

class PDESystem:
    """
    A base class for solving a system of equations
    """
    def __init__(self, system_composition, mesh, parameters):      # removed problem(mesh)
        self.system_composition = system_composition         # Total system comp.
        self.system_names = []                               # Compounds solved for
        self.names = []                                      # All components
        self.mesh = mesh
        self.prm = parameters
        
        
        self.t0 = 0                                           # Simulation time
        self.tstep = 0                                       # Time step
        self.total_number_iters = 0                          #
        self.num_timesteps = 0                               #
                
        for sub_system in system_composition: 
            system_name = ''.join(sub_system) 
            self.system_names.append(system_name)
            for n in sub_system:       # Run over all individual components
                self.names.append(n)
                
        self.setup_system()
        
    def setup_system(self):
        self.define_function_spaces(self.mesh, self.prm['degree'], # removed self.mesh
                              self.prm['space'], self.prm['family']) # removed cons
        self.setup_subsystems()
        self.pdesubsystems = dict((name, None) for name in self.system_names)
        self.normalize     = dict((name, None) for name in self.system_names)
        self.bc            = dict((name,   []) for name in self.system_names)
        self.setup_form_args()
        
    def define_function_spaces(self, mesh, degree, space, family):
        """Define functionspaces for names and system_names"""
        V = self.V = dict((name, space[name](mesh, family[name], degree[name])) for name in self.names)

    def setup_subsystems(self):
        V, sys_names, sys_comp = \
               self.V, self.system_names, self.system_composition

        q = dict((name, None) for name in sys_names)
        v = dict((name, None) for name in sys_names)
        
        # Create compound functions for the various sub systems
        for sub_sys in sys_comp:
            if len(sub_sys) > 1:
                for name in sub_sys:
                    q.update(dict([(name+'_trl', fd.TrialFunction(V[name]))]))
                    v.update(dict([(name+'_tst', fd.TestFunction(V[name]))]))

        self.qt, self.vt = q, v
    
    def setup_form_args(self):
        V, sys_comp = self.V, self.system_composition
        form_args = {}
        for sub_sys in sys_comp:
            if len(sub_sys) > 1:
                for name in sub_sys:
                    form_args.update(dict( [ (name + '_', fd.Function(V[name]) ) ] ) ) # current
                    form_args.update(dict( [ (name + '_n', fd.Function(V[name]) ) ] ) )
        
        form_args.update(self.qt)
        form_args.update(self.vt)
            
        self.form_args = form_args
        
    def solve(self):
        ''' call on the pdesubsystems to solve for their respective variables'''
        for name in self.system_names:
            self.pdesubsystems[name].solve_subsystem() 
            
class Subdict(dict):
    """Dictionary that looks for missing keys in the solver_namespace
       Currently not in use.
    """
    def __init__(self, solver_namespace, sub_name, **kwargs):
        dict.__init__(self, **kwargs)
        self.solver_namespace = solver_namespace
        self.sub_name = sub_name
    
    def __missing__(self, key):
        try:
            self[key] = self.solver_namespace['prm'][key][self.sub_name]
        except:
            self[key] = self.solver_namespace['prm'][key]
        return self[key]

class PDESubsystembase:
    def __init__(self, solver_namespace, sub_system, bcs, forms_cnt):        
        self.solver_namespace = solver_namespace
        self.vars = sub_system
        self.mesh = solver_namespace['mesh']
        self.prm = solver_namespace['prm']
        self.bcs = bcs
        self.forms_cnt = forms_cnt
        
        self.setup_base()
        
    def setup_base(self):
        self.query_args()
        self.get_form(self.form_args)
        self.setup_prob()
        self.setup_solver()
        
    def query_args(self):
        """Extract the form_args that have been supplied by the system.
           Obtain the solution functions that are being solved for in the problems.
           Obtain the old solution functions that will be replaced after solving at each timestep.
        """
        form_args = self.solver_namespace['form_args']
        self.form_args = form_args

    def get_form(self, form_args):
        x = range(1, self.forms_cnt+1)
        self.F = dict.fromkeys(x, None)
        self.a = dict.fromkeys(x, None)
        self.L = dict.fromkeys(x, None)
        
        for i in range(1, self.forms_cnt+1):
            self.F[i] = eval('self.form%d(**form_args)'%(i))
            self.a[i], self.L[i] = fd.system(self.F[i])
        
    def setup_prob(self):
        self.problems = [] # create a list
        for i in range(1, self.forms_cnt+1):
            self.problems.append(fd.LinearVariationalProblem(self.a[i], self.L[i], # solve for the fd.Function
                                                         self.form_args[self.vars[i-1]+'_'],
                                                         bcs=self.bcs[i-1]))
            
    def setup_solver(self):
        self.solvers = [] # create a list
        for i in range(self.forms_cnt):
            self.solvers.append(fd.LinearVariationalSolver(self.problems[i], 
                                                    solver_parameters=
                                                           {'ksp_type': self.prm['linear_solver']['up'],
                                                                       'pc_type': self.prm['precond']['up']}))

class PDESubsystem(PDESubsystembase):
    """Base class for most PDESubSystems"""
    def __init__(self, solver_namespace, sub_system, bcs, forms_cnt):
        PDESubsystembase.__init__(self, solver_namespace, sub_system, bcs, forms_cnt)
        

    def solve_subsystem(self):
        t0 = self.solver_namespace['t0']
        dt = self.prm['dt']
        T = self.prm['T']
        
        while t0 < T:
            for i in range(self.forms_cnt):
                self.solvers[i].solve()
            for i in range(self.forms_cnt):
                self.form_args[self.vars[i]+'_n'].assign(self.form_args[self.vars[i]+'_'])
            
            t0 += dt
            if( np.abs( t0 - np.round(t0,decimals=0) ) < 1.e-8): 
                print('time = {0:.3f}'.format(t0))

