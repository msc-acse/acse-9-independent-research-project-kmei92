import sys
sys.path.append("..")
import PDESystem, PDESubsystem

def test_load_parameters():

    dict = {
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'linear_solver': {'u': 'gmres', 'p' :'gmres'},
	'subsystem_class' : {'up' : navier_stokes},
	'precond': {'u': 'sor', 'p': 'sor'},
	'dt' : 0.01,
	'T' : 10
	}

    solver_parameters = recursive_update(solver_parameters, dict)

    assert(bool(solver_parameters) is True)


