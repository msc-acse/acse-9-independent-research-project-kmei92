import pytest
import sys
sys.path.append("..")
from PDESystem import *
from PDESubsystem import *
from demo.pdeforms import navier_stokes
import firedrake as fd

def test_load_parameters():

    diction = {
	'space': {'u': fd.VectorFunctionSpace},
	'degree': {'u': 2},
	'ksp_type': {'u': 'gmres', 'p' :'gmres'},
	'subsystem_class' : {'up' : navier_stokes},
	'precond': {'u': 'sor', 'p': 'sor'},
	'dt' : 0.01,
	'T' : 10
	}

    solver_parameters = recursive_update(default_solver_parameters, diction)

    assert(False is True)

if __name__ == '__main__':
    pytest.main()
