import sys
sys.path.append("..")
from PDESystem import *
from PDESubsystem import *
from pdeforms import *

class pde_solver(PDESystem):
    def __init__(self, comp, mesh, parameters):
        PDESystem.__init__(self, comp, mesh, parameters)

    def setup_bcs(self):
        x, y = fd.SpatialCoordinate(self.mesh)

        bcu = [fd.DirichletBC(self.V['u'], fd.Constant((0,0)), (10, 12)), # top-bottom and cylinder
          fd.DirichletBC(self.V['u'], ((1.0*(y - 1)*(2 - y))/(0.5**2) ,0), 9)] # inflow
        bcp = [fd.DirichletBC(self.V['p'], fd.Constant(0), 11)]  # outflow

        self.bc['u'][0] = [bcu, None, None, None,'fixed']
        self.bc['p'] = [[bcp, None, None, None, 'fixed']]

    def setup_constants(self):
        self.constants = {
            'k' : fd.Constant(self.prm['dt']),
            'Kd' : fd.Constant(0.01),
            'k1' : fd.Constant(0.5),
            'k2' : fd.Constant(0.01),
            'lamd1' : fd.Constant(1.5),
            'lamd2' : fd.Constant(0.),
            'rho_s' : fd.Constant(1.),
            'L' :  fd.Constant(1.),
            'phi' : fd.Constant(0.3),
            'f' : fd.Constant(1.),
            'n' : fd.FacetNormal(self.mesh),
            'f' : fd.Constant((0.0, 0.0)),
            'nu' : fd.Constant(0.001),
            'frac' : fd.Constant(1.)
        }

if __name__ == '__main__':
    solver_parameters = recursive_update(solver_parameters,
    {
    'space': {'u': fd.VectorFunctionSpace, 'cs': fd.MixedFunctionSpace, 'cd' : fd.MixedFunctionSpace, 'as' : fd.MixedFunctionSpace},
    'degree': {'u': 2, 'p': 1, 'cs': 1, 'cd' : 1, 'as' : 1},
    'order' : {'u': 1, 'p': 1, 'cs' : 2, 'cd' : 2, 'as' : 2, 'cdcsas' : 2},
    'ksp_type': {'u': 'gmres', 'p': 'gmres', 'cs': 'gmres', 'cd': 'gmres', 'as': 'gmres'},
    'subsystem_class' : {'up': navier_stokes, 'cdcsas' : radio_transport},
    'precond': {'u': 'sor', 'p' : 'sor', 'cs': 'sor', 'cd': 'sor', 'as': 'sor'},
    'T': 1.0
    }
    )

    #define mesh
    mesh = fd.Mesh("../../meshes/step3.msh")

    # add subsystems
    solver = pde_solver([['u', 'p']], mesh, solver_parameters)
    solver.add_subsystem(['cd', 'cs', 'as'], solver_parameters)

    #setup system and define subsystems
    solver.setup_constants()
    solver.define(['u', 'p', 'u'], 'up')
    solver.define(['cd', 'cs', 'as'], 'cdcsas')
    solver.setup_bcs()


    #setup initial condition
    x, y = fd.SpatialCoordinate(mesh)
    c = fd.conditional(pow(x-1, 2)+pow(y-1.5,2)<0.05*0.05, 10, 0)
    solver.setup_initial('cd_n', c, mixedspace=True, index=0)

    #solve
    solver.solve()

    # #plot
    # fig1 = plt.figure(figsize=(16, 2.5))
    # ax1 = fig1.add_subplot(111)
    # ax1.set_xlabel('$x$', fontsize=16)
    # ax1.set_ylabel('$y$', fontsize=16)
    # ax1.set_title('FEM Navier-Stokes - channel flow - pressure', fontsize=16)
    # fd.plot(solver.form_args['p_'],axes=ax1)
    # ax1.axis('equal')
    # plt.pause(6)
    #
    # fig2 = plt.figure(figsize=(16, 2.5))
    # ax2 = fig2.add_subplot(111)
    # ax2.set_xlabel('$x$', fontsize=16)
    # ax2.set_ylabel('$y$', fontsize=16)
    # ax2.set_title('FEM Navier-Stokes - channel flow - velocity', fontsize=16)
    # fd.plot(solver.form_args['u_'],axes=ax2)
    # ax2.axis('equal');
    # plt.pause(6)
    #
    # cd1, cd2 = solver.form_args['cd_'].split()
    # cs1, cs2 = solver.form_args['cs_'].split()
    # as1, as2 = solver.form_args['as_'].split()
    #
    # fig3 = plt.figure(figsize=(16, 2.5))
    # ax3 = fig3.add_subplot(111)
    # ax3.set_xlabel('$x$', fontsize=16)
    # ax3.set_ylabel('$y$', fontsize=16)
    # ax3.set_title('FEM Navier-Stokes - channel flow - dissolved1', fontsize=16)
    # fd.plot(cd1,axes=ax3)
    # ax3.axis('equal');
    # plt.pause(6)
    #
    # fig4 = plt.figure(figsize=(16, 2.5))
    # ax4 = fig4.add_subplot(111)
    # ax4.set_xlabel('$x$', fontsize=16)
    # ax4.set_ylabel('$y$', fontsize=16)
    # ax4.set_title('FEM Navier-Stokes - channel flow - suspended1', fontsize=16)
    # fd.plot(cs1, axes=ax4)
    # ax4.axis('equal');
    # plt.pause(6)
    #
    # fig5 = plt.figure(figsize=(16, 2.5))
    # ax5 = fig5.add_subplot(111)
    # ax5.set_xlabel('$x$', fontsize=16)
    # ax5.set_ylabel('$y$', fontsize=16)
    # ax5.set_title('FEM Navier-Stokes - channel flow - sediment1', fontsize=16)
    # fd.plot(as1, axes=ax5)
    # ax5.axis('equal');
    # plt.pause(6)
    #
    # fig6 = plt.figure(figsize=(16, 2.5))
    # ax6 = fig6.add_subplot(111)
    # ax6.set_xlabel('$x$', fontsize=16)
    # ax6.set_ylabel('$y$', fontsize=16)
    # ax6.set_title('FEM Navier-Stokes - channel flow - dissolved2', fontsize=16)
    # fd.plot(cd2, axes=ax6)
    # ax6.axis('equal');
    # plt.pause(6)
    #
    # fig7 = plt.figure(figsize=(16, 2.5))
    # ax7 = fig7.add_subplot(111)
    # ax7.set_xlabel('$x$', fontsize=16)
    # ax7.set_ylabel('$y$', fontsize=16)
    # ax7.set_title('FEM Navier-Stokes - channel flow - suspended2', fontsize=16)
    # fd.plot(cs2, axes=ax7)
    # ax7.axis('equal');
    # plt.pause(6)
    #
    # fig8 = plt.figure(figsize=(16, 2.5))
    # ax8 = fig8.add_subplot(111)
    # ax8.set_xlabel('$x$', fontsize=16)
    # ax8.set_ylabel('$y$', fontsize=16)
    # ax8.set_title('FEM Navier-Stokes - channel flow - sediment2', fontsize=16)
    # fd.plot(as2, axes=ax8)
    # ax8.axis('equal');
    # plt.pause(6)
