"""
author : Keer Mei
email: keer.mei18@imperial.ac.uk
github username: kmei92
"""

from .PDESubsystem import *

"""
This module contains all the variational forms for the demontration problems
in Fireframe. They are all written in Firedrake syntax.

navier_stokes: contains the three forms of Chorin's scheme

reactions: contains the nonlinear form of 3 chemicals first order reaction

reactions_uncoupled: contains 3 linear forms of 3 chemicals first order reaction

radio_transport_coupled: contains the nonlinear form of 2 radionuclide 3 phase
transfer transport equations coupled to the navier stokes equation

radio_transport_coupled_mms: contains the nonlinear form of the
radionuclide transport equations with a manufcatured solution to the dissolved
phase radionuclide 1 specific activity. Manufactured solution is obtained by hand
differentiation of manufactured solution.

radio_transport_coupled_mms_ufl: contains the nonlinear form of the
radionuclide transport equations with a manufcatured solution to the dissolved
phase radionuclide 1 specific activity. Manufactured solution is obtained by UFL
expressions of the manufactured solution

radio_transport_hydro: contains the nonlinear form of 2 radionuclide 3 phase
transfer transport equations coupled to the shallow water equation

temp_poisson: contains the nonlinear form of the time varying temperature diffusion equation.

depth_avg_hydro: contains the nonlinear form of the
no normal boundary condition shallow water equation

depth_avg_hydro_flather: contains the nonlinear form of the
flather boundary condition shallow water equation
"""
class navier_stokes(PDESubsystem):

	def form1(self, u_trl, u_tst, u_n, p_n, n, deltat, f, nu, **kwargs):
		"""
		Solving for intermediate velocity
		"""
		def sigma(u, p):
			return 2*nu*fd.sym(fd.nabla_grad(u)) - p*fd.Identity(len(u))

		u_mid = 0.5 * (u_n + u_trl)

		Form = fd.inner((u_trl - u_n)/deltat, u_tst) * fd.dx \
		+ fd.inner(fd.dot(u_n, fd.nabla_grad(u_mid)), u_tst) * fd.dx \
		+ fd.inner(sigma(u_mid, p_n), fd.sym(fd.nabla_grad(u_tst))) * fd.dx \
		+ fd.inner(p_n * n, u_tst) * fd.ds \
		- fd.inner(nu * fd.dot(fd.nabla_grad(u_mid), n), u_tst) * fd.ds \
		- fd.inner(f, u_tst) * fd.dx

		return Form

	def form2(self, p_trl, p_tst, p_n, u_, deltat, **kwargs):
		"""
		Solving for pressure
		"""
		Form = fd.inner(fd.nabla_grad(p_trl), fd.nabla_grad(p_tst)) * fd.dx \
		- fd.inner(fd.nabla_grad(p_n), fd.nabla_grad(p_tst)) * fd.dx \
		+ (1/deltat) * fd.inner(fd.div(u_), p_tst) * fd.dx

		return Form

	def form3(self, u_trl, u_tst, u_, p_n, p_, deltat, **kwargs):
		"""
		Updating velocity for divergency free condition
		"""
		Form = fd.inner(u_trl, u_tst) * fd.dx \
		- fd.inner(u_, u_tst) * fd.dx \
		+ deltat * fd.inner(fd.nabla_grad(p_ - p_n), u_tst) * fd.dx

		return Form

class reactions(PDESubsystem):

	def form1(self, c_1, c_n1, c_tst1, c_2, c_n2, c_tst2, c_3, c_n3,
				c_tst3, u_, eps, K, deltat, f_1, f_2, f_3, **kwargs):
		"""
		3 componenet first order chemical reactions system in nonlinear form
		"""
		Form = ((c_1 - c_n1) / deltat)*c_tst1*fd.dx \
		+ ((c_2 - c_n2) / deltat)*c_tst2*fd.dx \
		+ ((c_3 - c_n3) / deltat)*c_tst3*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_1)), c_tst1)*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_2)), c_tst2)*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_3)), c_tst3)*fd.dx \
		+ eps*fd.dot(fd.grad(c_1), fd.grad(c_tst1))*fd.dx \
		+ eps*fd.dot(fd.grad(c_2), fd.grad(c_tst2))*fd.dx \
		+ eps*fd.dot(fd.grad(c_3), fd.grad(c_tst3))*fd.dx \
		+ K*c_1*c_2*c_tst1*fd.dx  \
		+ K*c_1*c_2*c_tst2*fd.dx  \
		- K*c_1*c_2*c_tst3*fd.dx \
		+ K*c_3*c_tst3*fd.dx \
		- f_1*c_tst1*fd.dx \
		- f_2*c_tst2*fd.dx \
		- f_3*c_tst3*fd.dx

		return Form

class reactions_uncoupled(PDESubsystem):

	def form1(self, c1_trl, c1_n, c1_tst, c2_n, u_, eps, K, deltat, f_1, **kwargs):
		"""
		linear form of first chemical reactant mass balance
		"""
		Form = ((c1_trl - c1_n) / deltat)*c1_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c1_trl)), c1_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c1_trl), fd.grad(c1_tst))*fd.dx \
		+K*c1_trl*c2_n*c1_tst*fd.dx \
		-f_1*c1_tst*fd.dx

		return Form

	def form2(self, c2_trl, c2_n, c2_tst, c1_n, u_, eps, K, deltat, f_2, **kwargs):
		"""
		linear form of second chemical reactant mass balance
		"""
		Form = ((c2_trl - c2_n) / deltat)*c2_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c2_trl)), c2_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c2_trl), fd.grad(c2_tst))*fd.dx \
		+K*c2_trl*c1_n*c2_tst*fd.dx \
		-f_2*c2_tst*fd.dx

		return Form

	def form3(self, c3_trl, c3_n, c3_tst, c1_n, c2_n, u_, eps, K, deltat, f_3, **kwargs):
		"""
		linear form of third chemical product mass balance
		"""
		Form = ((c3_trl - c3_n) / deltat)*c3_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c3_trl)), c3_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c3_trl), fd.grad(c3_tst))*fd.dx \
		- K*c1_n*c2_n*c3_tst*fd.dx \
		+ K*c3_trl*c3_tst*fd.dx \
		-f_3*c3_tst*fd.dx

		return Form

class radio_transport_coupled(PDESubsystem):

	def form1(self, c_1, c_n1, c_2, c_n2, c_3, c_n3, c_tst1, c_tst2, c_tst3, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, source1, **kwargs):
		"""
		nonlinear system of equations for first radionuclide with half life lamd1.
		Includes 3 phases: dissolved, suspended, and sediment
		"""
		# dissolved phase
		Form = ((c_1 - c_n1) / deltat)*c_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_1)), c_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_1), fd.grad(c_tst1))*fd.dx \
		+ k1*c_1*c_tst1*fd.dx \
		- k2*c_2*c_tst1*fd.dx \
		- k2*c_3*L*rho_s*frac*phi*c_tst1*fd.dx \
		+ lamd1*c_1*c_tst1*fd.dx \
		- source1*c_tst1*fd.dx

		# suspended phase
		Form += ((c_2 - c_n2) / deltat)*c_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_2)), c_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_2), fd.grad(c_tst2))*fd.dx \
		- k1*c_1*c_tst2*fd.dx \
		+ k2*c_2*c_tst2*fd.dx \
		+ lamd1*c_2*c_tst2*fd.dx

		# sediment phase
		Form += ((c_3 - c_n3) / deltat)*c_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*c_1*c_tst3*fd.dx \
		+ k2*c_1*phi*c_tst3*fd.dx \
		+ lamd1*c_3*c_tst3*fd.dx

		return Form

	def form2(self, d_1, d_n1, d_2, d_n2, d_3, d_n3, d_tst1, d_tst2, d_tst3, c_n1, c_n2,
				u_, c_n3, deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, source1, **kwargs):
		"""
		nonlinear system of equations for second radionuclide with half life lamd2.
		Includes 3 phases: dissolved, suspended, and sediment
		"""
		# dissolved phase
		Form = ((d_1 - d_n1) / deltat)*d_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(d_1)), d_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_1), fd.grad(d_tst1))*fd.dx \
		+ k1*d_1*d_tst1*fd.dx \
		- k2*d_2*d_tst1*fd.dx \
		- k2*d_3*L*rho_s*frac*phi*d_tst1*fd.dx \
		- lamd1*c_n1*d_tst1*fd.dx \
		+ lamd2*d_1*d_tst1*fd.dx

		# suspended phase
		Form += ((d_2 - d_n2) / deltat)*d_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(d_2)), d_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_2), fd.grad(d_tst2))*fd.dx \
		- k1*d_1*d_tst2*fd.dx \
		+ k2*d_2*d_tst2*fd.dx \
		- lamd1*c_n2*d_tst2*fd.dx \
		+ lamd2*d_2*d_tst2*fd.dx

		# sediment phase
		Form += ((d_3 - d_n3) / deltat)*d_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*d_1*d_tst3*fd.dx \
		+ k2*d_1*phi*d_tst3*fd.dx \
		- lamd1*c_n3*d_tst3*fd.dx \
		+ lamd2*d_3*d_tst3*fd.dx

		return Form

class radio_transport_coupled_mms(PDESubsystem):

	def form1(self, c_1, c_n1, c_2, c_n2, c_3, c_n3, c_tst1, c_tst2, c_tst3, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, t, **kwargs):

		"""
		manufactured equation for nonlinear system of equations for first radionuclide with half life lamd1.
		Includes 3 phases: dissolved, suspended, and sediment. Includes an extra source term.
		"""

		# manufactured source term
		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		dc_dt = x*y*c_source
		grad_c = fd.as_vector((y*t*c_source, x*t*c_source))
		laplace_c = (y*t)**2 * c_source + (x*t)**2*c_source
		source1 = dc_dt + fd.dot(u_, grad_c) - Kd*laplace_c + k1*c_source + lamd1*c_source
		source2 = -k1*c_source
		source3 = -(1./(L * rho_s * frac)) * k1 * c_source + k2*phi*c_source

		# dissolved phase
		Form = ((c_1 - c_n1) / deltat)*c_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_1)), c_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_1), fd.grad(c_tst1))*fd.dx \
		+ k1*c_1*c_tst1*fd.dx \
		- k2*c_2*c_tst1*fd.dx \
		- k2*c_3*L*rho_s*frac*phi*c_tst1*fd.dx \
		+ lamd1*c_1*c_tst1*fd.dx \
		- source1*c_tst1*fd.dx

		# suspended phase
		Form += ((c_2 - c_n2) / deltat)*c_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_2)), c_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_2), fd.grad(c_tst2))*fd.dx \
		- k1*c_1*c_tst2*fd.dx \
		+ k2*c_2*c_tst2*fd.dx \
		+ lamd1*c_2*c_tst2*fd.dx \
		- source2*c_tst2*fd.dx

		# sediment phase
		Form += ((c_3 - c_n3) / deltat)*c_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*c_1*c_tst3*fd.dx \
		+ k2*c_1*phi*c_tst3*fd.dx \
		+ lamd1*c_3*c_tst3*fd.dx \
		- source3*c_tst3*fd.dx

		return Form

	def form2(self, d_1, d_n1, d_2, d_n2, d_3, d_n3, d_tst1, d_tst2, d_tst3, c_n1, c_n2,
				u_, c_n3, deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, t, **kwargs):
		"""
		manufactured equation for nonlinear system of equations for second radionuclide with half life lamd2.
		Includes 3 phases: dissolved, suspended, and sediment. No source term.
		"""
		# manufactured source term
		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		source1 = -lamd1*c_source

		# dissolved phase
		Form = ((d_1 - d_n1) / deltat)*d_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(d_1)), d_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_1), fd.grad(d_tst1))*fd.dx \
		+ k1*d_1*d_tst1*fd.dx \
		- k2*d_2*d_tst1*fd.dx \
		- k2*d_3*L*rho_s*frac*phi*d_tst1*fd.dx \
		- lamd1*c_n1*d_tst1*fd.dx \
		+ lamd2*d_1*d_tst1*fd.dx \
		- source1*d_tst1*fd.dx

		# suspended phase
		Form += ((d_2 - d_n2) / deltat)*d_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(d_2)), d_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_2), fd.grad(d_tst2))*fd.dx \
		- k1*d_1*d_tst2*fd.dx \
		+ k2*d_2*d_tst2*fd.dx \
		- lamd1*c_n2*d_tst2*fd.dx \
		+ lamd2*d_2*d_tst2*fd.dx

		# sediment phase
		Form += ((d_3 - d_n3) / deltat)*d_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*d_1*d_tst3*fd.dx \
		+ k2*d_1*phi*d_tst3*fd.dx \
		- lamd1*c_n3*d_tst3*fd.dx \
		+ lamd2*d_3*d_tst3*fd.dx

		return Form

class radio_transport_coupled_mms_ufl(PDESubsystem):

	def form1(self, c_1, c_n1, c_2, c_n2, c_3, c_n3, c_tst1, c_tst2, c_tst3, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, t, **kwargs):

		"""
		manufactured equation for nonlinear system of equations for first radionuclide with half life lamd1.
		Includes 3 phases: dissolved, suspended, and sediment. Includes an extra source term.
		"""

		# dissolved phase
		Form = ((c_1 - c_n1) / deltat)*c_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_1)), c_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_1), fd.grad(c_tst1))*fd.dx \
		+ k1*c_1*c_tst1*fd.dx \
		- k2*c_2*c_tst1*fd.dx \
		- k2*c_3*L*rho_s*frac*phi*c_tst1*fd.dx \
		+ lamd1*c_1*c_tst1*fd.dx \
		- source1*c_tst1*fd.dx

		# suspended phase
		Form += ((c_2 - c_n2) / deltat)*c_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c_2)), c_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_2), fd.grad(c_tst2))*fd.dx \
		- k1*c_1*c_tst2*fd.dx \
		+ k2*c_2*c_tst2*fd.dx \
		+ lamd1*c_2*c_tst2*fd.dx \
		- source2*c_tst2*fd.dx

		# sediment phase
		Form += ((c_3 - c_n3) / deltat)*c_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*c_1*c_tst3*fd.dx \
		+ k2*c_1*phi*c_tst3*fd.dx \
		+ lamd1*c_3*c_tst3*fd.dx \
		- source3*c_tst3*fd.dx

		if 'analytical' in kwargs:
			dc_dt = fd.diff(kwargs['analytical'], t)
			grad_c = fd.grad(kwargs['analytical'])
			laplace = fd.div(fd.grad(kwargs['analytical']))
			source1 = dc_dt + fd.dot(u_, grad_c) - Kd*kwargs['analytical'] + k1*kwargs['analytical'] + lamd1*kwargs['analytical']
			source2 = -k1 * kwargs['analytical']
			source3 = - (1./(L * rho_s * frac))*k1*kwargs['analytical'] + k2*kwargs['analytical']*phi
			Form += -source1*c_tst1*fd.dx\
			-source2*c_tst2*fd.dx\
			-source3*c_tst3*fd.dx


		return Form

	def form2(self, d_1, d_n1, d_2, d_n2, d_3, d_n3, d_tst1, d_tst2, d_tst3, c_n1, c_n2,
				u_, c_n3, deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, t, **kwargs):
		"""
		manufactured equation for nonlinear system of equations for second radionuclide with half life lamd2.
		Includes 3 phases: dissolved, suspended, and sediment. No source term.
		"""

		# dissolved phase
		Form = ((d_1 - d_n1) / deltat)*d_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(d_1)), d_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_1), fd.grad(d_tst1))*fd.dx \
		+ k1*d_1*d_tst1*fd.dx \
		- k2*d_2*d_tst1*fd.dx \
		- k2*d_3*L*rho_s*frac*phi*d_tst1*fd.dx \
		- lamd1*c_n1*d_tst1*fd.dx \
		+ lamd2*d_1*d_tst1*fd.dx \

		# suspended phase
		Form += ((d_2 - d_n2) / deltat)*d_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(d_2)), d_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_2), fd.grad(d_tst2))*fd.dx \
		- k1*d_1*d_tst2*fd.dx \
		+ k2*d_2*d_tst2*fd.dx \
		- lamd1*c_n2*d_tst2*fd.dx \
		+ lamd2*d_2*d_tst2*fd.dx

		# sediment phase
		Form += ((d_3 - d_n3) / deltat)*d_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*d_1*d_tst3*fd.dx \
		+ k2*d_1*phi*d_tst3*fd.dx \
		- lamd1*c_n3*d_tst3*fd.dx \
		+ lamd2*d_3*d_tst3*fd.dx

		if 'analytical' in kwargs:
			source1 = -lamd1*kwargs['analytical']
			Form += -source1*d_tst1*fd.dx

		return Form

class radio_transport_hydro(PDESubsystem):

	def form1(self, c_1, c_n1, c_2, c_n2, c_3, c_n3, c_tst1, c_tst2, c_tst3, z_1,
			  deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, source1, **kwargs):
		"""
		nonlinear system of equations for first radionuclide with half life lamd1.
		Includes 3 phases: dissolved, suspended, and sediment. The velocity term
		is changed from u_ to z_1 as it uses the mixedfunctionspace velocity
		from the shallow water equation. Everything else is kept same.
		"""
		# dissolved phase
		Form = ((c_1 - c_n1) / deltat)*c_tst1*fd.dx \
		+ fd.inner(fd.dot(z_1, fd.nabla_grad(c_1)), c_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_1), fd.grad(c_tst1))*fd.dx \
		+ k1*c_1*c_tst1*fd.dx \
		- k2*c_2*c_tst1*fd.dx \
		- k2*c_3*L*rho_s*frac*phi*c_tst1*fd.dx \
		+ lamd1*c_1*c_tst1*fd.dx \
		- source1*c_tst1*fd.dx

		# suspended phase
		Form += ((c_2 - c_n2) / deltat)*c_tst2*fd.dx \
		+ fd.inner(fd.dot(z_1, fd.nabla_grad(c_2)), c_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(c_2), fd.grad(c_tst2))*fd.dx \
		- k1*c_1*c_tst2*fd.dx \
		+ k2*c_2*c_tst2*fd.dx \
		+ lamd1*c_2*c_tst2*fd.dx

		# sediment phase
		Form += ((c_3 - c_n3) / deltat)*c_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*c_1*c_tst3*fd.dx \
		+ k2*c_1*phi*c_tst3*fd.dx \
		+ lamd1*c_3*c_tst3*fd.dx

		return Form

	def form2(self, d_1, d_n1, d_2, d_n2, d_3, d_n3, d_tst1, d_tst2, d_tst3, c_n1, c_n2,
				z_1, c_n3, deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, source1, **kwargs):
		"""
		nonlinear system of equations for second radionuclide with half life lamd2.
		Includes 3 phases: dissolved, suspended, and sediment. The velocity term
		is changed from u_ to z_1 as it uses the mixedfunctionspace velocity
		from the shallow water equation. Everything else is kept same.
		"""
		# dissolved phase
		Form = ((d_1 - d_n1) / deltat)*d_tst1*fd.dx \
		+ fd.inner(fd.dot(z_1, fd.nabla_grad(d_1)), d_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_1), fd.grad(d_tst1))*fd.dx \
		+ k1*d_1*d_tst1*fd.dx \
		- k2*d_2*d_tst1*fd.dx \
		- k2*d_3*L*rho_s*frac*phi*d_tst1*fd.dx \
		- lamd1*c_n1*d_tst1*fd.dx \
		+ lamd2*d_1*d_tst1*fd.dx

		# suspended phase
		Form += ((d_2 - d_n2) / deltat)*d_tst2*fd.dx \
		+ fd.inner(fd.dot(z_1, fd.nabla_grad(d_2)), d_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(d_2), fd.grad(d_tst2))*fd.dx \
		- k1*d_1*d_tst2*fd.dx \
		+ k2*d_2*d_tst2*fd.dx \
		- lamd1*c_n2*d_tst2*fd.dx \
		+ lamd2*d_2*d_tst2*fd.dx

		# sediment phase
		Form += ((d_3 - d_n3) / deltat)*d_tst3*fd.dx \
		- (1./(L * rho_s * frac)) * k1*d_1*d_tst3*fd.dx \
		+ k2*d_1*phi*d_tst3*fd.dx \
		- lamd1*c_n3*d_tst3*fd.dx \
		+ lamd2*d_3*d_tst3*fd.dx

		return Form

class temp_poisson(PDESubsystem):

	def form1(self, T_, T_tst, T_n, deltat, source, **kwargs):
		"""
		variational form of Poisson's equation for Temperature
		"""
		Form = T_*T_tst*fd.dx \
		+ deltat*fd.dot(fd.nabla_grad(T_), fd.nabla_grad(T_tst))*fd.dx \
		- (T_n + deltat * source)*T_tst*fd.dx

		return Form

class depth_avg_hydro(PDESubsystem):

	def form1(self,  z_trl1, z_n1, z_trl2, z_n2, z_tst1, z_tst2, D, n, g, A, tau, deltat, boundary_marker,
	 			boundary_marker2, **kwargs):
		"""
		Shallow water hydrodynamic equation with no normal boundary conditions.
		boundary_marker1 is the inlet.
		"""
		# mid point rule
		u_mid = 0.5*(z_trl1 + z_n1)
		h_mid = 0.5*(z_trl2 + z_n2)
		magnitude = fd.sqrt(fd.dot(z_n1, z_n1))
		H = D + z_n2

		# Continuity equation
		Form = (1.0/deltat)*fd.inner(z_trl2 - z_n2, z_tst2)*fd.dx \
		- H*fd.inner(u_mid, fd.grad(z_tst2))*fd.dx \
		+ fd.inner(u_mid, n)*z_tst2*fd.ds(boundary_marker) \
		+ fd.inner(u_mid, n)*z_tst2*fd.ds(boundary_marker2)
		# Momentum equation
		Form += (1.0/deltat)*fd.inner(z_trl1 - z_n1, z_tst1)*fd.dx \
		+ fd.inner(fd.dot(fd.grad(u_mid), z_n1), z_tst1)*fd.dx \
		+ fd.dot(g*fd.grad(h_mid), z_tst1)*fd.dx \
		+ A*fd.inner(fd.grad(u_mid)+fd.grad(u_mid).T, fd.grad(z_tst1))*fd.dx \
		- A*(2./3.)*fd.inner(fd.div(u_mid)*fd.Identity(len(u_mid)), fd.grad(z_tst1))*fd.dx \
		+ fd.dot(z_tst1, tau*magnitude/(H)*u_mid)*fd.dx

		return Form

class depth_avg_hydro_flather(PDESubsystem):

	def form1(self,  z_trl1, z_n1, z_trl2, z_n2, z_tst1, z_tst2, D, n, g, A, tau, deltat, boundary_marker,
	 			boundary_marker2, u_ext, h_ext, **kwargs):
		"""
		Shallow water hydrodynamic equation with Flather boundary conditions.
		bounadry_marker is the inlet and boundary_marker2 is the outlet.
		"""
		# mid point rule
		u_mid = 0.5*(z_trl1 + z_n1)
		h_mid = 0.5*(z_trl2 + z_n2)
		magnitude = fd.sqrt(fd.dot(z_n1, z_n1))
		H = D + z_n2

		# Continuity equation
		Form = (1.0/deltat)*fd.inner(z_trl2 - z_n2, z_tst2)*fd.dx \
		- H*fd.inner(u_mid, fd.grad(z_tst2))*fd.dx \
		+ fd.inner(u_mid, n)*z_tst2*fd.ds(boundary_marker) \
		+ H*fd.sqrt(g/H)*(h_mid - h_ext)*z_tst2*fd.ds(boundary_marker2) \
		+ H*fd.inner(fd.dot(u_ext, n), z_tst2)*fd.ds(boundary_marker2)

		# Momentum equation
		Form += (1.0/deltat)*fd.inner(z_trl1 - z_n1, z_tst1)*fd.dx \
		+ fd.inner(fd.dot(fd.grad(u_mid), z_n1), z_tst1)*fd.dx \
		+ fd.dot(g*fd.grad(h_mid), z_tst1)*fd.dx \
		+ A*fd.inner(fd.grad(u_mid)+fd.grad(u_mid).T, fd.grad(z_tst1))*fd.dx \
		- A*(2./3.)*fd.inner(fd.div(u_mid)*fd.Identity(len(u_mid)), fd.grad(z_tst1))*fd.dx \
		+ fd.dot(z_tst1, tau*magnitude/(H)*u_mid)*fd.dx

		return Form
