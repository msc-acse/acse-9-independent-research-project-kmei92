"""
author : Keer Mei
email: keer.mei18@imperial.ac.uk
github username: kmei92
"""

from acse.fireframe.PDESubsystem import *

class navier_stokes(PDESubsystem):

	def form1(self, u_trl, u_tst, u_n, p_n, n, deltat, f, nu, **kwargs):

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

		Form = fd.inner(fd.nabla_grad(p_trl), fd.nabla_grad(p_tst)) * fd.dx \
		- fd.inner(fd.nabla_grad(p_n), fd.nabla_grad(p_tst)) * fd.dx \
		+ (1/deltat) * fd.inner(fd.div(u_), p_tst) * fd.dx

		return Form

	def form3(self, u_trl, u_tst, u_, p_n, p_, deltat, **kwargs):

		Form = fd.inner(u_trl, u_tst) * fd.dx \
		- fd.inner(u_, u_tst) * fd.dx \
		+ deltat * fd.inner(fd.nabla_grad(p_ - p_n), u_tst) * fd.dx

		return Form

class reactions(PDESubsystem):

	def form1(self, c_1, c_n1, c_tst1, c_2, c_n2, c_tst2, c_3, c_n3, c_tst3, u_, eps, K, deltat, f_1, f_2, f_3, **kwargs):

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

		Form = ((c1_trl - c1_n) / deltat)*c1_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c1_trl)), c1_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c1_trl), fd.grad(c1_tst))*fd.dx \
		+K*c1_trl*c2_n*c1_tst*fd.dx \
		-f_1*c1_tst*fd.dx

		return Form

	def form2(self, c2_trl, c2_n, c2_tst, c1_n, u_, eps, K, deltat, f_2, **kwargs):

		Form = ((c2_trl - c2_n) / deltat)*c2_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c2_trl)), c2_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c2_trl), fd.grad(c2_tst))*fd.dx \
		+K*c2_trl*c1_n*c2_tst*fd.dx \
		-f_2*c2_tst*fd.dx

		return Form

	def form3(self, c3_trl, c3_n, c3_tst, c1_n, c2_n, u_, eps, K, deltat, f_3, **kwargs):

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

		# manufactured source term
		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		dc_dt = x*y*c_source
		grad_c = fd.as_vector((y*t*c_source, x*t*c_source))
		laplace_c = (y*t)**2 * c_source + (x*t)**2*c_source
		source1 = dc_dt + fd.dot(u_, grad_c) - Kd*laplace_c + k1*c_source + lamd1*c_source
		source2 = lamd1*c_source

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
				u_, c_n3, deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, **kwargs):

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

class temp_poisson(PDESubsystem):

	def form1(self, T_, T_tst, T_n, deltat, source, **kwargs):
		Form = T_*T_tst*fd.dx \
		+ deltat*fd.dot(fd.nabla_grad(T_), fd.nabla_grad(T_tst))*fd.dx \
		- (T_n + deltat * source)*T_tst*fd.dx

		return Form

class radio_transport(PDESubsystem):

	def form1(self, cd_1, cd_n1, cd_tst1, cd_2, cd_n2, cd_tst2, as_n1, cs_n1, as_n2, cs_n2, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, source1, **kwargs):

		# first species
		Form = ((cd_1 - cd_n1) / deltat)*cd_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cd_n1)), cd_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cd_n1), fd.grad(cd_tst1))*fd.dx \
		+ k1*cd_1*cd_tst1*fd.dx \
		- k2*cs_n1*cd_tst1*fd.dx \
		- k2*as_n1*L*rho_s*frac*phi*cd_tst1*fd.dx \
		+ lamd1*cd_1*cd_tst1*fd.dx \
		- source1*cd_tst1*fd.dx

		# #second species
		# Form += ((cd_2 - cd_n2) / deltat)*cd_tst2*fd.dx \
		# + fd.inner(fd.dot(u_, fd.nabla_grad(cd_2)), cd_tst2)*fd.dx \
		# + Kd*fd.dot(fd.grad(cd_2), fd.grad(cd_tst2))*fd.dx \
		# + k1*cd_2*cd_tst2*fd.dx \
		# - k2*cs_n2*cd_tst2*fd.dx \
		# - k2*as_n2*L*rho_s*frac*phi*cd_tst2*fd.dx \
		# + lamd2*cd_2*cd_tst2*fd.dx \
		# - lamd1*cd_1*cd_tst2*fd.dx

		return Form

	def form2(self, cs_1, cs_n1, cs_tst1, cs_2, cs_n2, cs_tst2, cd_n1, cd_n2, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, **kwargs):
		# first species
		Form = ((cs_1 - cs_n1) / deltat)*cs_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_1)), cs_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_1), fd.grad(cs_tst1))*fd.dx \
		- k1*cd_n1*cs_tst1*fd.dx \
		+ k2*cs_1*cs_tst1*fd.dx \
		+ lamd1*cs_1*cs_tst1*fd.dx

		# # second species
		# Form += ((cs_2 - cs_n2) / k)*cs_tst2*fd.dx \
		# + fd.inner(fd.dot(u_, fd.nabla_grad(cs_2)), cs_tst2)*fd.dx \
		# + Kd*fd.dot(fd.grad(cs_2), fd.grad(cs_tst2))*fd.dx \
		# - k1*cd_n2*cs_tst2*fd.dx \
		# + k2*cs_2*cs_tst2*fd.dx \
		# + lamd2*cs_2*cs_tst2*fd.dx \
		# - lamd1*cs_1*cs_tst2*fd.dx

		return Form


	def form3(self, as_1, as_n1, as_tst1, as_2, as_n2, as_tst2, cd_n1, cd_n2, deltat, k1, k2, lamd1,
			  lamd2, rho_s, L, frac, phi, **kwargs):

		#first species
		Form = ((as_1 - as_n1) / deltat)*as_tst1*fd.dx \
		- (1./(L * rho_s * frac)) * k1*cd_n1*as_tst1*fd.dx \
		+ k2*as_1*phi*as_tst1*fd.dx \
		+ lamd1*as_1*as_tst1*fd.dx

		# #second species
		# Form += ((as_2 - as_n2) / deltat)*as_tst2*fd.dx \
		# - (1./(L * rho_s * frac)) * k1*cd_n2*as_tst2*fd.dx \
		# + k2*as_2*phi*as_tst2*fd.dx \
		# + lamd2*as_2*as_tst2*fd.dx \
		# - lamd1*as_1*as_tst2*fd.dx

		return Form

class radio_transport_mms(PDESubsystem):

	def form1(self, cd_1, cd_n1, cd_tst1, cd_2, cd_n2, cd_tst2, as_1, cs_1, as_2, cs_2, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, t, **kwargs):

		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		dc_dt = x*y*c_source
		grad_c = fd.as_vector((y*t*c_source, x*t*c_source))
		laplace_c = (y*t)**2 * c_source + (x*t)**2*c_source
		source1 = dc_dt + fd.dot(u_, grad_c) - Kd*laplace_c + k1*c_source + lamd1*c_source
		source2 = lamd1*c_source

		Form = ((cd_1 - cd_n1) / deltat)*cd_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cd_1)), cd_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cd_1), fd.grad(cd_tst1))*fd.dx \
		+ k1*cd_1*cd_tst1*fd.dx \
		- k2*cs_1*cd_tst1*fd.dx \
		- k2*as_1*L*rho_s*frac*phi*cd_tst1*fd.dx \
		+ lamd1*cd_1*cd_tst1*fd.dx \
		+ ((cd_2 - cd_n2) / deltat)*cd_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cd_2)), cd_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(cd_2), fd.grad(cd_tst2))*fd.dx \
		+ k1*cd_2*cd_tst2*fd.dx \
		- k2*cs_2*cd_tst2*fd.dx \
		- k2*as_2*L*rho_s*frac*phi*cd_tst2*fd.dx \
		+ lamd2*cd_2*cd_tst2*fd.dx \
		- lamd1*cd_1*cd_tst2*fd.dx \
		- source1*cd_tst1*fd.dx \
		- source2*cd_tst2*fd.dx

		return Form

	def form2(self, cs_1, cs_n1, cs_tst1, cs_2, cs_n2, cs_tst2, cd_1, cd_2, u_,
			  deltat, Kd, k1, k2, lamd1, lamd2, t, **kwargs):

		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		source1 = -k1*c_source

		Form = ((cs_1 - cs_n1) / deltat)*cs_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_1)), cs_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_1), fd.grad(cs_tst1))*fd.dx \
		- k1*cd_1*cs_tst1*fd.dx \
		+ k2*cs_1*cs_tst1*fd.dx \
		+ lamd1*cs_1*cs_tst1*fd.dx \
		+ ((cs_2 - cs_n2) / deltat)*cs_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_2)), cs_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_2), fd.grad(cs_tst2))*fd.dx \
		- k1*cd_2*cs_tst2*fd.dx \
		+ k2*cs_2*cs_tst2*fd.dx \
		+ lamd2*cs_2*cs_tst2*fd.dx \
		- lamd1*cs_1*cs_tst2*fd.dx \
		- source1*cs_tst1*fd.dx

		return Form

	def form3(self, as_1, as_n1, as_tst1, as_2, as_n2, as_tst2, cd_1, cd_2, deltat, k1, k2, lamd1,
			  lamd2, rho_s, L, frac, phi, t, **kwargs):
		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		source1 = (1./(L * rho_s * frac * phi)) * c_source

		Form = ((as_1 - as_n1) / deltat)*as_tst1*fd.dx \
		- (1./(L * rho_s * frac * phi)) * k1*cd_1*as_tst1*fd.dx \
		+ k2*as_1*phi*as_tst1*fd.dx \
		+ lamd1*as_1*as_tst1*fd.dx \
		+((as_2 - as_n2) / deltat)*as_tst2*fd.dx \
		- (1./(L * rho_s * frac * phi)) * k1*cd_2*as_tst2*fd.dx \
		+ k2*as_2*phi*as_tst2*fd.dx \
		+ lamd2*as_2*as_tst2*fd.dx \
		- lamd1*as_1*as_tst2*fd.dx \
		- source1*as_tst1*fd.dx

		return Form

class depth_avg_hydro(PDESubsystem):

	def form1(self, u_, u_n, u_tst, h_, h_n, deltat, A, g, D, tau, **kwargs):
		u_mid = 0.5 * (u_ + u_n)
		h_mid = 0.5 * (h_ + h_n)
		H = D + h_
		magnitude = fd.sqrt(fd.dot(u_n, u_n))

		Form = (1.0/deltat)*fd.inner(u_ - u_n, u_tst)*fd.dx \
		+ fd.inner(fd.dot(fd.grad(u_mid), u_mid), u_tst)*fd.dx \
		+ A*fd.inner(fd.grad(u_mid)+fd.grad(u_mid).T, fd.grad(u_tst))*fd.dx \
		# - A*(2./3.)*fd.inner(fd.div(u_mid)*fd.Identity(len(u_mid)), fd.grad(u_tst))*fd.dx \
		# + g*fd.inner(u_tst * fd.grad(h_mid))*fd.dx \
		# + fd.inner(u_tst, (tau*magnitude/H)*u_mid)*fd.dx

		return Form

	def form2(self, h_, h_n, h_tst, u_n, u_, D, deltat, **kwargs):
		u_mid = 0.5 * (u_ + u_n)
		H = D + h_

		Form = (1.0/deltat)*fd.inner(h_ - h_n, h_tst)*fd.dx\
		+ fd.inner(h_tst, fd.div(H * u_mid))*fd.dx

		return Form
