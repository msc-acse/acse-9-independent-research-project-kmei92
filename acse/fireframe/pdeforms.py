from PDESubsystem import *

class navier_stokes(PDESubsystem):

	def form1(self, u_trl, u_tst, u_n, p_n, n, k, f, nu, **kwargs):

		def sigma(u, p):
			return 2*nu*fd.sym(fd.nabla_grad(u)) - p*fd.Identity(len(u))

		u_mid = 0.5 * (u_n + u_trl)

		Form = fd.inner((u_trl - u_n)/k, u_tst) * fd.dx \
		+ fd.inner(fd.dot(u_n, fd.nabla_grad(u_mid)), u_tst) * fd.dx \
		+ fd.inner(sigma(u_mid, p_n), fd.sym(fd.nabla_grad(u_tst))) * fd.dx \
		+ fd.inner(p_n * n, u_tst) * fd.ds \
		- fd.inner(nu * fd.dot(fd.nabla_grad(u_mid), n), u_tst) * fd.ds \
		- fd.inner(f, u_tst) * fd.dx

		return Form

	def form2(self, p_trl, p_tst, p_n, u_, k, **kwargs):

		Form = fd.inner(fd.nabla_grad(p_trl), fd.nabla_grad(p_tst)) * fd.dx \
		- fd.inner(fd.nabla_grad(p_n), fd.nabla_grad(p_tst)) * fd.dx \
		+ (1/k) * fd.inner(fd.div(u_), p_tst) * fd.dx

		return Form

	def form3(self, u_trl, u_tst, u_, p_n, p_, k, **kwargs):

		Form = fd.inner(u_trl, u_tst) * fd.dx \
		- fd.inner(u_, u_tst) * fd.dx \
		+ k * fd.inner(fd.nabla_grad(p_ - p_n), u_tst) * fd.dx

		return Form

# class navier_stokes_coupled(PDESubsystem):
#
# 	def form1(self, up_trl1, up_trl2, up_tst1, up_tst2, up_1, up_n1, up_2, up_n2, f, nu, k, **kwargs):
#
# 		Form = fd.inner((up_trl1 - up_n1) / k), up_tst1) * fd.dx \
# 		+ nu*fd.inner((fd.grad(up_trl1) + fd.grad(up_trl1).T), fd.grad(up_tst1)) * fd.dx \
# 		- up_trl2*fd.div(up_tst1)*fd.dx -fd.div(up_trl1)*up_tst2*fd.dx \
# 		-fd.inner(f, up_tst1)*fd.dx
#
# 		return form

class reactions(PDESubsystem):

	def form1(self, c_1, c_n1, c_tst1, c_2, c_n2, c_tst2, c_3, c_n3, c_tst3, u_, eps, K, k, f_1, f_2, f_3, **kwargs):
		# x, y = fd.SpatialCoordinate(self.mesh)
		# f_1 = fd.conditional(pow(x-0.1, 2)+pow(y-0.1,2)<0.05*0.05, 0.1, 0)
		# f_2 = fd.conditional(pow(x-0.1, 2)+pow(y-0.3,2)<0.05*0.05, 0.1, 0)
		# f_3 = fd.Constant(0.0)

		Form = ((c_1 - c_n1) / k)*c_tst1*fd.dx \
		+ ((c_2 - c_n2) / k)*c_tst2*fd.dx \
		+ ((c_3 - c_n3) / k)*c_tst3*fd.dx \
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

	def form1(self, c1_trl, c1_n, c1_tst, c2_n, u_, eps, K, k, f_1, **kwargs):

		Form = ((c1_trl - c1_n) / k)*c1_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c1_trl)), c1_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c1_trl), fd.grad(c1_tst))*fd.dx \
		+K*c1_trl*c2_n*c1_tst*fd.dx \
		-f_1*c1_tst*fd.dx

		return Form

	def form2(self, c2_trl, c2_n, c2_tst, c1_n, u_, eps, K, k, f_2, **kwargs):

		Form = ((c2_trl - c2_n) / k)*c2_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c2_trl)), c2_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c2_trl), fd.grad(c2_tst))*fd.dx \
		+K*c2_trl*c1_n*c2_tst*fd.dx \
		-f_2*c2_tst*fd.dx

		return Form

	def form3(self, c3_trl, c3_n, c3_tst, c1_n, c2_n, u_, eps, K, k, f_3, **kwargs):

		Form = ((c3_trl - c3_n) / k)*c3_tst*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(c3_trl)), c3_tst)*fd.dx \
		+ eps*fd.dot(fd.grad(c3_trl), fd.grad(c3_tst))*fd.dx \
		- K*c1_n*c2_n*c3_tst*fd.dx \
		+ K*c3_trl*c3_tst*fd.dx \
		-f_3*c3_tst*fd.dx

		return Form

class radio_transport(PDESubsystem):

	def form1(self, cd_1, cd_n1, cd_tst1, cd_2, cd_n2, cd_tst2, as_1, cs_1, as_2, cs_2, u_,
			  k, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, **kwargs):

		Form = ((cd_1 - cd_n1) / k)*cd_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cd_1)), cd_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cd_1), fd.grad(cd_tst1))*fd.dx \
		+ k1*cd_1*cd_tst1*fd.dx \
		- k2*cs_1*cd_tst1*fd.dx \
		- k2*as_1*L*rho_s*frac*phi*cd_tst1*fd.dx \
		+ lamd1*cd_1*cd_tst1*fd.dx \
		+ ((cd_2 - cd_n2) / k)*cd_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cd_2)), cd_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(cd_2), fd.grad(cd_tst2))*fd.dx \
		+ k1*cd_2*cd_tst2*fd.dx \
		- k2*cs_2*cd_tst2*fd.dx \
		- k2*as_2*L*rho_s*frac*phi*cd_tst2*fd.dx \
		+ lamd2*cd_2*cd_tst2*fd.dx \
		- lamd1*cd_1*cd_tst2*fd.dx

		return Form

	def form2(self, cs_1, cs_n1, cs_tst1, cs_2, cs_n2, cs_tst2, cd_1, cd_2, u_,
			  k, Kd, k1, k2, lamd1, lamd2, **kwargs):

		Form = ((cs_1 - cs_n1) / k)*cs_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_1)), cs_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_1), fd.grad(cs_tst1))*fd.dx \
		- k1*cd_1*cs_tst1*fd.dx \
		+ k2*cs_1*cs_tst1*fd.dx \
		+ lamd1*cs_1*cs_tst1*fd.dx \
		+ ((cs_2 - cs_n2) / k)*cs_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_2)), cs_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_2), fd.grad(cs_tst2))*fd.dx \
		- k1*cd_2*cs_tst2*fd.dx \
		+ k2*cs_2*cs_tst2*fd.dx \
		+ lamd2*cs_2*cs_tst2*fd.dx \
		- lamd1*cs_1*cs_tst2*fd.dx

		return Form


	def form3(self, as_1, as_n1, as_tst1, as_2, as_n2, as_tst2, cd_1, cd_2, k, k1, k2, lamd1,
			  lamd2, rho_s, L, frac, phi, **kwargs):

		Form = ((as_1 - as_n1) / k)*as_tst1*fd.dx \
		- (1./(L * rho_s * frac * phi)) * k1*cd_1*as_tst1*fd.dx \
		+ k2*as_1*phi*as_tst1*fd.dx \
		+ lamd1*as_1*as_tst1*fd.dx \
		+((as_2 - as_n2) / k)*as_tst2*fd.dx \
		- (1./(L * rho_s * frac * phi)) * k1*cd_2*as_tst2*fd.dx \
		+ k2*as_2*phi*as_tst2*fd.dx \
		+ lamd2*as_2*as_tst2*fd.dx \
		- lamd1*as_1*as_tst2*fd.dx

		return Form

class temp_poisson(PDESubsystem):

	def form1(self, T_, T_tst, T_n, k, source, **kwargs):
		Form = T_*T_tst*fd.dx \
		+ k*fd.dot(fd.nabla_grad(T_), fd.nabla_grad(T_tst))*fd.dx \
		- (T_n + k * source)*T_tst*fd.dx

		return Form


class radio_transport_mms(PDESubsystem):

	def form1(self, cd_1, cd_n1, cd_tst1, cd_2, cd_n2, cd_tst2, as_1, cs_1, as_2, cs_2, u_,
			  k, Kd, k1, k2, lamd1, lamd2, rho_s, L, phi, frac, t, **kwargs):

		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		dc_dt = x*y*c_source
		grad_c = fd.as_vector((y*t*c_source, x*t*c_source))
		laplace_c = (y*t)**2 * c_source + (x*t)**2*c_source
		source1 = dc_dt + fd.dot(u_, grad_c) - Kd*laplace_c + k1*c_source + lamd1*c_source
		source2 = lamd1*c_source

		Form = ((cd_1 - cd_n1) / k)*cd_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cd_1)), cd_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cd_1), fd.grad(cd_tst1))*fd.dx \
		+ k1*cd_1*cd_tst1*fd.dx \
		- k2*cs_1*cd_tst1*fd.dx \
		- k2*as_1*L*rho_s*frac*phi*cd_tst1*fd.dx \
		+ lamd1*cd_1*cd_tst1*fd.dx \
		+ ((cd_2 - cd_n2) / k)*cd_tst2*fd.dx \
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
			  k, Kd, k1, k2, lamd1, lamd2, t, **kwargs):

		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		source1 = -k1*c_source

		Form = ((cs_1 - cs_n1) / k)*cs_tst1*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_1)), cs_tst1)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_1), fd.grad(cs_tst1))*fd.dx \
		- k1*cd_1*cs_tst1*fd.dx \
		+ k2*cs_1*cs_tst1*fd.dx \
		+ lamd1*cs_1*cs_tst1*fd.dx \
		+ ((cs_2 - cs_n2) / k)*cs_tst2*fd.dx \
		+ fd.inner(fd.dot(u_, fd.nabla_grad(cs_2)), cs_tst2)*fd.dx \
		+ Kd*fd.dot(fd.grad(cs_2), fd.grad(cs_tst2))*fd.dx \
		- k1*cd_2*cs_tst2*fd.dx \
		+ k2*cs_2*cs_tst2*fd.dx \
		+ lamd2*cs_2*cs_tst2*fd.dx \
		- lamd1*cs_1*cs_tst2*fd.dx \
		- source1*cs_tst1*fd.dx

		return Form

	def form3(self, as_1, as_n1, as_tst1, as_2, as_n2, as_tst2, cd_1, cd_2, k, k1, k2, lamd1,
			  lamd2, rho_s, L, frac, phi, t, **kwargs):
		x, y = fd.SpatialCoordinate(self.mesh)
		c_source = fd.exp(x*y*t)
		source1 = (1./(L * rho_s * frac * phi)) * c_source

		Form = ((as_1 - as_n1) / k)*as_tst1*fd.dx \
		- (1./(L * rho_s * frac * phi)) * k1*cd_1*as_tst1*fd.dx \
		+ k2*as_1*phi*as_tst1*fd.dx \
		+ lamd1*as_1*as_tst1*fd.dx \
		+((as_2 - as_n2) / k)*as_tst2*fd.dx \
		- (1./(L * rho_s * frac * phi)) * k1*cd_2*as_tst2*fd.dx \
		+ k2*as_2*phi*as_tst2*fd.dx \
		+ lamd2*as_2*as_tst2*fd.dx \
		- lamd1*as_1*as_tst2*fd.dx \
		- source1*as_tst1*fd.dx

		return Form
