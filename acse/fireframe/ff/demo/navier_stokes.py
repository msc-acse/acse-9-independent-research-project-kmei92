import sys
sys.path.append("..")
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
