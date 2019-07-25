import sys
sys.path.append("..")
from PDESubsystem import *

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
        - (1./(L * rho_s * frac)) * k1*cd_1*as_tst1*fd.dx \
        + k2*as_1*phi*as_tst1*fd.dx \
        + lamd1*as_1*as_tst1*fd.dx \
        +((as_2 - as_n2) / k)*as_tst2*fd.dx \
        - (1./(L * rho_s * frac)) * k1*cd_2*as_tst2*fd.dx \
        + k2*as_2*phi*as_tst2*fd.dx \
        + lamd2*as_2*as_tst2*fd.dx \
        - lamd1*as_1*as_tst2*fd.dx

        return Form
