FROM firedrakeproject/firedrake

#Install

RUN . firedrake/bin/activate; pip3 install numpy
RUN . firedrake/bin/activate; pip3 install matplotlib
RUN . firedrake/bin/activate; pip3 install pytest

