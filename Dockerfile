FROM firedrakeproject/firedrake

#Install

RUN . firedrake/bin/activate; pip3 install numpy
RUN . firedrake/bin/activate; pip3 install matplotlib
RUN . firedrake/bin/activate; pip3 install pytest

WORKDIR ./acse/fireframe
RUN ./acse/fireframe/ firedrake/bin/activate; export PYTHONPATH=$(pwd):$PYTHONPATH; pytest test/tests.py
