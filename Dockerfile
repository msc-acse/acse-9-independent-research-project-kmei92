FROM firedrakeproject/firedrake

#Install

RUN . firedrake/bin/activate; pip3 install numpy
RUN . firedrake/bin/activate; pip3 install matplotlib
RUN . firedrake/bin/activate; pip3 install pytest

RUN . firedrake/bin/activate; export PYTHONPATH=$(pwd):$PYTHONPATH;
COPY acse/fireframe/test/tests.py .
CMD firedrake/bin/activate && pytest tests.py
