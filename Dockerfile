FROM firedrakeproject/firedrake

#Install
COPY requirements.txt .
RUN . firedrake/bin/activate; pip3 install -r requirements.txt
RUN . firedrake/bin/activate; pip3 install pytest


RUN mkdir -p /home/firedrake/src/acse/fireframe/
WORKDIR /home/firedrake/src/acse/fireframe/
COPY . /home/firedrake/src/acse/fireframe/
RUN . /home/firedrake/firedrake/bin/activate; export PYTHONPATH=$(pwd):$PYTHONPATH; pytest acse/fireframe/tests_local.py
