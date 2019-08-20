FROM firedrakeproject/firedrake

#Install
COPY requirements.txt .
RUN . firedrake/bin/activate; pip3 install -r requirements.txt
RUN . firedrake/bin/activate; pip3 install pytest


RUN mkdir -m /home/
WORKDIR /home/
COPY . /home/
RUN . /home/firedrake/bin/activate; export PYTHONPATH=$(pwd):$PYTHONPATH; pytest tests/tests.py
