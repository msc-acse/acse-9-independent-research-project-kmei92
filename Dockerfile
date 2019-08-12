FROM firedrakeproject/firedrake

#Install
COPY requirements.txt .
RUN . firedrake/bin/activate; pip3 install -r requirements.txt
RUN . firedrake/bin/activate; pip3 install pytest

RUN mkdir -p /home/firedrake/src/
WORKDIR /home/firedrake/src/
COPY ./home/firedrake/src/
RUN . /home/firedrake/firedrake/bin/activate; xport PYTHONPATH=$(pwd):$PYTHONPATH; pytest
