FROM firedrakeproject/firedrake

#Install

RUN . firedrake/bin/activate; pip3 install numpy
RUN . firedrake/bin/activate; pip3 install matplotlib

RUN mkdir -p /home/firedrake/src/
WORKDIR /home/firedrake/src/
COPY . /home/firedrake/src/
RUN . /home/firedrake/firedrake/bin/activate
