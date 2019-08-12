FROM firedrakeproject/firedrake

#Install

RUN . firedrake/bin/activate; pip3 install numpy
RUN . firedrake/bin/activate; pip3 install matplotlib
RUN . firedrake/bin/activate; pip3 install pytest

RUN . firedrake/bin/activate
COPY acse/fireframe/test/tests.py .
CMD ["pytest", "tests.py"]
