FROM firedrakeproject/firedrake

#Install
COPY requirements.txt .
RUN . firedrake/bin/activate; pip3 install -r requirements.txt
RUN . firedrake/bin/activate; pip3 install pytest

COPY acse/fireframe/test/tests.py .
CMD . firedrake/bin/activate && pytest tests.py
