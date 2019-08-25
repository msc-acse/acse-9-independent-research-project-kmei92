### Build Status
[![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-kmei92.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-kmei92)

<img src="imgs/fireframe_rxns.png" title="Three component flow coupled chemical reactions" width="400" height="300" /><img src="imgs/gibraltar_flow.png" title="Velocity of shallow water equations solved on the Strait of Gibraltar" width="400" height="300" />

## Introduction
Fireframe is a programmable framework built on top of the Firedrake finite element library. Firedrake aims to automate the process
of setting up function spaces, creating finite elements, and performing manufactured solutions testing.
The goal of Fireframe is to allow users to quickly and efficiently analyze different combinations of coupled partial differential equations.

Included in this repository are demonstration notebooks that highlight how to pose Firedrake problems using Fireframe.

Example problems are presented and solved, including:
 - A simple channel flow navier-stokes equation (see: demo/flow_past_cylinder_demo.ipynb)
 - A poisson equation (see: demo/poisson_temp_demo.ipynb)
 - A three component chemical reactions  equation coupled to the navier-stokes equation (see: demo/cylinder_rxn_demo.ipynb)
 - A radionuclide transport equation coupled to the navier-stokes equation (see: demo/radio_transport_demo.ipynb)
 - A radionuclide transport equation coupled to the shallow water equation (see: demo/hydrodynamics_demo.ipynb)
 - A method of manufactured solutions verification on the radionuclide transport problem (see: demo/MMS_transport/demo.ipynb)

## Installation instructions
In order to use Fireframe, please first install the latest version of Firedrake [here](https://www.firedrakeproject.org/download.html)

To download Fireframe:
```bash
  git clone https://github.com/msc-acse/acse-9-independent-research-project-kmei92.git
```
Before attempting to use Fireframe or any Firedrake functions, activate the Firedrake virtualenv first:
```bash
  source firedrake/bin/activate
```
To use Fireframe, import the following:
```python
from PDESystem import *

from PDESubsystem import *

from pdeforms import *
```
## Documentation
The Fireframe documentation can be compiled using Sphinx by running
```bash
make html
```

from the `docs` directory. After compiling the documentation, access the html file `index.html` inside the `docs/build/html` directory.

## Dependencies
Fireframe relies on the following external libraries:

 - numpy >= 1.16.4
 - sympy >= 1.4
 - matplotlib >= 3.1.0
 - jupyter >= 1.0.0 (to run demo notebooks)
 - Sphinx >= 1.8.5 (to compile documentation)
## Repository Information
* __demo__		- demonstration examples using Fireframe
* __docs__		- all files required to compile documentation
* __fireframe__		- main repository containing core modules `PDESystem`, `PDESubsystem` and `pdeforms`
* __imgs__		- image files
* __meshes__		- all gmsh files used in demonstration examples
* __tests__		- pytest / travis test file

## Author and Course Information
__Author:__ Keer Mei
__Github:__ kmei92
__CID:__ 01545321

This project is completed for Imperial College's MSc in Applied Computational Science and Engineering program,
as part of the final course module ACSE9. This project was completed under the supervision of Professor Matthew Piggott. 
## License
Fireframe is licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-kmei92/blob/master/LICENSE)

