### Build Status
[![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-kmei92.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-kmei92)

!(imgs/fireframe_rxns.png=200x50 "Demonstration of three component reaction flow")

## Introduction
Fireframe is a programmable framework built on top of the Firedrake finite element library. Firedrake aims to automate the process
of setting up function spaces, creating finite elements, and performing manufactured solutions testing.
The goal of Fireframe is to allow users to quickly and efficiently analyze different combinations of coupled partial differential equations.

Included in this repository are demonstration notebooks that highlight how to pose Firedrake problems using Fireframe.

Example problems are presented and solved, including:
 - A simple channel flow navier-stokes equation
 - A poisson equation 
 - A three component chemical reactions  equation coupled to the navier-stokes equation
 - A radionuclide transport equation coupled to the navier-stokes equation
 - A radionuclide transport equation coupled to the shallow water equation
 - A method of manufactured solutions verification on the radionuclide transport problem

## Installation instructions
In order to use Fireframe, please first install the latest version of Firedrake [here](https://www.firedrakeproject.org/download.html)

To download Fireframe:

  git clone https://github.com/msc-acse/acse-9-independent-research-project-kmei92.git

Before attempting to use Fireframe or any Firedrake functions, activate the Firedrake virtualenv first:

  source firedrake/bin/activate

To use Fireframe, import the following:
```python
from PDESystem import *

from PDESubsystem import *

from pdeforms import *
```

## Dependencies
Fireframe relies on the following external libraries:

 - numpy >= 1.16.4
 - sympy >= 1.4
 - matplotlib >= 3.1.0
 - jupyter >= 1.0.0 (to run demo notebooks)

## Repository Information
* __docs__				-
* __tests__				-
* __acse__				-
    * __acse__/meshes/			-
    * __acse__/fireframe/		-
        * __acse__/fireframe/demo	-

## Author and Course Information
__Author:__ Keer Mei
__Github:__ kmei92
__CID:__ 01545321

This project is completed for Imperial College's MSc in Applied Computational Science and Engineering program,
as part of the final course module ACSE9. This project was completed under and supervision of Professor Matthew Piggott. 
## License
Fireframe is licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-kmei92/blob/master/LICENSE)

