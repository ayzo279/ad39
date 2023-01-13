# AD39
[![Tests](https://code.harvard.edu/CS107/team39/actions/workflows/test.yml/badge.svg?branch=master)](https://code.harvard.edu/CS107/team39/actions/workflows/test.yml)
[![Coverage](https://code.harvard.edu/CS107/team39/actions/workflows/coverage.yml/badge.svg?branch=master)](https://code.harvard.edu/CS107/team39/actions/workflows/coverage.yml)

This package provides a python implementation for automatic differentiation (AD).
The package supports both forward mode (FM) as well as reverse mode (RM) AD. To this end, the library supports differentiation 
for both univariate as well as multivariate inputs and supports both scalar and vector functions. That is, in the most general case
this librabry supports use cases of the form: $f: \mathbb{R}^m \mapsto \mathbb{R}^n$.

Installation
============

### Install Package
- From PyPI:

      python -m pip install --extra-index-url https://pypi.org/simple/ -i https://test.pypi.org/simple/ ad39-package

- From GitHub:

      git clone git@code.harvard.edu:CS107/team39.git

### Dependencies
AD39 requires:
* Python
* NumPy

Examples
=========================

```python
# Import forward and reverse mode classes (along with coressponding methods)
# from the two subpackages of `ad39_package`
from ad39_package.forward import FM
from ad39_package.reverse import RM

# Import numpy for later use
import numpy as np
```
### Forward Mode

```python
# Define target function: this can be a scalar- or vector-valued function,
# although vector-valued options must output a numpy array with the function
# operations defined within the array, i.e. 
# DO NOT USE `lambda x : np.array([1, 2]) + x`
# USE `lambda x : np.array([1 + x[0], 2 + x[1])`
f = lambda x: np.array([x[0] ** 2 + x[1] ** 2, FM.sqrt(x[0])])  

# Define example input vector
vec = np.array([4, 3])

# Instantiate an object for performing forward mode AD on the desired function
FMtest = FM(f)

# Evaluate the function by calling the `eval` method with argument `vec`
result = FMtest.eval(vec)
print(f'f([2, 3]) = {result}') # result = [25, 2]

# Compute the derivative of the function w.r.t. each of the two vector inputs
# evaluated at `vec` by calling the `grad` method with argument `vec` and
# `seed=[1,0]` or `seed=[0,1]` (if no seed is supplied, default will be to
# take derivative w.r.t. the first input - in this case x_0)
deriv_x0 = FMtest.grad(vec, seed=[1,0])
print(f'deriv w.r.t x_0 = {deriv_x0}') # deriv_x0 = [8, 0.25]

deriv_x1 = FMtest.grad(vec, seed=[0, 1])
print(f'deriv w.r.t x_1 = {deriv_x1}') # deriv_x1 = [6, 0]

# Compute the Jacobian evaluated at `vec` by calling the `jacobian`
# method with argument `vec`
jacob = FMtest.jacobian(vec)
print(f'The Jacobian of f([2, 3]) is {jacob}.') # jacob = [[8, 6], [0.25, 0]]
```
### Extension 1: Forward Mode for multiple scalar and/or vector inputs

```python
# Define target function with multiple vector and/or scalar inputs
f = lambda x, y: x[0] + y ** 2

# Define example inputs
vec = np.array([1, 2])
scalar = 3

# Instantiate an object for performing forward mode AD on the desired function
FMtest = FM(f)

# Evaluate the function with multiple inputs passed as a tuple
result = FMtest.eval((vec, scalar))
print(f'f([1, 2], 3) = {result}') # result = 10

# Compute the derivative of the function w.r.t. each of the two vector inputs
# and one scalar input evaluated at the tuple of inputs
deriv_x0 = FMtest.grad((vec, scalar), seed=[1,0,0])
print(f'deriv w.r.t x_0 = {deriv_x0}') # deriv_x0 = 1

deriv_x1 = FMtest.grad((vec, scalar), seed=[0, 1, 0])
print(f'deriv w.r.t x_1 = {deriv_x1}') # deriv_x1 = 0

deriv_y = FMtest.grad((vec, scalar), seed=[0,0,1])
print(f'deriv w.r.t y = {deriv_y}') # deriv_y = 6

# Compute the Jacobian evaluated at the tuple of inputs
jacob = FMtest.jacobian((vec, scalar))
print(f'The Jacobian of f([1, 2], 3) is {jacob}') # jacob = [1, 0, 6]
```

### Extension 2: Reverse Mode

```python
# Define target function: this can be a scalar- or vector-valued function,
# although vector-valued options must output a numpy array with the function
# operations defined within the array, i.e. 
# DO NOT USE `lambda x : np.array([1, 2]) + x`
# USE `lambda x : np.array([1 + x[0], 2 + x[1])`
f = lambda x: np.array([x[0] ** 2 + x[1] ** 2, RM.sqrt(x[0])])  

# Define example input vector
vec = np.array([4, 3])

# Instantiate an object for performing forward mode AD on the desired function
RMtest = RM(f)

# Evaluate the function by calling the `eval` method with argument `vec`
result = RMtest.eval(vec)
print(f'f([2, 3]) = {result}') # result = [25, 2]

# Compute the derivative of the function w.r.t. each of the two vector inputs
# evaluated at `vec` by calling the `grad` method with argument `vec`. Note,
# that no seed vector is required; `RM.grad` returns the entire gradient w.r.t.
# each input.
gradient_vec = RMtest.grad(vec)
print(f'gradient = {gradient_vec}') # gradient = [[8, 0.25], [6, 0]]

# Compute the Jacobian evaluated at `vec` by calling the `jacobian`
# method with argument `vec`
jacob = FMtest.jacobian(vec)
print(f'The Jacobian of f([2, 3]) is {jacob}.') # jacob = [[8, 6], [0.25, 0]]
```
### Extension 4: Computational Graph

```python
# Draw the computational graph of the function evaluated at the vector defined above
RMtest.graph(vec)

```

Broader Impact and Inclusivity Statement
=========================

#### Broader Impact

Automatic Differentiation (AD) has emerged as an indispendable tool in nearly all computational fields today. As such,
it has far reaching implications on many areas of our daily lives, with many people not being aware of this. The arguably most 
prominent example of this is AD's role in modern machine learning algorithms, first and foremost it's application in training
deep neural networks. That is, training neural networks utilizes a form of reverse mode AD called the backpropagation algorithm. 
Consequently, AD's is indirectly impacting many fields in modern society in which deep learning is being utilized such as face recognition software,
loan-approval or other rating algorithms. Predictions from such algorithms are known to fail in certain cases and can produce biased results.
It is therefore critical to always question the results obtained by such algorithms and not to base decisions solely on what the results suggest.

#### Inclusivity

We explicitly welcome individuals from any background, ethnicity or sexual orientation to contribute to this project. We realize is often 
easier said that done. If you are interested in joining our core developer team or learn about other ways to contribute please don't hesiatate to reach out to us. 
As our code base and documentation is currently written entirley in English we would particularly welcome contributions in helping translate our documentation into other 
languages to make it easier to use for a broader audience.

#### How To Contribute

If you wish to contribute to this package we recommend creating a fork from https://code.harvard.edu/CS107/team39.git and submitting a pull request with a clear discription of the problem/desired feature you think is missing along with your proposed solution.