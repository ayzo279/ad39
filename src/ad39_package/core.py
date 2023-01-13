#!/usr/bin/env python3
# File       : core.py
# Description: Module that provides base class for automatic differentiation (AD)

import numpy as np

class AD:
    
    def __init__(self, f):
        """Initialize an instance of an automatic differentiation object given a scalar-valued function `f`.

        Parameters
        ----------
        f : (int, float, np.ndarry, tuple) -> (int, float, np.ndarry)
            Function to which methods from the AD class are applied.
            Note the `reverse_mode` method excludes functionality for 
            functions taking tuple as input.
        
        See Also
        ----------
        ad39_package.forward.FM
        ad39_package.reverse.RM
        """
        self.f = f

        # Store number of args the function expects
        self._nargs = f.__code__.co_argcount


    def eval(self, inputs):
        """Evaluate the value of `self.f` at `inputs`."""
        raise NotImplementedError

    def grad(self, inputs, *args):
        """Compute the derivative of `self.f` evaluated at `inputs` using forward mode AD."""
        raise NotImplementedError
    
    def jacobian(self, inputs):
        """Compute the Jacobian of `self.f` evaluated at `inputs` using forward mode AD."""
        raise NotImplementedError

    @staticmethod
    def sqrt(x):
        """Compute sine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def sin(x):
        """Compute sine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def cos(x):
        """Compute cosine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def tan(x):
        """Compute sine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def arcsin(x):
        """Compute arcsine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def arccos(x):
        """Compute arccosine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def arctan(x):
        """Compute arctangent of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def sinh(x):
        """Compute sinh of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError
    
    @staticmethod
    def cosh(x):
        """Compute cosh of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def tanh(x):
        """Compute tanh of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError

    @staticmethod
    def log(x, base = np.exp(1)):
        """Compute log base `base` of `x` in a function. Output is not accessible unless `AD.eval(x)` is called."""
        raise NotImplementedError
    
    @staticmethod
    def exp(x, base=np.exp(1)):
        """Compute exp of `x` or `base` to the power of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.
        """
        raise NotImplementedError

    @staticmethod    
    def logistic(x, k=1, L=1, x_0=0):
        """Compute logistic function applied to `x`. Default settings are `k=1, L=1, x_0=0`
        Output is not accessible unless `AD.eval(x)` is called.
        """
        raise NotImplementedError