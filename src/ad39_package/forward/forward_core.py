#!/usr/bin/env python3
# File       : forward_core.py
# Description: Module that provides core functions for forward mode automatic differentiation (AD)

from ..core import AD
from ad39_package.forward.forward_helpers import *

class FM(AD):

    def _valid_seed(self, seed, dim):
        """This private method checks whether a seed vector is a valid one-hot encoded vector"""
        assert len(seed) == dim, f"Provided {len(seed)} elements in seed vector but {dim} were expected"
        
        one_found = False
        for s in seed:
            if s == 1:
                if one_found:
                    raise Exception(f"Seed vector {seed} is not valid, must be one-hot encoded")
                else:
                    one_found = True
            elif s == 0:
                pass
            else:
                raise Exception(f"Seed vector {seed} is not valid, must be one-hot encoded")
        
        assert one_found, f"Seed vector {seed} is not valid, must be one-hot encoded"

    def _apply_f(self, dual_inputs):
        """This private method applies the function `self.f` given some multiple vector and/or scalar dual number inputs `dual_inputs`"""
        try:
            # For functions with no vector subscripting
            # e.g. lambda x, y: x + y
            res = np.vectorize(self.f)(*dual_inputs)
        except TypeError:
            # For functions of vectors and scalars
            # e.g. lambda x, y: np.array([x[0] + y, x[1] - y])
            for i, d in enumerate(dual_inputs):
                if len(d) == 1:
                    dual_inputs[i] = d[0]
            res = self.f(*dual_inputs)
        
        return res


    def _dual_compute(self, inputs, part="real", seed=None):
        """This private method is the helper function to both `eval` and `forward_mode` handling all that is needed with dual numbers"""
        if isinstance(inputs, DualNumber._supported_scalars):
            assert 1 == self._nargs, f"Provided 1 argument to function but {self._nargs} were expected"

            # Make inputs dual number object
            dual_inputs = DualNumber(inputs)

            if part == 'dual': 
                if seed is None: seed = 1
                else: self._valid_seed(seed, 1)
            
            # Evaluate function at given inputs
            res = self.f(dual_inputs)

        elif isinstance(inputs, np.ndarray):
            assert 1 == self._nargs, f'Provided 1 argument to function but {self._nargs} were expected'

            # Make inputs dual number objects
            dual_inputs = np.array(DualNumber._from_list(inputs))

            if part == 'dual': 
                if seed is None: seed = [1 if i == 0 else 0 for i in range(len(inputs))]
                else: self._valid_seed(seed, len(inputs))

                for i, d in enumerate(dual_inputs):
                    d.dual = seed[i]
            
            # Evaluate function at given inputs
            res = self.f(dual_inputs)

        elif isinstance(inputs, tuple):
            assert len(inputs) == self._nargs, f"Provided {len(inputs)} arguments to function but {self._nargs} were expected"

            # Make inputs dual number object
            lst = []
            for inp in inputs:
                if isinstance(inp, DualNumber._supported_scalars):
                    lst.append(np.array([inp]))
                else:
                    lst.append(inp)
            if len(inputs) == len(lst):
                inputs = lst
            
            dual_inputs = [DualNumber._from_list(i) for i in inputs]

            if part == 'dual':
                n_elts = sum(len(i) for i in dual_inputs)

                if seed is None: seed = [1 if i == 0 else 0 for i in range(n_elts)]
                else: self._valid_seed(seed, n_elts)

                counter = 0
                for i, d in enumerate(dual_inputs):
                    if isinstance(d, list):
                            for di in d:
                                if seed[counter] == 0:
                                    di.dual = 0
                                counter+= 1
            
            # Evaluate function at given inputs
            res = self._apply_f(dual_inputs)

        else:
            raise TypeError(f"Incompatible input. Must be np.array, int or float but was of type: {type(inputs)}")

        # Check if result is iterable (i.e. an array)
        if part == "real":
            if isinstance(res, np.ndarray):
                if isinstance(res[0], np.ndarray):
                    return np.array([d.real for d in res[0]])
                else:
                    try:
                        return np.array([d.real for d in res])
                    except:
                        return float(res[0].real)
            else:
                return float(res.real)
        else: 
            if isinstance(res, np.ndarray):
                if isinstance(res[0], np.ndarray):
                    return np.array([d.dual for d in res[0]])
                else:
                    try:
                        return np.array([d.dual for d in res])
                    except:
                        return float(res[0].dual)
            else:
                return float(res.dual)
    

    def eval(self, inputs):
        """Evaluate the value of `self.f` at `inputs`.

        Parameters
        ----------
        inputs : int, float, np.ndarray, tuple
            Inputs at which to evaluate function.

        Returns
        -------
        res : float, numpy.ndarray
            Result of `self.f` evaluated at `inputs`.
        
        """
        return self._dual_compute(inputs, part="real")


    def grad(self, inputs, seed=None):
        """Compute the derivative of `self.f` evaluated at `inputs` using forward mode AD.

        Parameters
        ----------
        inputs : int, float, np.ndarray, tuple
            Inputs at which to evaluate derivative.
        
        seed: tuple, list, np.ndarray (optional)
            Seed vector representing variable for which to calculate derivative.
            Default is to calculate derivative of first variable inputted
            (either a scalar or the first element of a vector).

        Returns
        -------
        deriv : float, numpy.ndarray
            Derivative of `self.f` evaluated at `x`.

        """
        return self._dual_compute(inputs, part="dual", seed=seed)

    
    def jacobian(self, inputs):
        """Compute the Jacobian of `self.f` evaluated at `inputs` using forward mode AD.

        Parameters
        ----------
        inputs : int, float, np.ndarray, tuple
            Inputs at which to evaluate Jacobian.

        mode: "forward", "f", "reverse", "r" (optional)
            Specification of whether to use forward ot reverse
            mode to compute the Jacobian. Note that inputs of type 
            tuple will not work with `mode="reverse"` or `mode="r"`.

        Returns
        -------
        jacobian : numpy.ndarray
            Jacobian of `self.f` evaluated at `inputs`.
        """
        if isinstance(inputs, tuple):
            assert len(inputs) == self._nargs, f"Provided {len(inputs)} arguments to function but {self._nargs} were expected"

            # Make inputs dual number object
            lst = []
            for inp in inputs:
                if isinstance(inp, DualNumber._supported_scalars):
                    lst.append(np.array([inp]))
                else:
                    lst.append(inp)
            if len(inputs) == len(lst):
                inps = lst
            
            dual_inputs = [DualNumber._from_list(i) for i in inps]

            n_elts = sum(len(i) for i in dual_inputs)
        else:
            try:
                n_elts = len(inputs)
            except TypeError:
                n_elts = 1
        
        jacob = []

        for i in range(n_elts):
            seed = np.zeros(n_elts)
            seed[i] = 1
            jacob.append(self.grad(inputs, seed))
        
        return np.array(jacob).T.astype(float)
    

    """The following functions extend the capabilities of the DualNumbers used to compute and are available to the user for use within functions."""
    # Unary operations
    @staticmethod
    def sqrt(x):
        """Compute the positive square root of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to square root function. Must be positive.

        Example
        -------
        ad = FM(lambda x : FM.sqrt(x))
        ad.eval(4) # equals 2
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.sqrt(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.sqrt(x.real), x.dual*0.5 * (x.real ** -0.5))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def sin(x):
        """Compute sine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to sine function.

        Example
        -------
        ad = FM(lambda x : FM.sin(x))
        ad.eval(math.pi) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.sin(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.sin(x.real), x.dual * np.cos(x.real))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def cos(x):
        """Compute cosine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to cosine function.

        Example
        -------
        ad = FM(lambda x : FM.cos(x))
        ad.eval(math.pi) # equals -1
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.cos(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.cos(x.real), - x.dual * np.sin(x.real))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def tan(x):
        """Compute tangent of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to tangent function.

        Example
        -------
        ad = FM(lambda x : FM.tan(x))
        ad.eval(math.pi) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.tan(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.tan(x.real), x.dual * 1 / (np.cos(x.real) ** 2))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def arcsin(x):
        r"""Compute arcsine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to arcsine function. $x \in [-1, 1]$

        Example
        -------
        ad = FM(lambda x : FM.arcsin(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.arcsin(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.arcsin(x.real), x.dual * 1 / np.sqrt(1 - (x.real ** 2)))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def arccos(x):
        r"""Compute arccosine of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to arcosine function. $x \in [-1, 1]$

        Example
        -------
        ad = FM(lambda x : FM.arccos(x))
        ad.eval(1) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.arccos(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.arccos(x.real), - x.dual * 1 / np.sqrt(1 - (x.real ** 2)))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def arctan(x):
        r"""Compute arctangent of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to arctangent function. $x \in [-\pi / 2, \pi / 2]$

        Example
        -------
        ad = FM(lambda x : FM.arctan(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.arctan(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.arctan(x.real), x.dual / (1 + (x.real ** 2)))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def sinh(x):
        """Compute sinh of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to sinh function.

        Example
        -------
        ad = FM(lambda x : FM.sinh(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.sinh(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.sinh(x.real), x.dual * np.cosh(x.real))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def cosh(x):
        """Compute cosh of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to cosh function.

        Example
        -------
        ad = FM(lambda x : FM.cosh(x))
        ad.eval(0) # equals 1
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.cosh(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.cosh(x.real), x.dual * np.sinh(x.real))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def tanh(x):
        """Compute tanh of `x` in a function.
            Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to tanh function.

        Example
        -------
        ad = FM(lambda x : FM.tanh(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(np.tanh(x))

        elif isinstance(x, DualNumber):
            return DualNumber(np.tanh(x.real), x.dual * 1/(np.cosh(x.real) ** 2))
        else:
            raise TypeError(f'Unsupported type: {type(x)}')

    # Other functions
    @staticmethod
    def log(x, base = np.exp(1)):
        """Compute log base `base` of `x` in a function. Natural log / `base` is np.exp(1) by default.
            Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to log function.
        
        base : int, float (optional)
            Base of log function. Greater than 0. np.exp(1) by default.

        Examples
        -------
        ad = FM(lambda x : FM.log(x))
        ad.eval(np.exp(1)) # equals 1

        ad = FM(lambda x : FM.log(x, 10))
        ad.eval(100) # equals 2
        """
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(float(np.log(x)/np.log(base)))

        elif isinstance(x, DualNumber):
            return DualNumber(float(np.log(x.real)/np.log(base)), x.dual*(1/(np.log(base)*x.real)))

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def exp(x, base=np.exp(1)):
        """Compute exp of `x` or `base` to the power of `x` in a function. `base` is np.exp(1) by default.
            Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to exp function - known as the exponent.
        
        base : int, float (optional)
            Base of exp function. np.exp(1) by default.

        Examples
        -------
        ad = FM(lambda x : FM.exp(x))
        ad.eval(1) # equals np.exp(1)

        ad = FM(lambda x : FM.exp(x, 10))
        ad.eval(1) # equals 10
        """
        if isinstance(x, DualNumber._supported_scalars):
            if base == np.exp(1):
                return DualNumber(np.exp(x))
            else:
                return DualNumber(base ** x)
        if isinstance(x, DualNumber):
            if base == np.exp(1):
                return DualNumber(np.exp(x.real), x.dual * np.exp(x.real))
            else:
                if isinstance(base, DualNumber._supported_scalars):
                    return DualNumber(base ** x.real, x.dual * base ** x.real * np.log(base))
                else:
                    raise TypeError(f'Unsupported type: {type(base)}')
        else:
            raise TypeError(f'Unsupported type: {type(x)}')
          
    @staticmethod          
    def logistic(x, k=1, L=1, x_0=0):
        """Compute logistic function applied to `x`. Default settings are `k=1, L=1, x_0=0`
            Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to logistic function.
        
        k : int, float (optional)
            Logistic growth rate. 1 by default.
        
        L : int, float (optional)
            Maximum value. 1 by default.
        
        x_0 : int, float (optional)
            x value of sigmoid midpoint. 0 by default.

        Example
        -------
        ad = FM(lambda x : FM.logistic(x))
        ad.eval(1) # equals np.exp(1) / (1 + np.exp(1))
        """
        def f(arg):
            return L / (1 + np.exp(-k*(arg-x_0)))
        if isinstance(x, DualNumber._supported_scalars):
            return DualNumber(f(x))
        elif isinstance(x, DualNumber):
            return DualNumber(f(x.real), x.dual * f(x.real) * (1-f(x.real)))
        else:
            raise TypeError(f'Unsupported type: {type(x)}')