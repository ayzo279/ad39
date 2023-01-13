#!/usr/bin/env python3
# File       : test_reverse_core.py
# Description: Tests for core functions in RM class.

import pytest
import numpy as np
from ad39_package.reverse.reverse_core import RM
from ad39_package.reverse.reverse_helpers import *

class TestRM:
    """Test class for RM types"""

    # Define np.array
    a1 = np.array([1, 2.0])
    a2 = np.array([3, 1.])

    # Define scalar
    s1 = 2
    s2 = 1

    # Define cases
    ad1 = RM(lambda x: 2 * x ** 2)                              # scalar input, scalar output
    ad2 = RM(lambda x: 2 * x[0] ** 2 + x[1])                    # vector input, scalar output
    ad3 = RM(lambda x: np.array([2 * x ** 2, 3 * x]))           # scalar input, vector output
    ad4 = RM(lambda x: np.array([2 * x[0], 3 * x[1]]))          # vector input, vector output

    def test_init(self):

        # Define lambda function
        f = lambda x: x + 3

        # Initialize with that function
        ad = RM(f)

        # Check for identity
        assert ad.f is f

        # Check n_args (probably redundant after checking identity)
        assert ad._nargs == f.__code__.co_argcount

        with pytest.raises(AssertionError):
            f = lambda x, y: x + y
            ad = RM(f)
    
    def test_eval(self):

        # Check cases
        assert self.ad1.eval(self.s1) == 8
        assert self.ad2.eval(self.a1) == 4
        assert self.ad3.eval(self.s1)[0] == 8
        assert self.ad3.eval(self.s1)[1] == 6
        assert self.ad4.eval(self.a1)[0] == 2
        assert self.ad4.eval(self.a1)[1] == 6

        # Test for exception when called with unsupported type
        with pytest.raises(Exception):
            self.ad1.eval('some_test_string')

    def test_grad(self):

        # Check cases
        assert self.ad1.grad(self.s1) == 8
        assert (self.ad2.grad(self.a1) == np.array([4, 1])).all()
        assert (self.ad3.grad(self.s1) == np.array([8, 3])).all()
        assert (self.ad4.grad(self.a1)[0] == np.array([2., 0.])).all()
        assert (self.ad4.grad(self.a1)[1] == np.array([0., 3.])).all()
    
    def test_jacobian(self):
        
        # Check using reverse mode
        assert self.ad1.jacobian(self.s1) == 8
        assert self.ad2.jacobian(self.a1)[0] == 4
        assert self.ad2.jacobian(self.a1)[1] == 1
        assert (self.ad3.jacobian(self.s1)[0] == np.array([8])).all()
        assert (self.ad3.jacobian(self.s1)[1] == np.array([3])).all()
        assert self.ad4.jacobian(self.a1)[0, 0] == 2
        assert self.ad4.jacobian(self.a1)[0, 1] == 0
        assert self.ad4.jacobian(self.a1)[1, 0] == 0
        assert self.ad4.jacobian(self.a1)[1, 1] == 3

    def test_valid_seed(self):
        seed = [1, 1]
        with pytest.raises(Exception):
            self.ad1._valid_seed(seed, 2)

        seed = [2, 1]
        with pytest.raises(Exception):
            self.ad1._valid_seed(seed, 2)
    
    def test_example_rm(self):

        RMtest = RM(lambda x: 4 * RM.sin(x) ** 2)
        # Evaluate the function at x = pi/4 by calling the `eval(x)` method with argument pi/4
        result = RMtest.eval(np.pi/4)
        assert pytest.approx(result, 1e-10) == 2

        # Compute the derivative evaluated at pi/4 by calling the `forward_mode(x)` method with argument pi/4
        deriv = RMtest.grad(np.pi/4)
        assert pytest.approx(deriv, 1e-10) == 4

        # Compute the Jacobian evaluated at 2 by calling the `jacobian(x)` method with argument pi/4
        jacob = RMtest.jacobian(np.pi/4)
        assert pytest.approx(jacob, 1e-10) == 4