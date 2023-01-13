#!/usr/bin/env python3
# File       : test_forward_core.py
# Description: Tests for core functions in FM class.

import pytest
import numpy as np
from ad39_package.forward.forward_core import FM
from ad39_package.forward.forward_helpers import *

class TestFM:
    """Test class for FM"""

    # Define np.array
    a1 = np.array([1, 2.0])
    a2 = np.array([3, 1.])

    # Define scalar
    s1 = 2
    s2 = 1

    # Define cases
    ad1 = FM(lambda x: 2 * x ** 2)                              # scalar input, scalar output
    ad2 = FM(lambda x: 2 * x[0] ** 2 + x[1])                    # vector input, scalar output
    ad3 = FM(lambda x: np.array([2 * x ** 2, 3 * x]))           # scalar input, vector output
    ad4 = FM(lambda x: np.array([2 * x[0], 3 * x[1]]))          # vector input, vector output

    ad5 = FM(lambda x, y: x[0] + y[1])                          # multiple vector input, scalar output
    ad6 = FM(lambda x, y: x - y)                                # multiple scalar input, scalar output
    ad7 = FM(lambda x, y: x[0] + x[1] - y)                      # mixed scalar/vector input, scalar output
    ad8 = FM(lambda x, y: np.array([x[0] + x[1], y[0] - y[1]])) # multiple vector input, vector output
    ad9 = FM(lambda x, y: np.array([x, y]))                     # multiple scalar input, vector output
    ad10 = FM(lambda x, y: np.array([x[0] + y, x[1] - y]))      # mixed scalar/vector input, vector output

    def test_init(self):

        # Define lambda function
        f = lambda x, y: x + y

        # Initialize with that function
        ad = FM(f)

        # Check for identity
        assert ad.f is f

        # Check n_args (probably redundant after checking identity)
        assert ad._nargs == f.__code__.co_argcount
    
    def test_eval(self):

        # Check cases
        assert self.ad1.eval(self.s1) == 8
        assert self.ad2.eval(self.a1) == 4
        assert self.ad3.eval(self.s1)[0] == 8
        assert self.ad3.eval(self.s1)[1] == 6
        assert self.ad4.eval(self.a1)[0] == 2
        assert self.ad4.eval(self.a1)[1] == 6

        assert self.ad5.eval((self.a1, self.a2)) == 2 
        assert self.ad6.eval((self.s1, self.s2)) == 1
        assert self.ad7.eval((self.a1, self.s2)) == 2
        assert self.ad8.eval((self.a1, self.a2))[0] == 3
        assert self.ad8.eval((self.a1, self.a2))[1] == 2
        assert self.ad9.eval((self.s1, self.s2))[0] == 2
        assert self.ad9.eval((self.s1, self.s2))[1] == 1
        assert self.ad10.eval((self.a1, self.s2))[0] == 2
        assert self.ad10.eval((self.a1, self.s2))[1] == 1

        # Test for exception when called with unsupported type
        with pytest.raises(Exception):
            self.ad1.eval('some_test_string')
        
    
    def test_grad(self):

        # Check cases
        assert self.ad1.grad(self.s1) == 8
        assert self.ad2.grad(self.a1, seed = [1, 0]) == 4
        assert self.ad2.grad(self.a1, seed = [0, 1]) == 1
        assert self.ad3.grad(self.s1)[0] == 8
        assert self.ad3.grad(self.s1)[1] == 3
        assert self.ad4.grad(self.a1, seed = [1, 0])[0] == 2
        assert self.ad4.grad(self.a1, seed = [1, 0])[1] == 0
        assert self.ad4.grad(self.a1, seed = [0, 1])[0] == 0
        assert self.ad4.grad(self.a1, seed = [0, 1])[1] == 3

        assert self.ad5.grad((self.a1, self.a2), seed = [1, 0, 0, 0]) == 1
        assert self.ad5.grad((self.a1, self.a2), seed = [0, 1, 0, 0]) == 0
        assert self.ad5.grad((self.a1, self.a2), seed = [0, 0, 1, 0]) == 0
        assert self.ad5.grad((self.a1, self.a2), seed = [0, 0, 0, 1]) == 1

        assert self.ad6.grad((self.s1, self.s2), seed = [1, 0]) == 1
        assert self.ad6.grad((self.s1, self.s2), seed = [0, 1]) == -1

        assert self.ad7.grad((self.a1, self.s1), seed = [1, 0, 0]) == 1
        assert self.ad7.grad((self.a1, self.s1), seed = [0, 1, 0]) == 1
        assert self.ad7.grad((self.a1, self.s1), seed = [0, 0, 1]) == -1

        assert self.ad8.grad((self.a1, self.a2), seed = [1, 0, 0, 0])[0] == 1
        assert self.ad8.grad((self.a1, self.a2), seed = [0, 1, 0, 0])[0] == 1
        assert self.ad8.grad((self.a1, self.a2), seed = [0, 0, 1, 0])[0] == 0
        assert self.ad8.grad((self.a1, self.a2), seed = [0, 0, 0, 1])[0] == 0

        assert self.ad8.grad((self.a1, self.a2), seed = [1, 0, 0, 0])[1] == 0
        assert self.ad8.grad((self.a1, self.a2), seed = [0, 1, 0, 0])[1] == 0
        assert self.ad8.grad((self.a1, self.a2), seed = [0, 0, 1, 0])[1] == 1
        assert self.ad8.grad((self.a1, self.a2), seed = [0, 0, 0, 1])[1] == -1

        assert self.ad9.grad((self.s1, self.s2), seed = [1, 0])[0] == 1
        assert self.ad9.grad((self.s1, self.s2), seed = [1, 0])[1] == 0
        assert self.ad9.grad((self.s1, self.s2), seed = [0, 1])[0] == 0
        assert self.ad9.grad((self.s1, self.s2), seed = [0, 1])[1] == 1

        assert self.ad10.grad((self.a1, self.s2), seed = [1, 0, 0])[0] == 1
        assert self.ad10.grad((self.a1, self.s2), seed = [1, 0, 0])[1] == 0
        assert self.ad10.grad((self.a1, self.s2), seed = [0, 1, 0])[0] == 0
        assert self.ad10.grad((self.a1, self.s2), seed = [0, 1, 0])[1] == 1
        assert self.ad10.grad((self.a1, self.s2), seed = [0, 0, 1])[0] == 1
        assert self.ad10.grad((self.a1, self.s2), seed = [0, 0, 1])[1] == -1

    def test_jacobian(self):
        
        # Check using forward mode
        assert self.ad1.jacobian(self.s1) == 8
        assert self.ad2.jacobian(self.a1)[0] == 4
        assert self.ad2.jacobian(self.a1)[1] == 1
        assert self.ad3.jacobian(self.s1)[0] == 8
        assert self.ad3.jacobian(self.s1)[1] == 3
        assert self.ad4.jacobian(self.a1)[0, 0] == 2
        assert self.ad4.jacobian(self.a1)[0, 1] == 0
        assert self.ad4.jacobian(self.a1)[1, 0] == 0
        assert self.ad4.jacobian(self.a1)[1, 1] == 3

        assert self.ad5.jacobian((self.a1, self.a2))[0] == 1
        assert self.ad5.jacobian((self.a1, self.a2))[1] == 0

        assert self.ad6.jacobian((self.s1, self.s2))[0][0] == 1
        assert self.ad6.jacobian((self.s1, self.s2))[0][1] == -1

    def test_valid_seed(self):
        seed = [1, 1]
        with pytest.raises(Exception):
            self.ad1._valid_seed(seed, 2)

        seed = [2, 1]
        with pytest.raises(Exception):
            self.ad1._valid_seed(seed, 2)
    
    def test_example_fm(self):

        # Define target function
        f = lambda x: np.array([x[0] ** 2 + x[1] ** 2, x[0] ** 2 - x[1] ** 2])  

        # Instantiate FM object
        ad = FM(f)

        # Define input array
        x = np.array([2, 3])

        # Evaluate function at input
        result = ad.eval(x)

        assert result[0] == 13
        assert result[1] == -5

        # Evaluate derivative at input for both seed vectors
        D1 = ad.grad(x, seed = [1, 0])
        D2 = ad.grad(x, seed = [0, 1])

        assert D1[0] == 4
        assert D1[1] == 4
        assert D2[0] == 6
        assert D2[1] == -6

        # Evaluate the Jacobian at input
        J = ad.jacobian(x)
        assert J[0, 0] == 4
        assert J[0, 1] == 6
        assert J[1, 0] == 4
        assert J[1, 1] == -6