#!/usr/bin/env python3
# File       : test_forward_helpers.py
# Description: Tests for helper functions provided by DualNumber class and supporting functions.

import pytest
import numpy as np

# Import names to test (this assumes we have added ../src/ad39_package to our pythonpath)
from ad39_package.forward.forward_helpers import *
from ad39_package.forward.forward_core import FM

class TestDualNumbers:
    """Test class for dual number types"""

    # D1 = DualNumber(2, 2), D2 = DualNumber(3, 3)
    r1, r2 = 2, 3
    d1, d2 = 2, 3
    # D3 = DualNumber(0.5, 0.5) for inverse trig functions
    rr1 = 0.5
    dd1 = 0.5

    def test_init(self):
        '''Test instantiation for a dual number object'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create an instance of DualNumber
        d = DualNumber(R[0], D[0])

        assert d.real == R[0]
        assert d.dual == D[0]

    def test_repr(self):
        '''Test repr for a dual number object'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create an instance of DualNumber
        d = DualNumber(R[0], D[0])

        assert f"{type(d).__name__}({(d.real, d.dual)})" == f"DualNumber({(R[0], D[0])})"

    def test_supported_scalars(self):
        '''Test supported scalars decorator for a dual number object'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create an instance of DualNumber
        d = DualNumber(R[0], D[0])

        assert d.supported_scalars == DualNumber._supported_scalars

    def test_add(self):
        '''Test addition for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Add two dual numbers
        d = DualNumber(R[0], D[0]) + DualNumber(R[1], D[1])

        assert d.real == R[0] + R[1]
        assert d.dual == D[0] + D[1]

        # Add int or float to dual number
        for other in (1, 2.0):

            dn = d + other

            assert dn.real == d.real + other
            assert dn.dual == d.dual

        # Test type errors
        with pytest.raises(TypeError):
            d + '1'
        with pytest.raises(TypeError):
            '1' + d
        with pytest.raises(TypeError):
            d + []
        with pytest.raises(TypeError):
            [] + d

    def test_sub(self):
        '''Test substraction for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Substract two dual numbers
        d = DualNumber(R[0], D[0]) - DualNumber(R[1], D[1])

        assert d.real == R[0] - R[1]
        assert d.dual == D[0] - D[1]

        # Substract int or float from dual number
        for other in (1, 2.0):

            dn = d - other

            assert dn.real == d.real - other
            assert dn.dual == d.dual

        # Test type errors
        with pytest.raises(TypeError):
            d - '1'
        with pytest.raises(TypeError):
            '1' - d
        with pytest.raises(TypeError):
            d - []
        with pytest.raises(TypeError):
            [] - d

    def test_mul(self):
        '''Test multiplication for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Mul two dual numbers
        d = DualNumber(R[0], D[0]) * DualNumber(R[1], D[1])

        assert d.real == R[0] * R[1]
        assert d.dual == D[0] * R[1] + R[0] * D[1]

        # Mul int or float with dual number
        for other in (1, 2.0):

            dn = d * other

            assert dn.real == d.real * other
            assert dn.dual == d.dual * other

        with pytest.raises(TypeError):
            d * '1'
        with pytest.raises(TypeError):
            '1' * d
        with pytest.raises(TypeError):
            d * []
        with pytest.raises(TypeError):
            [] * d

    def test_truediv(self):
        '''Test division for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Divide two dual numbers
        d = DualNumber(R[0], D[0]) / DualNumber(R[1], D[1])

        assert d.real == R[0] / R[1]
        assert d.dual == (D[0] * R[1] - R[0] * D[1])/R[1]**2

        # Mul int or float with dual number
        for other in (1, 2.0):

            dn = d / other

            assert dn.real == d.real / other
            assert dn.dual == d.dual / other

        with pytest.raises(TypeError):
            d / '1'
        with pytest.raises(TypeError):
            '1' / d
        with pytest.raises(TypeError):
            d / []
        with pytest.raises(TypeError):
            [] / d

    def test_radd(self):
        '''Test reverse addition for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instance
        d = DualNumber(R[0], D[0])

        # Add dual number to int or float
        for other in (1, 2.0):

            dn = other + d

            assert dn.real == d.real + other
            assert dn.dual == d.dual
    
    def test_rmul(self):
        '''Test reverse multiplication for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instance
        d = DualNumber(R[0], D[0])

        # Mul dual number to int or float
        for other in (1, 2.0):

            dn = other * d

            assert dn.real == d.real * other
            assert dn.dual == d.dual * other

    def test_rsub(self):
        '''Test reverse substraction for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instance
        d = DualNumber(R[0], D[0])

        # Add dual number to int or float
        for other in (1, 2.0):

            dn = other - d

            assert dn.real == other - d.real
            assert dn.dual == - d.dual

    def test_rtruediv(self):
        '''Test reverse division for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instance
        d = DualNumber(R[0], D[0])

        # Add dual number to int or float
        for other in (1, 2.0):

            dn = other / d

            assert dn.real == other / d.real
            assert dn.dual == - other * d.dual/d.real**2
    
    def test_neg(self):
        '''Test negation for dual number objects'''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instance
        d = DualNumber(R[0], D[0])
        assert d.real == 2
        assert d.dual == 2

        # Negate instance
        d = -d
        assert d.real == -2
        assert d.dual == -2

    def test_log(self):
        '''Test log for dual number objects '''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instances
        s = FM.log(R[0])
        d = FM.log(DualNumber(R[0], D[0]))

        # Define base 
        base = np.exp(1)

        assert s.real == float(np.log(R[0])/np.log(base))
        assert s.dual == 1
        assert d.real == FM.log(R[0], base = base).real
        assert d.dual == D[0] * (1/(np.log(base)*R[0]))

        with pytest.raises(TypeError):
            FM.log('1')
        with pytest.raises(TypeError):
            FM.log(d, '1')
        with pytest.raises(TypeError):
            FM.log([])
        with pytest.raises(TypeError):
            FM.log(d, [])

    def test_sqrt(self):
        '''Test square root for dual number objects '''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instances
        s = FM.sqrt(R[0])
        d = FM.sqrt(DualNumber(R[0], D[0]))

        assert s.real == np.sqrt(R[0])
        assert s.dual == 1
        assert d.real == np.sqrt(R[0])
        assert pytest.approx(d.dual, 10) == D[0]/(2*np.sqrt(R[0]))

        with pytest.raises(TypeError):
            FM.sqrt('s')
        with pytest.raises(TypeError):
            FM.sqrt([])

    def test_sin(self):
        '''Test sine for dual number objects '''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instances
        s = FM.sin(R[0])
        d = FM.sin(DualNumber(R[0], D[0]))

        assert s.real == np.sin(R[0])
        assert s.dual == 1
        assert d.real == np.sin(R[0])
        assert d.dual == D[0] * np.cos(R[0])

        with pytest.raises(TypeError):
            # Strings
            FM.sin('s')

            # Lists
            FM.sin([])

    def test_cos(self):
        '''Test cosine for dual number objects '''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instances
        s = FM.cos(R[0])
        d = FM.cos(DualNumber(R[0], D[0]))

        assert s.real == np.cos(R[0])
        assert s.dual == 1
        assert d.real == np.cos(R[0])
        assert d.dual == D[0] * -np.sin(R[0])

        with pytest.raises(TypeError):
            FM.cos('s')
        with pytest.raises(TypeError):
            FM.cos([])

    def test_tan(self):
        '''Test tangent for dual number objects '''

        # Get dual and real parts
        R, D = (self.r1, self.r2), (self.d1, self.d2)

        # Create instances
        s = FM.tan(R[0])
        d = FM.tan(DualNumber(R[0], D[0]))

        assert s.real == np.tan(R[0])
        assert s.dual == 1
        assert d.real == np.tan(R[0])
        assert d.dual == D[0] / (np.cos(R[0]) ** 2)

        with pytest.raises(TypeError):
            FM.tan('s')
        with pytest.raises(TypeError):
            FM.tan([])

    def test_arcsin(self):
        '''Test arcsine for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.arcsin(R)
        d = FM.arcsin(DualNumber(R, D))

        assert s.real == np.arcsin(R)
        assert s.dual == 1
        assert d.real == np.arcsin(R)
        assert d.dual == D / np.sqrt(1-R**2)

        with pytest.raises(TypeError):
            FM.arcsin('s')
        with pytest.raises(TypeError):
            FM.arcsin([])

    def test_arccos(self):
        '''Test arccosine for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.arccos(R)
        d = FM.arccos(DualNumber(R, D))

        assert s.real == np.arccos(R)
        assert s.dual == 1
        assert d.real == np.arccos(R)
        assert d.dual == -D / np.sqrt(1-R**2)

        with pytest.raises(TypeError):
            FM.arccos('s')
        with pytest.raises(TypeError):
            FM.arccos([])

    def test_arctan(self):
        '''Test arctangent for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.arctan(R)
        d = FM.arctan(DualNumber(R, D))

        assert s.real == np.arctan(R)
        assert s.dual == 1
        assert d.real == np.arctan(R)
        assert d.dual == D / (1+R**2)
        with pytest.raises(TypeError):
            FM.arctan('s')
        with pytest.raises(TypeError):
            FM.arctan([])
    
    def test_exp(self):
        '''Test exp for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances (without base)
        s = FM.exp(R)
        d = FM.exp(DualNumber(R, D))

        assert s.real == np.exp(R)
        assert s.dual == 1
        assert d.real == np.exp(R)
        assert d.dual == D * np.exp(R)

        # Create instances (with base)
        s = FM.exp(R, base=2)
        d = FM.exp(DualNumber(R, D), base=2)

        assert s.real == 2 ** R
        assert s.dual == 1
        assert d.real == 2 ** R
        assert d.dual == D * 2 ** R * np.log(2)

        with pytest.raises(TypeError):
            FM.exp('s')
        with pytest.raises(TypeError):
            FM.exp(2, base='s')
        with pytest.raises(TypeError):
            FM.exp([])
        with pytest.raises(TypeError):
            FM.exp(2, base=[])
    
    def test_sinh(self):
        '''Test sinh for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.sinh(R)
        d = FM.sinh(DualNumber(R, D))

        assert s.real == np.sinh(R)
        assert s.dual == 1
        assert d.real == np.sinh(R)
        assert d.dual == D * np.cosh(R)
        with pytest.raises(TypeError):
            FM.sinh('s')
        with pytest.raises(TypeError):
            FM.sinh([])
    
    def test_cosh(self):
        '''Test cosh for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.cosh(R)
        d = FM.cosh(DualNumber(R, D))

        assert s.real == np.cosh(R)
        assert s.dual == 1
        assert d.real == np.cosh(R)
        assert d.dual == D * np.sinh(R)
        with pytest.raises(TypeError):
            FM.cosh('s')
        with pytest.raises(TypeError):
            FM.cosh([])
        
    def test_tanh(self):
        '''Test tanh for dual number objects '''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.tanh(R)
        d = FM.tanh(DualNumber(R, D))

        assert s.real == np.tanh(R)
        assert s.dual == 1
        assert d.real == np.tanh(R)
        assert pytest.approx(d.dual, 1e-10) == D * (1 - np.tanh(R) ** 2)
        with pytest.raises(TypeError):
            FM.tanh('s')
        with pytest.raises(TypeError):
            FM.tanh([])
    
    def test_pow(self):
        '''Test power for dual number objects'''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = DualNumber(R, D) ** 2
        d = DualNumber(R, D) ** DualNumber(R, D)

        assert s.real == R ** 2
        assert s.dual == D * 2 * R
        assert d.real == R ** R
        assert d.dual == R ** R * (D * np.log(R) + D)
        with pytest.raises(TypeError):
            DualNumber(R, D) ** 's'
        with pytest.raises(TypeError):
            [] ** DualNumber(R, D)

    def test_logistic(self):
        '''Test logistic for dual number objects (using default arguments)'''

        # Get dual and real parts
        R, D = self.rr1, self.dd1

        # Create instances
        s = FM.logistic(R)
        d = FM.logistic(DualNumber(R, D))

        assert s.real == 1/(1+np.exp(-R))
        assert s.dual == 1
        assert d.real == 1/(1+np.exp(-R))
        assert d.dual == D * d.real * (1 - d.real)
        with pytest.raises(TypeError):
            FM.logistic('s')
        with pytest.raises(TypeError):
            FM.logistic([])
