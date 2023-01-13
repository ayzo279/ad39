#!/usr/bin/env python3
# File       : test_reverse_helpers.py
# Description: Tests for helper functions provided by Node class and supporting functions.

import pytest
import numpy as np
import math

# Import names to test (this assumes we have added ../src/ad39_package to our pythonpath)
from ad39_package.reverse.reverse_helpers import *
from ad39_package.reverse.reverse_core import RM

class TestNodes:
    """Test class for node types"""

    val1 = 3
    val2 = 9

    def test_init(self):
        '''Test instantiation for a node object'''

        # Create an instance of Node
        n = Node(self.val1)

        assert n.val == 3
        assert n.partials == [1]
        assert n.children == []
        assert n.adjoints == 0
        
    def test_repr(self):
        '''Test repr for a dual number object'''

        # Get dual and real parts
        n = Node(self.val1)
        assert f"{type(n).__name__} object\n\
                 - val={n.val}\n\
                 - partials={n.partials}\n\
                 - adjoints={n.adjoints}\n\
                 - num. children={len([])}\n\
                 - op={n.op}" == f"Node object\n\
                 - val={3}\n\
                 - partials={[1]}\n\
                 - adjoints={0}\n\
                 - num. children={0}\n\
                 - op=None"

    def test_supported_scalars(self):
        '''Test supported scalars decorator for a node object'''

        # Create an instance of Node
        n = Node(self.val1)

        assert n.supported_scalars == Node._supported_scalars

    def test_add(self):
        '''Test addition for node objects'''

        # Add two nodes
        n1 = Node(self.val1)
        n2 = Node(self.val2)
        n = n1 + n2

        assert n.val == n1.val + n2.val
        assert n.partials == [1, 1]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)
        assert n2.children[0] == (1, n)

        # Add int or float to node
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = n1 + other

            assert n.val == n1.val + other
            assert n.partials == [1]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)

        # Test type errors
        with pytest.raises(TypeError):
            # Strings
            n + '1'
        with pytest.raises(Exception):
            '1' + n
        with pytest.raises(Exception):
            # Lists
            n + []
        with pytest.raises(Exception):
            [] + n

    def test_sub(self):
        '''Test subtraction for node objects'''

        # Subtract two nodes
        n1 = Node(self.val1)
        n2 = Node(self.val2)
        n = n1 - n2

        assert n.val == n1.val - n2.val
        assert n.partials == [1, -1]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)
        assert n2.children[0] == (1, n)

        # Subtract int or float from node
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = n1 - other

            assert n.val == n1.val - other
            assert n.partials == [1]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)

        # Test type errors
        with pytest.raises(TypeError):
            # Strings
            n - '1'
        with pytest.raises(Exception):
            '1' - n
        with pytest.raises(Exception):
            # Lists
            n - []
        with pytest.raises(Exception):
            [] - n

    def test_mul(self):
        '''Test multiplication for node objects'''

        # Multiply two nodes
        n1 = Node(self.val1)
        n2 = Node(self.val2)
        n = n1 * n2

        assert n.val == n1.val * n2.val
        assert n.partials == [n2.val, n1.val]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)
        assert n2.children[0] == (1, n)

        # Multiplify node with int or float
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = n1 * other

            assert n.val == n1.val * other
            assert n.partials == [other]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)

        # Test type errors
        with pytest.raises(TypeError):
            # Strings
            n * '1'
        with pytest.raises(Exception):
            '1' * n
        with pytest.raises(Exception):
            # Lists
            n * []
        with pytest.raises(Exception):
            [] * n

    def test_truediv(self):
        '''Test division for node objects'''

        # Divide two nodes
        n1 = Node(self.val1)
        n2 = Node(self.val2)
        n = n1 / n2

        assert n.val == n1.val / n2.val
        assert n.partials == [1/n2.val, -n1.val/n2.val**2]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)
        assert n2.children[0] == (1, n)

        # Divide node with int or float
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = n1 / other

            assert n.val == n1.val / other
            assert n.partials == [1/other]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)

        # Test type errors
        with pytest.raises(TypeError):
            # Strings
            n / '1'
        with pytest.raises(Exception):
            '1' / n
        with pytest.raises(Exception):
            # Lists
            n / []
        with pytest.raises(Exception):
            [] / n

    def test_radd(self):
        '''Test reverse addition for node objects'''

        # Add node to int or float
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = other + n1

            assert n.val == n1.val + other
            assert n.partials == [1]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)
    
    def test_rmul(self):
        '''Test reverse multiplication for node objects'''


        # Multiply int or float with node
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = other * n1

            assert n.val == n1.val * other
            assert n.partials == [other]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)

    def test_rsub(self):
        '''Test reverse subtraction for node objects'''

        # Subtract node from int or float
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = other - n1

            assert n.val == other - n1.val
            assert n.partials == [-1]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)

    def test_rtruediv(self):
        '''Test reverse division for node objects'''

        # Divide int or float with node
        for other in (1, 2.0):
            n1 = Node(self.val1)
            n = other / n1

            assert n.val == other / n1.val
            assert n.partials == [-other/n1.val**2]
            assert n.adjoints == 0
            assert n1.children[0] == (0, n)
    
    def test_neg(self):
        '''Test negation for node objects'''

        # Create instance
        n = Node(self.val1)
        assert n.val == self.val1
        assert n.partials == [1]
        assert n.adjoints == 0
        assert n.children == []

        # Negate instance
        n1 = -n
        assert n1.val == -self.val1
        assert n1.partials == [-1]
        assert n1.adjoints == 0
        assert n.children[0] == (0, n1)

    def test_log(self):
        '''Test log for node objects '''

        # Apply log on scalar 
        assert pytest.approx(RM.log(self.val1), 10) == np.log(self.val1)

        # Create instance
        n1 = Node(self.val1)
        n = RM.log(n1)
        n2 = Node(self.val2)
        n10 = RM.log(n2, 10)

        # Define different base 
        base = 10

        assert pytest.approx(n.val, 10) == np.log(n1.val)
        assert pytest.approx(n.partials[0], 10) == 1/n1.val
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        assert pytest.approx(n10.val, 10) == np.log10(n2.val)
        assert pytest.approx(n10.partials[0], 10) == 1/(np.log(10) * n2.val)
        assert n10.adjoints == 0
        assert (n2.children[0]) == (0, n10)

        with pytest.raises(TypeError):
            # Strings
            RM.log('1')
        with pytest.raises(Exception):
            RM.log(n, '1')
        with pytest.raises(Exception):
            # Lists
            RM.log([])
        with pytest.raises(Exception):
            RM.log(n, [])

    def test_sqrt(self):
        '''Test square root for node objects '''

        # Apply square root function on scalar
        assert RM.sqrt(self.val1) == np.sqrt(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.sqrt(n1)

        assert n.val == np.sqrt(n1.val)
        assert len(n.partials) == 1
        assert pytest.approx(n.partials[0], 10) == 1/(2*np.sqrt(n1.val))
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.sqrt('s')

            # Lists
            RM.sqrt([])

    def test_sin(self):
        '''Test sine for node objects '''

        # Apply sine function on scalar
        assert RM.sin(self.val1) == np.sin(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.sin(n1)

        assert n.val == np.sin(n1.val)
        assert n.partials == [np.cos(n1.val)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.sin('s')
        with pytest.raises(TypeError):
            # Lists
            RM.sin([])

    def test_cos(self):
        '''Test cosine for node objects '''

        # Apply cosine function on scalar
        assert RM.cos(self.val1) == np.cos(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.cos(n1)

        assert n.val == np.cos(n1.val)
        assert n.partials == [-np.sin(n1.val)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.cos('s')
        with pytest.raises(TypeError):
            # Lists
            RM.cos([])

    def test_tan(self):
        '''Test tangent for node objects '''

        # Apply tangent function on scalar
        assert RM.tan(self.val1) == np.tan(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.tan(n1)

        assert n.val == np.tan(n1.val)
        assert n.partials == [1/np.cos(n1.val) ** 2]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.tan('s')
        with pytest.raises(TypeError):
            # Lists
            RM.tan([])

    def test_arcsin(self):
        '''Test inverse sine for node objects '''

        # Apply inverse sine function on scalar
        assert RM.arcsin(0.5) == np.arcsin(0.5)

        # Create instances
        n1 = Node(0.5)
        n = RM.arcsin(n1)

        assert n.val == np.arcsin(n1.val)
        assert n.partials == [1/np.sqrt(1-n1.val**2)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.arcsin('s')
        with pytest.raises(TypeError):
            # Lists
            RM.arcsin([])

    def test_arccos(self):
        '''Test inverse cosine for node objects '''

        # Apply inverse cosine function on scalar
        assert RM.arccos(0.5) == np.arccos(0.5)

        # Create instances
        n1 = Node(0.5)
        n = RM.arccos(n1)

        assert n.val == np.arccos(n1.val)
        assert n.partials == [-1/np.sqrt(1-n1.val**2)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.arccos('s')
        with pytest.raises(TypeError):
            # Lists
            RM.arccos([])

    def test_arctan(self):
        '''Test inverse tangent for node objects '''

        # Apply inverse tangent function on scalar
        assert RM.arctan(self.val1) == np.arctan(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.arctan(n1)

        assert n.val == np.arctan(n1.val)
        assert n.partials == [1/(1+n1.val**2)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.arctan('s')
        with pytest.raises(TypeError):
            # Lists
            RM.arctan([])
    
    def test_exp(self):
        '''Test exp for node objects '''

        # Apply exponential function on scalars
        assert RM.exp(self.val1) == np.exp(self.val1)
        assert RM.exp(self.val1, base=2) == 2 ** self.val1

        # Create instances (without base)
        n1 = Node(self.val1)
        n = RM.exp(n1)
        assert n.val == np.exp(n1.val)
        assert n.partials == [np.exp(n1.val)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        # Create instances (with base)
        n1 = Node(self.val1)
        n = RM.exp(n1, base=2)

        assert n.val == 2 ** n1.val
        assert n.partials == [np.log(2) * 2 ** n1.val]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.exp('s')
        with pytest.raises(TypeError):
            RM.exp(2, base='s')
        with pytest.raises(TypeError):
            # Lists
            RM.exp([])
        with pytest.raises(TypeError):
            RM.exp(2, base=[])
    
    def test_sinh(self):
        '''Test hyperbolic sine for node objects '''

        # Apply hyperbolic sine function on scalar
        assert RM.sinh(self.val1) == np.sinh(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.sinh(n1)

        assert n.val == np.sinh(n1.val)
        assert n.partials == [np.cosh(n1.val)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.sinh('s')
        with pytest.raises(TypeError):
            # Lists
            RM.sinh([])
    
    def test_cosh(self):
        '''Test hyperbolic cosine for node objects '''

        # Apply hyperbolic cosine function on scalar
        assert RM.cosh(self.val1) == np.cosh(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.cosh(n1)

        assert n.val == np.cosh(n1.val)
        assert n.partials == [np.sinh(n1.val)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.cosh('s')
        with pytest.raises(TypeError):
            # Lists
            RM.cosh([])
        
    def test_tanh(self):
        '''Test hyperbolic tangent for node objects '''

        # Apply hyperbolic cosine function on scalar
        assert RM.tanh(self.val1) == np.tanh(self.val1)

        # Create instances
        n1 = Node(self.val1)
        n = RM.tanh(n1)

        assert n.val == np.tanh(n1.val)
        assert n.partials == [1/np.cosh(n1.val)**2]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.tanh('s')
        with pytest.raises(TypeError):
            # Lists
            RM.tanh([])
    
    def test_pow(self):
        '''Test power for node objects'''

        n1 = Node(self.val1)
        n2 = Node(self.val2)
        n3 = Node(5)
        ns = n3 ** 3
        n = n1 ** n2

        assert ns.val == n3.val ** 3
        assert ns.partials == [3 * n3.val ** 2]
        assert ns.adjoints == 0
        assert n3.children[0] == (0, ns)

        assert n.val == n1.val ** n2.val
        assert n.partials == [n2.val * n1.val ** (n2.val - 1), np.log(n1.val) * n1.val ** n2.val]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)
        assert n2.children[0] == (1, n)

        with pytest.raises(TypeError):
            # Strings
            n1 ** 's'
        with pytest.raises(TypeError):
            # Lists
            [] ** n1

    def test_logistic(self):
        '''Test logistic for dual number objects (using default arguments)'''

        assert RM.logistic(self.val1) == 1/(1+np.exp(-self.val1))

        # Create instances
        n1 = Node(self.val1)
        n = RM.logistic(n1)

        assert n.val == 1/(1+np.exp(-n1.val))
        assert n.partials == [n1.val * (1 - n1.val)]
        assert n.adjoints == 0
        assert n1.children[0] == (0, n)

        with pytest.raises(TypeError):
            # Strings
            RM.logistic('s')
        with pytest.raises(TypeError):
            # Lists
            RM.logistic([])
