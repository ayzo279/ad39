import pytest
from ad39_package.core import *


f = lambda x, y: x + y

# Initialize with that function
ad = AD(f)

# Define x
x = 5        

class TestAD:

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_eval(self):
        ad.eval(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_grad(self):
        ad.grad(x)
    
    @pytest.mark.xfail(raises=NotImplementedError)
    def test_jacobian(self):
        ad.jacobian(x)

    @pytest.mark.xfail(raises=NotImplementedError) 
    def test_sqrt(self):
        AD.sqrt(x)
    
    @pytest.mark.xfail(raises=NotImplementedError)
    def test_sin(self):
        AD.sin(x)

    @pytest.mark.xfail(raises=NotImplementedError)  
    def test_cos(self):
        AD.cos(x)

    @pytest.mark.xfail(raises=NotImplementedError)  
    def test_tan(self):
        AD.cos(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_arcsin(self):
        AD.arcsin(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_arccos(self):
        AD.arccos(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_arctan(self):
        AD.arctan(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_sinh(self):
        AD.sinh(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_cosh(self):
        AD.cosh(x)
    
    @pytest.mark.xfail(raises=NotImplementedError)
    def test_tanh(self):
        AD.tanh(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_log(self):
        AD.log(x)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_exp(self):
        AD.exp(x)

    @pytest.mark.xfail(raises=NotImplementedError)             
    def logistic(self):
        AD.logistic(x)
