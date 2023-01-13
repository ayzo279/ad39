#!/usr/bin/env python3
# File       : forward_helpers.py
# Description: Module that provides helper functions for forward mode automatic differentiation (AD)

import numpy as np


class DualNumber:
    """Dual number base class - facilitates forward mode AD. Hidden from the user."""

    _supported_scalars = (float, int)

    def __init__(self, real, dual=None):
        self.real = real

        if dual is not None:
            self.dual = dual
        elif isinstance(real, np.ndarray):
            self.dual = np.ones(len(real))
        else:
            self.dual = 1.0

    @classmethod
    def _from_list(cls, inputs):
        """Construct multiple dual numbers, returning a list of dual numbers"""
        if np.ndim(inputs) != 1:
            raise Exception(f"Provided {np.ndim(inputs)}-dimensional input array but must be 1-dimensional")
        else:
            return [cls(real = i) for i in inputs]
    
    @property
    def supported_scalars(self):
        """Get suppported scalars"""
        return self._supported_scalars

    def __repr__(self):
        class_name = type(self).__name__
        instance_state = (self.real, self.dual)

        return f"{class_name}({instance_state})"

    # Binary operations
    def __add__(self, other):
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real + other, self.dual)

        elif isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    def __sub__(self, other):
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real - other, self.dual)

        elif isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    def __mul__(self, other):
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real * other, self.dual * other)

        elif isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.dual * other.real + self.real * other.dual)

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real / other, self.dual / other)

        elif isinstance(other, DualNumber):
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual) / other.real**2)

        else:
            raise TypeError(f'Unsupported type: {type(other)}')
    
    def __pow__(self, other):
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real ** other, self.dual * other * self.real ** (other - 1))

        elif isinstance(other, DualNumber):
            return DualNumber(self.real ** other.real, self.real ** other.real * (other.dual * np.log(self.real) + self.dual*other.real/self.real))

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    # Reverse binary operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return DualNumber(other - self.real, -self.dual)

    def __rtruediv__(self, other):
        return DualNumber(other/self.real, -other * self.dual / self.real**2)
    
    # Unary operations
    def __neg__(self):
        return DualNumber(- self.real, - self.dual)