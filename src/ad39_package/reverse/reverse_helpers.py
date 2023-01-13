#!/usr/bin/env python3
# File       : reverse_helpers.py
# Description: Module that provides helper functions for reverse mode automatic differentiation (AD)

import numpy as np


class Node:
    _supported_scalars = (float, int)

    def __init__(self, val, partials = 1, children = None, adjoints = 0, op = None):

        self.val = val
        self.partials = [partials] if isinstance(partials, self._supported_scalars) else partials
        self.adjoints = adjoints
        self.children = children if children is not None else []
        self.op = op
    
    @classmethod
    def _from_list(cls, inputs):
        """Construct multiple dual numbers, returning a list of dual numbers"""
        if np.ndim(inputs) != 1:
            raise Exception(f"Provided {np.ndim(inputs)}-dimensional input array but must be 1-dimensional")
        else:
            return [cls(val = i) for i in inputs]
    
    @property
    def supported_scalars(self):
        """Get suppported scalars"""
        return self._supported_scalars
    
    def __repr__(self):
        class_name = type(self).__name__ 

        return f"{class_name} object\n\
                 - val={self.val}\n\
                 - partials={self.partials}\n\
                 - adjoints={self.adjoints}\n\
                 - num. children={len(self.children) if self.children is not None else 0}\n\
                 - op={self.op}"
    
    # Binary operations
    def __add__(self, other):
        if isinstance(other, self._supported_scalars):
            child = Node(self.val + other, op = f"(+ {other})")
            self.children.append((0, child))
            return child

        elif isinstance(other, Node):
            child = Node(self.val + other.val, [1, 1], op = "(+)")
            self.children.append((0, child))
            other.children.append((1, child))
            return child

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    def __sub__(self, other):
        if isinstance(other, self._supported_scalars):
            child = Node(self.val - other, op = f"(- {other})")
            self.children.append((0, child))
            return child

        elif isinstance(other, Node):
            child = Node(self.val - other.val, [1, -1], op = "(-)")
            self.children.append((0, child))
            other.children.append((1, child))
            return child

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    def __mul__(self, other):
        if isinstance(other, self._supported_scalars):
            child = Node(self.val * other, other, op = f"(* {other})")
            self.children.append((0, child))
            return child

        elif isinstance(other, Node):
            child = Node(self.val * other.val, [other.val, self.val], op = "(*)")
            self.children.append((0, child))
            other.children.append((1, child))
            return child

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, self._supported_scalars):
            child = Node(self.val / other, 1 / other, op = f"(/ {other})")
            self.children.append((0, child))
            return child

        elif isinstance(other, Node):
            child = Node(self.val / other.val, [1/other.val, -self.val/(other.val ** 2)], op = "(/)")
            self.children.append((0, child))
            other.children.append((1, child))
            return child

        else:
            raise TypeError(f'Unsupported type: {type(other)}')
    
    def __pow__(self, other):
        if isinstance(other, self._supported_scalars):
            child = Node(self.val ** other, [other * self.val ** (other - 1)], op = f"pow()")
            self.children.append((0, child))
            return child

        elif isinstance(other, Node):
            child = Node(self.val ** other.val, [other.val*self.val**(other.val - 1), np.log(self.val) * self.val ** other.val], op="pow($\cdot$)")
            self.children.append((0, child))
            other.children.append((1, child))
            return child

        else:
            raise TypeError(f'Unsupported type: {type(other)}')

    # Reverse binary operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        child = Node(other - self.val, -1, op ="(-)")
        self.children.append((0, child))
        return child

    def __rtruediv__(self, other):
        child = Node(other/self.val, -other / self.val**2, op = "(/)")
        self.children.append((0, child))
        return child
    
    # Unary operations
    def __neg__(self):
        child = Node(- self.val, -1, op ="-(\cdot)")
        self.children.append((0, child))
        return child
    
    @property
    def supported_scalars(self):
        """Get suppported scalars"""
        return self._supported_scalars