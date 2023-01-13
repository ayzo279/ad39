#!/usr/bin/env python3
# File       : reverse_core.py
# Description: Module that provides core functions for reverse mode automatic differentiation (AD)

from ..core import AD
from ad39_package.reverse.reverse_helpers import *
import networkx as nx
import matplotlib.pyplot as plt

class RM(AD):

    def __init__(self, f):
        super().__init__(f)
        assert 1 == self._nargs, f"`reverse_mode` method only accepts functions with one scalar or vector input"

    
    def _get_adjoint(self, node):
        """This private method performs a step in the reverse pass for a node used in `reverse_mode`"""
        # First check edge case for identity function within vector output
        try:
            if node.next:
                return 1
        except AttributeError:
            pass

        if len(node.children) == 0:
            if node.next:
                return 1
            else:
                return 0
        node.adjoints = 0
        for child in node.children:
            node.adjoints += self._get_adjoint(child[1]) * child[1].partials[child[0]]
        return node.adjoints


    def eval(self, inputs):
        """Evaluate the value of `self.f` at `inputs`.

        Parameters
        ----------
        inputs : int, float, np.ndarray
            Inputs at which to evaluate function.

        Returns
        -------
        res : float, numpy.ndarray
            Result of `self.f` evaluated at `inputs`.
        
        """
        if isinstance(inputs, Node._supported_scalars):
                    # Make inputs nodes object
                    node_inputs = Node(inputs)
                    
                    # Evaluate function at given inputs
                    output = self.f(node_inputs)

        elif isinstance(inputs, np.ndarray):
            # Make inputs node objects
            node_inputs = np.array(Node._from_list(inputs))
            
            # Evaluate function at given inputs
            output = self.f(node_inputs)

        else:
            raise TypeError(f"Incompatible input. Must be np.array, int or float but was of type: {type(inputs)}")
        
        if isinstance(output, np.ndarray):
                return np.array([float(out.val) for out in output])
        else:
            return float(output.val)


    def grad(self, inputs):
        """Compute the derivative of `self.f` evaluated at `inputs` using reverse mode AD.

        Parameters
        ----------
        inputs : int, float, np.ndarray
            Inputs at which to evaluate derivative.
            Note the `reverse_mode` method excludes functionality for 
            functions taking tuple as input.

        Returns
        -------
        deriv : float, numpy.ndarray
            Derivative of `self.f` evaluated at `x`.
        """
        if isinstance(inputs, Node._supported_scalars):
            # Make inputs nodes object
            node_inputs = Node(inputs)
            
            # Evaluate function at given inputs
            output = self.f(node_inputs)

            # Calculate the reverse pass
            if isinstance(output, np.ndarray):
                n_fun = len(output)
                grad = []
                for fnode in output:
                    fnode.next = False
                output[0].next = True
                grad.append([self._get_adjoint(node_inputs)])
                for i in range(1, n_fun):
                    output[i - 1].next = False
                    output[i].next = True
                    grad.append([self._get_adjoint(node_inputs)])
            else:
                output.next = True
                grad = [self._get_adjoint(node_inputs)]

        elif isinstance(inputs, np.ndarray):
            # Make inputs node objects
            node_inputs = np.array(Node._from_list(inputs))
            
            # Evaluate function at given inputs
            output = self.f(node_inputs)

            # Calculate the reverse pass
            if isinstance(output, np.ndarray):
                n_fun = len(output)
                grad = []
                for fnode in output:
                    fnode.next = False
                output[0].next = True
                grad.append([self._get_adjoint(node) for node in node_inputs])
                for i in range(1, n_fun):
                    output[i - 1].next = False
                    output[i].next = True
                    grad.append([self._get_adjoint(node) for node in node_inputs])
            else:
                output.next = True
                grad = [self._get_adjoint(node) for node in node_inputs]

        else:
            raise TypeError(f"Incompatible input. Must be np.array, int or float but was of type: {type(inputs)}")
        
        if len(grad) == 1:
            try:
                return float(grad[0][0])
            except:
                return float(grad[0])
        else:
            return np.array(grad).T.astype(float)
       
    def jacobian(self, inputs):
        """Compute the Jacobian of `self.f` evaluated at `inputs` using reverse mode AD.

        Parameters
        ----------
        inputs : int, float, np.ndarray
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
        grad = self.grad(inputs)

        if isinstance(grad, np.ndarray):
            return grad.T
        else:
            return grad


    def graph(self, inputs, size = (8, 6)):
        """Draw the computational graph in reverse mode AD.

        Parameters
        ----------
        inputs : int, float, np.ndarray, tuple
            Inputs for which to draw the graph.
        size: tuple
            Figure size

        mode: "forward", "f", "reverse", "r" (optional)
            Specification of whether to use forward ot reverse
            mode to compute the Jacobian. Note that inputs of type 
            tuple will not work with `mode="reverse"` or `mode="r"`.

        Returns
        -------
        None
        
        """
        G = nx.DiGraph()
        if isinstance(inputs, Node._supported_scalars):
            # Make inputs nodes object
            node_inputs = Node(inputs)
            G.add_node(node_inputs, name = "$x = v_0$")
        elif isinstance(inputs, np.ndarray):
            # Make inputs node objects
            node_inputs = np.array(Node._from_list(inputs))
            for i in range(len(node_inputs)):
                G.add_node(node_inputs[i], name = f'$x_{i} = v_{i}$')
        else:
            raise TypeError(f"Incompatible input. Must be np.array, int or float but was of type: {type(inputs)}")

        self.f(node_inputs)

        def draw(x, G):
            inc = len(G)
            for child in x.children:
                cnode = child[1]
                if not G.has_node(cnode):
                    G.add_node(cnode, name = f'$v_{inc}$')
                    inc+=1
                    G.add_edge(x, cnode, id = cnode.op)
                    draw(cnode, G)
                elif len(G.in_edges(x)) != 0:
                    G.nodes[x]['name'] = G.nodes[cnode]['name']
                    G.nodes[cnode]['name'] = f'$v_{inc - 1}$'
                    draw(cnode, G)
                G.add_edge(x, cnode, id = cnode.op)
        try:
            for node in node_inputs:
                draw(node, G)
        except:
            draw(node_inputs, G)

        node_labels = nx.get_node_attributes(G, 'name')
        edge_labels = nx.get_edge_attributes(G, 'id')
        col = []
        for node in G:
            if not G.in_edges(node):
                col.append('#fa4153')
            elif not G.out_edges(node): 
                col.append('#44f272')  
            else:
                col.append('#6272fc')
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=size) 
        nx.draw(G, pos, labels = node_labels, node_color = col, node_size = 1000, font_size = 10)
        nx.draw_networkx_edge_labels(G, pos, label_pos = 0.3, edge_labels=edge_labels, rotate=False, font_color = "#490e9c", font_weight = "bold")
        plt.show()

    # The following functions extend the capabilities of the Nodes used to compute RM AD and are available to the user for use within functions
    
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
        ad = RM(lambda x : RM.sqrt(x))
        ad.eval(4) # equals 2
        """
        if isinstance(x, Node._supported_scalars):
            return np.sqrt(x)

        elif isinstance(x, Node):
            child = Node(np.sqrt(x.val), 0.5 * (x.val ** -0.5), op = "sqrt()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.sin(x))
        ad.eval(math.pi) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.sin(x)

        elif isinstance(x, Node):
            child = Node(np.sin(x.val), np.cos(x.val), op = "sin()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.cos(x))
        ad.eval(math.pi) # equals -1
        """
        if isinstance(x, Node._supported_scalars):
            return np.cos(x)

        elif isinstance(x, Node):
            child = Node(np.cos(x.val), -np.sin(x.val), op="cos()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.tan(x))
        ad.eval(math.pi) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.tan(x)

        elif isinstance(x, Node):
            child = Node(np.tan(x.val), 1 / (np.cos(x.val) ** 2), op="tan()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.arcsin(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.arcsin(x)

        elif isinstance(x, Node):
            child = Node(np.arcsin(x.val), 1 / np.sqrt(1 - (x.val ** 2)), op = "arcsin()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.arccos(x))
        ad.eval(1) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.arccos(x)

        elif isinstance(x, Node):
            child = Node(np.arccos(x.val), - 1 / np.sqrt(1 - (x.val ** 2)), op = "arccos()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.arctan(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.arctan(x)

        elif isinstance(x, Node):
            child = Node(np.arctan(x.val), 1 / (1 + (x.val ** 2)), op="arctan()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.sinh(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.sinh(x)

        elif isinstance(x, Node):
            child = Node(np.sinh(x.val), np.cosh(x.val), op="sinh()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.cosh(x))
        ad.eval(0) # equals 1
        """
        if isinstance(x, Node._supported_scalars):
            return np.cosh(x)

        elif isinstance(x, Node):
            child = Node(np.cosh(x.val), np.sinh(x.val), op = "cosh()")
            x.children.append((0, child))
            return child

        else:
            raise TypeError(f'Unsupported type: {type(x)}')
    
    @staticmethod
    def tanh(x):
        """Compute tanh of `x` in a function. Output is not accessible unless `AD.eval(x)` is called.

        Parameters
        ----------
        x : int, float
            Argument to tanh function.

        Example
        -------
        ad = RM(lambda x : RM.tanh(x))
        ad.eval(0) # equals 0
        """
        if isinstance(x, Node._supported_scalars):
            return np.tanh(x)

        elif isinstance(x, Node):
            child = Node(np.tanh(x.val), 1 / (np.cosh(x.val) ** 2), op = "tanh()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.log(x))
        ad.eval(np.exp(1)) # equals 1

        ad = RM(lambda x : RM.log(x, 10))
        ad.eval(100) # equals 2
        """
        if isinstance(x, Node._supported_scalars):
            return float(np.log(x)/np.log(base))

        elif isinstance(x, Node):
            child = Node(float(np.log(x.val)/np.log(base)), 1/(np.log(base)*x.val), op = "log()")
            x.children.append((0, child))
            return child

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
        ad = RM(lambda x : RM.exp(x))
        ad.eval(1) # equals np.exp(1)

        ad = RM(lambda x : RM.exp(x, 10))
        ad.eval(1) # equals 10
        """
        if isinstance(x, Node._supported_scalars):
            if base == np.exp(1):
                return np.exp(x)
            else:
                return base ** x
        if isinstance(x, Node):
            if base == np.exp(1):
                child = Node(np.exp(x.val), np.exp(x.val), op = "exp()")
                x.children.append((0, child))
                return child
            else:
                if isinstance(base, Node._supported_scalars):
                    child = Node(base ** x.val, base ** x.val * np.log(base), op = "exp()")
                    x.children.append((0, child))
                    return child
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
        ad = RM(lambda x : RM.logistic(x))
        ad.eval(1) # equals np.exp(1) / (1 + np.exp(1))
        """
        def f(arg):
            return L / (1 + np.exp(-k*(arg-x_0)))
        if isinstance(x, Node._supported_scalars):
            return f(x)

        elif isinstance(x, Node):
            child = Node(f(x.val), x.val * (1-x.val), op = "logistic()")
            x.children.append((0, child))
            return child
        else:
            raise TypeError(f'Unsupported type: {type(x)}')