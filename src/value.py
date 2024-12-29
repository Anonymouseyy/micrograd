import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # Function to calculate gradients
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad # Chain rule as derivatice of + is 1
            other.grad += out.grad # Add gradients to accumulate them in multivariate case

        out._backward = _backward
        return out

    def __radd__(self, other): # Fallback for other + self (const + Value)
        return other + self
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad # Chain rule and derivative of * is other.data
            other.grad += self.data * out.grad
         
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # Fallback for other * self (const * Value)
        return other * self
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other): # self - other
        return self + -other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Value(self.value**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data**(other-1) * self.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad # Chain rule and derivative of tanh is 1-tanh^2
        
        out._backward = _backward
        return out

    def exp(self):
        n = self.data
        out = Value(math.exp(n), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v): # Build topological graph to perform backprop
            if v not in visited:
                visited.add(v)
                
                for child in v._prev:
                    build_topo(child)
                
                topo.append(v)
        
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()