import random
from src.value import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range (nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, x): # Perform forward pass
        assert len(x) == len(self.w), "need same amount of inputs as weights"
        
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class Convolutional_Layer2D_Flatten:

    def __init__(self, shapein, kernelsize, stride):
        self.kernel = Neuron(kernelsize[0] * kernelsize[1])
        self.kernelsize = kernelsize
        self.shapein = shapein
        self.stride = stride
    
    def __call__(self, x):
        outs = []

        for i in self.shapein[0]//self.stride:
            for j in self.shapein[1]//self.stride:
                inputs = [x[k+i][l+j] if k+i<self.shapein[0] and k+l<self.shapein[1] else 0 
                          for k in self.kernelsize[0] for l in self.kernelsize[1] ]
                outs.append(self.kernel(inputs))

        return outs
    
    def parameters(self):
        return [p for p in self.kernel.parameters()]

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.nin = nin
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        assert len(x) == self.nin, "need same amount of inputs lists as neuron inputs"

        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

class CNN:
    def __init__(self, conv, nouts):
        self.conv_layer = Convolutional_Layer2D_Flatten(*conv)
        self.MLP = MLP((conv[0][0]//conv[2])*(conv[0][1]//conv[2]), nouts)
    
    def __call__(self, x):
        x = self.conv_layer(x)
        x = self.MLP(x)
        return x
    
    def parameters(self):
        return self.conv_layer.parameters() + self.MLP.parameters()

