import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from src.nn import MLP

training_cycles = 50
step = 0.05

network = MLP(3, [5, 5, 1])

inputs = [
    [-2.0, 6.0, 1.2],
    [9.0, -3.1, 3.7],
    [3.5, 2.3, 9.2]
]

outputs = [
    0.1,
    1,
    -0.34
]

if input('Random inputs and outputs? (y/n) ').lower() == 'y':
    inputs = [[random.uniform(-10, 10) for _ in range(3)] for _ in range(3)]
    outputs = [random.uniform(-1, 1) for _ in range(3)]

for i in range(training_cycles):
    outs = [network(ins) for ins in inputs]
    L = sum((out-exp)**2 for out, exp in zip(outs, outputs))

    for p in network.parameters():
        p.grad = 0
    
    L.backward()

    for p in network.parameters():
        p.data += -step * p.grad
    
    print(f'Training step {i+1}: {L.data} loss')

print(f'Final: {L.data} loss')
print(f'Expected: {outputs}')
print(f'Outputs: {[network(ins).data for ins in inputs]}')
        