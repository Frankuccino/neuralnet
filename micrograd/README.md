[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Frankuccino/neuralnet/blob/main/micrograd.ipynb)

# Micrograd Neuron: Value

This guide and accompanying notebook cover the foundational concepts of building a tiny auto-gradient engine (**micrograd**), exploring how derivatives work conceptually and numerically to pave the way for neural network backpropagation.

---

## Overview & Core Concepts

This repository implements a lightweight, scalar-valued autograd engine and neural network library from scratch.

### 1. The Autograd Engine (`Value`)
At the lowest level, every scalar value is wrapped in a `Value` node that maintains its `.data` and `.grad`. It builds a dynamic Directed Acyclic Graph (DAG) tracking every mathematical operation (`+`, `*`, `tanh`, `relu`).

### 2. Backpropagation & Topological Sort
By calling `.backward()` on a final output (like the loss), the engine performs a Depth-First Search (DFS) topological sort to order the nodes. It then traverses them in reverse, applying the chain rule to accumulate gradients (`+=`) automatically across all parent paths.

### 3. Loss & Gradient Descent
* **Loss Function:** Measures prediction error using Mean Squared Error (MSE).
* **Optimization:** A training loop evaluates predictions, computes the loss, clears old gradients (`zero_grad()`), propagates new gradients backwards, and steps the weights downward via gradient descent ($p.\text{data} -= \text{lr} \times p.\text{grad}$).

## Run the Demos

Verify your installation by running any of the portfolio demo scripts from the root directory:
```zsh
python demos/micrograd_demo.py
```
### Customizing Hyperparameters
The demo script supports custom command-line arguments so you can experiment with training performance on the fly:

`--epochs`: Number of forward/backward training passes (default: 25)

`--lr`: Learning rate controlling the gradient descent step size (default: 0.05)

Example with custom parameters:
```zsh
python demos/micrograd_demo.py --epochs 50 --lr 0.1
```

When running the demos, you should be able to see similar output:
```zsh
--- Initializing Micrograd Demo ---
Config -> Epochs: 25 | Learning Rate: 0.05
Model created with 41 trainable parameters.

Starting training loop...
Epoch  0 | Loss: 6.206523
Epoch  1 | Loss: 4.806551
Epoch  2 | Loss: 4.048194
Epoch  3 | Loss: 3.824874
Epoch  4 | Loss: 3.609204
Epoch  5 | Loss: 3.386914
Epoch  6 | Loss: 3.147694
Epoch  7 | Loss: 2.883820
Epoch  8 | Loss: 2.594800
Epoch  9 | Loss: 2.282597
Epoch 10 | Loss: 1.948193
Epoch 11 | Loss: 1.594947
Epoch 12 | Loss: 1.238104
Epoch 13 | Loss: 0.911121
Epoch 14 | Loss: 0.649882
Epoch 15 | Loss: 0.465154
Epoch 16 | Loss: 0.342428
Epoch 17 | Loss: 0.261354
Epoch 18 | Loss: 0.206464
Epoch 19 | Loss: 0.168000
Epoch 20 | Loss: 0.140100
Epoch 21 | Loss: 0.119217
Epoch 22 | Loss: 0.103154
Epoch 23 | Loss: 0.090504
Epoch 24 | Loss: 0.080339

Training finished successfully!

--- Final Model Predictions ---
Input: [2.0, 3.0, -1.0] | Target:  1.0 | Prediction: 0.9011
Input: [3.0, -1.0, 0.5] | Target: -1.0 | Prediction: -0.9382
Input: [0.5, 1.0, 1.0] | Target: -1.0 | Prediction: -0.8195
Input: [1.0, 1.0, -1.0] | Target:  1.0 | Prediction: 0.8393
```

## Overview & Core Implementation

### 1. Polynomial Evaluation & Visualization
We start by defining a basic quadratic function and plotting its curve using NumPy and Matplotlib:

```python
import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5  # quadratic polynomial (trinomial)

# Evaluate at a specific point
print(f(3.0))

# Plotting the curve
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
```

### 2. Numerical Derivatives (The Slope)

In neural networks, we don't use symbolic differentiation because an expression with thousands or millions of terms would yield massive, unmanageable equations. Instead, we use the definition of a derivative via finite differences to measure the slope:

```python
h = 0.0000001  # Small increment (avoiding precision limits of finite memory)
x = 2/3
slope = (f(x + h) - f(x)) / h
print(slope)
```

### 3. Multi-Variable Functions & Sensitivity Analysis

Moving to a function with multiple scalar inputs (*a*, *b*, and *c*) and a single output (*d*):

```python
a = 2.0
b = -3.0
c = 10.0
d = a * b + c

print(d)
```

We evaluate partial derivatives by introducing a tiny increment (*h*) to individual inputs (e.g., *c += h*) to understand how each variable influences the final output:

```python
h = 0.0001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
c += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1) / h)
```

### 4. Next Steps: Building the Value Object

Because real-world neural networks consist of massive mathematical expression graphs, we need custom data structures to track operations and maintain dependencies. The next step is building the Value object (as seen in the micrograd repository) to handle automated expression tracking and backpropagation.

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # Derivative of loss/output with respect to this value
        self._prev = set(_children)
        self._op = _op   # The operation that produced this node
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __rmul__(self, other):  # Allows things like 2 * a
        return self * other

    def __radd__(self, other):  # Allows things like 2 + a
        return self + other
```

#### 5. Example Usage: Building an Expression Graph

With the Value class defined, you can chain operations together to construct a computational graph:

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b; e.label = 'e'
d = e + c; d.label = 'd'

print(d)
print(d._prev)
print(d._op)
```

### 6. Visualizing the Computational Graph

To inspect our expression graphs visually, we can use Python libraries like graphviz to trace nodes (_prev) and operations (_op):

```python
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # Left to right
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        # Create a rectangular node for each value with data and grad
        dot.node(name=uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # Create an operation node
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
            
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    return dot
```

### 7. Implementing Backpropagation (The Backward Pass)

Instead of manually calculating derivatives using finite differences (*h*), backpropagation computes gradients recursively using the chain rule ($\frac{\partial L}{\partial x} = \frac{\partial L}{\partial out} \cdot \frac{\partial out}{\partial x}$).We define a backward method that utilizes a topological sort to ensure nodes are processed in the correct order (children before parents):

#### Running a complete backward pass
```python
# 1. Forward pass
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

# 2. Backward pass (computes gradients automatically)
# remember that we would need to reset the gradient to 0
L.backward()

# Visualizing or inspecting the graph
draw_dot(L)
```

### 8. Building Neural Network Primitives (Neuron, Layer, MLP)

With our scalar Value engine capable of automatic differentiation and backpropagation, we can now assemble higher-level neural network components: individual Neurons, Layers, and a Multi-Layer Perceptron (MLP)

```python
import random

class Neuron:
    def __init__(self, nin):
        # Weights and bias initialized as Value objects
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()

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

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### 9. Putting It Together: Training Loop & Gradient Descent

To train our network, we define a dataset, calculate a loss function (such as Mean Squared Error), clear old gradients, perform a backward pass, and update the weights using simple gradient descent.

```python 
# 1. Initialize a small MLP (2 inputs, hidden layers of 16/16 neurons, 1 output)
model = MLP(2, [16, 16, 1])

# 2. Example dataset (inputs and desired targets)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

# 3. Training Loop
for k in range(20):
    
    # Forward pass: predict targets
    ypred = [model(x) for x in xs]
    
    # Loss computation (Mean Squared Error)
    loss = sum((y - yi)**2 for y, yi in zip(ys, ypred))
    
    # Zero out gradients from the previous iteration
    for p in model.parameters():
        p.grad = 0.0
        
    # Backward pass
    loss.backward()
    
    # Gradient descent update step
    for p in model.parameters():
        p.data -= 0.05 * p.grad
        
    print(f"Step {k:2d} | Loss: {loss.data:.4f}")
```
