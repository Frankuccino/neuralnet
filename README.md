# Neural Net

A professional engineering portfolio and monorepo dedicated to building neural networks and deep learning systems **completely from scratch**. 

This repository serves as a practical, first-principles exploration of machine learning primitives—moving from foundational mathematical engines up to advanced deep learning architectures—built to master the core mechanics of AI under the hood.

---

## 🧠 Architecture & Portfolio Showcase

- **`micrograd`**: A tiny, scalar-valued auto-gradient engine and modular Multi-Layer Perceptron (MLP) library built from first principles with automated backpropagation and a clean Python package structure.
- *More coming soon:* Advanced architectures, custom optimizers, and transformer blocks.

---


## Getting Started & Developer Workflow
To set up this repository locally for development, follow these standard steps:

### 1. Clone the Repository 
```zsh
git clone https://github.com/Frankuccino/neuralnet.git
cd neuralnet
```

### 2. Create and Activate a Virtual Environment:

Isolating your project dependencies prevents conflicts with your global Python environment.

macOS/Linux:
```zsh
python3 -m venv .venv
# for macOS
source .venv/bin/activate
```
Windows(PowerShell):
```bash
python -m venv .venv
# for Windows
.venv\Scripts\Activate.ps1

```

When you're done working on the project, you can turn it off by simply typing: `deactivate` in shell command.


### 3. Install the Package in Editable Mode

Install the monorepo package in editable mode(`-e`), which links your local code directly into your environment so any changes you make take effect immediately.

```zsh
# Install with optional development/visualization tools (graphviz, matplotlib, etc.)
pip install -e ".[dev]"
```

### 4. Run the Demos

Verify your installation by running any of the portfolio demo scripts from the root directory:
```zsh
python demos/micrograd_demo.py
```

## Demos README documentation
[micrograd](./micrograd/README.md)
