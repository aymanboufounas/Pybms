# Pybms

**Pybms** is a beginner-first deep-learning library built in pure Python + NumPy.

The design goal is simple: **make a neural network run with as little setup as possible**, while still exposing the ideas students need to learn: layers, activations, loss, forward propagation, backpropagation, gradients, optimizers, and parameter counts.

> Status: early prototype (`0.1.0`). Dense neural networks are supported. CNN/RNN/GPU support are future work.

## Easiest possible setup

```python
import pybms

model = pybms.easy(X, y, epochs=200)
predictions = model.predict(X)
```

Pybms automatically tries to infer the task, output size, output activation, loss, a small model architecture, optimizer, and input feature count. It prints the model structure and total parameter count.

## Simple custom model

```python
import pybms as pb

model = pb.Model()
model.add(8, activation="relu")
model.add(1, activation="sigmoid")

model.fit(X, y, epochs=500)
model.summary()
```

## Forgiving syntax

Pybms tries to recover from many common **valid-Python typos**:

```python
model = pb.Model()
model.add("dence", unit=8, activaton="reul")
model.add(1, activaton="sigmod")
model.complie(optmizer="adm", los="bce", lern_rate=0.01)
model.fitt(X, y, epocs=300)
model.summery()
```

Pybms warns you about each correction and continues when it can.

Important limitation: a library cannot repair Python code that **does not parse at all** (for example a missing quote, unmatched parenthesis, or invalid Python grammar), because Python fails before Pybms gets control. Pybms focuses on correcting library-level names, options, aliases, shapes, output setup, and common configuration mistakes.

## Automatic summary

Pybms reports layers, output shapes, total parameters, trainable parameters, and inferred task.

## Device

```python
import pybms as pb
pb.device()
```

Version 0.1 uses the NumPy CPU backend. The device API is included now so future GPU support can keep a stable syntax.

## XOR example

```python
import numpy as np
import pybms as pb

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([0,1,1,0])

model = pb.Model()
model.add(8, activation="tanh")
model.add(1, activation="sigmoid")
model.compile(optimizer="adam", loss="binary_crossentropy", learning_rate=0.03)
model.fit(X, y, epochs=1500, verbose=0)

model.summary()
print(model.predict(X, classes=True))
```

## Error philosophy

Pybms includes dedicated exceptions and checks empty targets, NaN/Inf data, X/y sample mismatch, wrong Dense dimensions, invalid unit counts, unknown activations/losses/optimizers, invalid epochs/batches, non-finite loss, and output-layer mismatch.

By default it repairs safe configuration mismatches when possible. Use `strict=True` if you prefer errors instead of auto-fixes.

## Roadmap

- [x] Dense layers
- [x] ReLU / Sigmoid / Tanh / Softmax / Linear
- [x] MSE / MAE / Binary Cross-Entropy / Categorical Cross-Entropy
- [x] SGD / Adam
- [x] Forward propagation
- [x] Backpropagation
- [x] Auto task inference
- [x] Auto model setup
- [x] Friendly typo recovery
- [x] Model summary + parameter count
- [ ] Save / load models
- [ ] Dropout
- [ ] Batch normalization
- [ ] Conv2D / pooling
- [ ] RNN / LSTM
- [ ] GPU backend
- [ ] Full tensor autograd engine
- [ ] PyPI release

## Install locally

```bash
git clone https://github.com/aymanboufounas/Pybms.git
cd Pybms
pip install -e .
```

## License

MIT
