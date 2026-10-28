# Kinetic 3D ROM prototype

Sparse, reversible 3D packet addressing with a local damped six-neighbour wave field and occupancy-restricted response scoring.

Run from repository root:

```bash
python -m pip install numpy
python -m unittest discover -s kinetic3drom -p 'test_*.py'
python kinetic3drom/kinetic3drom.py
```

The 100,000³ domain is **virtual addressing only**, not allocated memory. The model is a retrieval simulation, not a trained ANN, physical wave processor, or proof of global optimisation. Local wave boundaries are periodic due to np.roll; the radius is chosen larger than the number of propagation steps to prevent boundary wrap affecting the query's causal region during the simulated interval. Further work: formal CFL/energy stability analysis, negative-case tests, benchmarks, and embedding integration.
