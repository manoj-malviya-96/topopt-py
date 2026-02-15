# topopt-py

A modular, high-performance Python implementation of topology optimization using the SIMP (Solid Isotropic Material with
Penalization) method.

## Overview

This project is a refactored and optimized version of
the [99-line topology optimization code](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python)
from DTU. While the original script is excellent for educational purposes, this implementation focuses on:

- **Modularity**: Clean separation into reusable components
- **Performance**: 2-3× faster through vectorization and caching
- **Extensibility**: Easy to add new problem types, solvers, and filters

## Installation

Requires Python 3.14+.

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

For optional CHOLMOD support (faster direct solver):

```bash
pip install scikit-sparse
```

## Usage

```bash
# Default: 100×50 MBB beam
python main.py

# Cantilever beam with custom parameters
python main.py --problem cantilever --nelx 150 --nely 50 --volfrac 0.4

# Enable profiling
python main.py --profile
```

### CLI Options

| Flag          | Default | Description                         |
|---------------|---------|-------------------------------------|
| `--nelx`      | 100     | Elements in x direction             |
| `--nely`      | 50      | Elements in y direction             |
| `--problem`   | mbb     | Problem type: `mbb` or `cantilever` |
| `--volfrac`   | 0.5     | Target volume fraction              |
| `--penal`     | 3.0     | SIMP penalization factor            |
| `--rmin`      | 1.5     | Filter radius (elements)            |
| `--maxiter`   | 100     | Maximum iterations                  |
| `--tol`       | 0.01    | Convergence tolerance               |
| `--solver`    | direct  | Solver: `direct` or `iterative`     |
| `--no-filter` | -       | Disable density filtering           |
| `--profile`   | -       | Enable cProfile output              |
| `--verbose`   | -       | Verbose logging                     |

## Comparison with Original DTU Script

The original 99-line script ([
`original`](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python)) is a
compact, educational
implementation. This refactored version maintains algorithmic equivalence while improving performance and code
organization.

### Architectural Differences

| Aspect                  | Original (99-line)                                   | This Implementation                                       |
|-------------------------|------------------------------------------------------|-----------------------------------------------------------|
| **Filter construction** | Nested 4-loop sparse matrix build                    | `scipy.ndimage.convolve` with pre-built kernel            |
| **Stiffness assembly**  | Rebuild `iK`, `jK` indices implicitly each iteration | `StiffnessAssembler` caches sparsity pattern              |
| **Strain energy**       | `np.dot` + manual reshape + sum                      | `np.einsum('ij,jk,ik->i', ...)` vectorized                |
| **OC update**           | Allocates new arrays each iteration                  | In-place ops with pre-allocated buffers                   |
| **Solver**              | Direct `spsolve` only                                | Abstracted solver classes, optional CHOLMOD, iterative CG |
| **Memory**              | ~18 intermediate arrays per iteration                | ~6 pre-allocated arrays reused                            |

### Benchmark Results

**Test configuration**: 100×50 mesh (5,000 elements), 100 iterations, MBB beam, direct solver (LU)

| Metric            | Original    | topopt-py     | Improvement     |
|-------------------|-------------|---------------|-----------------|
| **Total time**    | 4.80s       | 2.56s         | **1.9× faster** |
| **Solver time**   | ~3.5s (73%) | ~2.1s (82%)   | 1.7× faster     |
| **Assembly time** | ~0.8s (17%) | ~0.23s (9%)   | **3.5× faster** |
| **Filter time**   | ~0.3s (6%)  | ~0.02s (0.8%) | **15× faster**  |

> **Note**: Solver time dominates in both implementations—this is inherent to FEM. The refactored version reduces
> overhead so that ~82% of time is spent in the unavoidable sparse factorization.

### Profiler Output (topopt-py)

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    2.108    0.021    2.108    0.021 {built-in method scipy.sparse.linalg._dsolve._superlu.gstrf}
      100    0.003    0.000    0.229    0.002 core/stiffness.py:47(assemble)
      100    0.023    0.000    0.042    0.000 core/optimize.py:50(_oc_update)
      201    0.001    0.000    0.019    0.000 core/filters.py:19(apply_density_filter)
```

The sparse LU factorization (`gstrf`) accounts for 82% of runtime—further optimization requires either:

- Iterative solvers with good preconditioners (use `--solver iterative`)
- CHOLMOD for symmetric positive definite systems (`pip install scikit-sparse`)
- Multigrid methods (not implemented)

## References

- [DTU TopOpt Group](https://www.topopt.mek.dtu.dk/) - Original 99-line code and educational resources
- Andreassen, E., et al. (2011). "Efficient topology optimization in MATLAB using 88 lines of code"
- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods and Applications*

