#!/usr/bin/env python3
"""Dr Moagi Hyper-Matrix: executable, bounded 3D mathematical engine.

SOURCE-DERIVED (Dr Moagi Hyper-Matrix Breakdown.pdf):
  - 3D surface mapping (u,v) -> (x,y,z), nominally 1,600 samples.
  - Scalar Gaussian pulse, threshold graph, synthetic spectral telemetry.
  - Intended continuum terms: tensor divergence, Gaussian/curl feedback,
    and second time derivative.

EXPLICIT ENGINEERING EXTENSIONS (not proven or specified by the PDF):
  - Pulse direction chosen radial; Gaussian length normalized by R0.
  - Consistently typed curl(g * curl(Omega)) replaces the PDF's rank-mismatched
    divergence of [rank-2 tensor + vector].
  - Periodic finite-difference *volumetric* PDE with specified dynamics.
  - Sparse graph fixed-point update with provable sup-norm contraction.
  - Reversible 3D Morton address mapping and integer lifting octree.
  - CLI, reproducibility, tests, and JSON/NPZ output.

No neural weights are trained. No claim of exponential speedup is made.
Needs: Python >=3.10, numpy, scipy. Run `python dr_moagi_hypermatrix_engine.py --self-test`.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Config:
    # Source-derived surface coefficients. Defaults beyond node count are
    # illustrative: the document supplies symbols but not baseline values.
    nu: int = 40
    nv: int = 40
    R0: float = 10.0
    A: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    B: float = 0.5
    C: float = 0.5
    Omega: float = 2.0
    gamma: float = 0.25
    pulse_amplitude: float = 0.15
    seed: int = 123
    # Extended 3D finite-difference model.
    side: int = 16
    dt: float = 0.01
    steps: int = 30
    density: float = 1.0
    diffusion: float = 0.02
    damping: float = 0.20
    coupling: float = 0.10
    rotation: float = 0.10
    # Graph fixed-point model.
    eta: float = 0.8
    local_feedback: float = 0.2
    neighbor_feedback: float = 0.5
    max_iterations: int = 100
    convergence_tolerance: float = 1e-10


def validate_config(c: Config) -> None:
    if c.nu < 2 or c.nv < 2 or c.side < 4:
        raise ValueError('nu,nv >=2 and side >=4 required')
    if c.side & (c.side - 1):
        raise ValueError('volumetric side must be power-of-two')
    if c.gamma < 0 or c.R0 <= 0:
        raise ValueError('gamma must be >= 0 and R0 positive')
    if c.dt <= 0 or c.steps < 0 or c.density <= 0 or c.diffusion < 0 or c.damping < 0:
        raise ValueError('invalid volumetric parameters')
    if not (0 < c.eta <= 1) or abs(c.local_feedback) + abs(c.neighbor_feedback) >= 1:
        raise ValueError('graph fixed-point contraction requires eta in (0,1] and |a|+|b|<1')
    if c.max_iterations < 1 or c.convergence_tolerance <= 0:
        raise ValueError('invalid fixed-point stopping rule')
    # This is a conservative, linear-diffusion CFL guide, NOT a nonlinear
    # stability theorem for the entire proposed PDE.
    dx = 2.0 / c.side
    c_wave = np.sqrt(c.diffusion / c.density)
    if c_wave > 0 and c.dt > 0.25 * dx / (np.sqrt(3) * c_wave):
        raise ValueError('dt exceeds conservative wave CFL guide; decrease dt')


def surface_points(c: Config) -> np.ndarray:
    """PDF parameterization sampled on a nu x nv grid, shape (N,3)."""
    u, v = np.meshgrid(np.linspace(0, 4 * np.pi, c.nu),
                       np.linspace(0, 2 * np.pi, c.nv), indexing='ij')
    rad = c.R0 + c.A * np.sin(c.alpha * u) * np.cos(c.beta * v)
    return np.stack((rad * np.sin(u) * np.cos(v),
                     rad * np.sin(u) * np.sin(v) + c.B * np.cos(c.Omega * u),
                     rad * np.cos(u) + c.C * np.sin(c.beta * v)), axis=-1).reshape(-1, 3)


def surface_pulse(points: np.ndarray, t: float, c: Config) -> np.ndarray:
    """Source scalar pulse + stated radial-vector and dimensional extensions."""
    rng = np.random.default_rng(c.seed)
    phase = rng.uniform(0.0, 2 * np.pi, len(points))
    radius = np.linalg.norm(points, axis=1)
    direction = points / np.maximum(radius[:, None], 1e-12)
    scalar = (c.pulse_amplitude * np.sin(c.Omega * t + phase)
              * np.exp(-c.gamma * (radius / c.R0)**2))
    return points + scalar[:, None] * direction


def build_graph(points: np.ndarray, gamma: float) -> tuple[csr_matrix, int, float]:
    """Sparse row-normalized adjacency based on the PDF's exact distance rule."""
    threshold = 32.0 / (1.0 + 10.0 * gamma)
    pairs = cKDTree(points).query_pairs(threshold, output_type='ndarray')
    n = len(points)
    if len(pairs):
        i = np.concatenate([pairs[:, 0], pairs[:, 1]])
        j = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    else:
        graph = csr_matrix((n, n), dtype=float)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    invdegree = np.zeros(n, dtype=float)
    np.divide(1.0, degree, out=invdegree, where=degree > 0)
    return (diags(invdegree) @ graph).tocsr(), len(pairs), threshold


def contraction_bound(c: Config) -> float:
    return 1.0 - c.eta + c.eta * (abs(c.local_feedback) + abs(c.neighbor_feedback))


def graph_fixed_point(P: csr_matrix, forcing: np.ndarray, c: Config) -> tuple[np.ndarray, dict]:
    """s'=(1-eta)s+eta*tanh(a*s+b*P@s+f), a contractive mapping.

    For a row-substochastic P, ||P||_inf<=1, and tanh 1-Lipschitz,
    ||F(s)-F(t)||_inf <= q ||s-t||_inf, q < 1.
    Fixed-point a posteriori error <= q/(1-q) * ||s_k - s_(k-1)||.
    """
    if forcing.ndim != 1 or forcing.shape[0] != P.shape[0]:
        raise ValueError('graph forcing vector size mismatch')
    q = contraction_bound(c)
    s = np.zeros_like(forcing, dtype=float)
    history = []
    for _ in range(c.max_iterations):
        nxt = (1-c.eta)*s + c.eta * np.tanh(c.local_feedback*s +
                                           c.neighbor_feedback*(P @ s) + forcing)
        residual = float(np.max(np.abs(nxt - s)))
        history.append(residual)
        s = nxt
        if residual <= c.convergence_tolerance:
            break
    fixed_residual = float(np.max(np.abs((1-c.eta)*s + c.eta*np.tanh(
        c.local_feedback*s + c.neighbor_feedback*(P @ s) + forcing) - s)))
    return s, {'iterations': len(history), 'step_residual': history[-1],
               'fixed_point_residual': fixed_residual, 'q_bound': q,
               'a_posteriori_error_bound': q/(1-q)*history[-1],
               'stopped_at_tolerance': history[-1] <= c.convergence_tolerance,
               'residual_history': history}


def morton_encode(x: int, y: int, z: int, bits: int) -> int:
    """Bijective 3D Morton index on [0,2**bits)^3, returns 3*bits bits."""
    if bits < 1 or not all(isinstance(v, (int, np.integer)) and
                            0 <= v < (1 << bits) for v in (x, y, z)):
        raise ValueError('invalid Morton coordinate or bit depth')
    index = 0
    for k in range(bits):
        index |= ((x >> k) & 1) << (3*k)
        index |= ((y >> k) & 1) << (3*k+1)
        index |= ((z >> k) & 1) << (3*k+2)
    return index


def morton_decode(index: int, bits: int) -> tuple[int, int, int]:
    if bits < 1 or not isinstance(index, (int, np.integer)) or not 0 <= index < 1 << (3*bits):
        raise ValueError('Morton index out of range')
    x = y = z = 0
    for k in range(bits):
        x |= ((index >> (3*k)) & 1) << k
        y |= ((index >> (3*k+1)) & 1) << k
        z |= ((index >> (3*k+2)) & 1) << k
    return x, y, z


OFFSETS = tuple((i, j, k) for i in range(2) for j in range(2)
                for k in range(2) if (i, j, k) != (0, 0, 0))


def octree_encode(cube: np.ndarray) -> tuple[int, list[dict[tuple[int,int,int], np.ndarray]]]:
    """7 residual arrays + anchor per parent; mathematically exact, not compression.

    Child (000) is the anchor and other seven values are child - anchor.
    Int64 is bounded: refuse ranges that could overflow signed subtraction.
    """
    cube = np.asarray(cube)
    if cube.ndim != 3 or len(set(cube.shape)) != 1 or cube.shape[0] < 1 or \
       cube.shape[0] & (cube.shape[0] - 1) or cube.dtype.kind not in 'iu':
        raise ValueError('require integer, cubic, power-of-two 3D array')
    if int(cube.max()) - int(cube.min()) > np.iinfo(np.int64).max or \
       int(cube.max()) > np.iinfo(np.int64).max:
        raise OverflowError('int64 residual difference would overflow')
    fine = cube.astype(np.int64, copy=True)
    levels = []
    while fine.shape[0] > 1:
        coarse = fine[0::2, 0::2, 0::2].copy()
        levels.append({offset: fine[offset[0]::2, offset[1]::2, offset[2]::2] - coarse
                       for offset in OFFSETS})
        fine = coarse
    return int(fine[0, 0, 0]), levels


def octree_decode(anchor: int, levels: list[dict]) -> np.ndarray:
    coarse = np.array([[[anchor]]], dtype=np.int64)
    for details in reversed(levels):
        n = coarse.shape[0]
        fine = np.empty((2*n, 2*n, 2*n), dtype=np.int64)
        fine[0::2, 0::2, 0::2] = coarse
        for offset in OFFSETS:
            i, j, k = offset
            fine[i::2, j::2, k::2] = coarse + details[offset]
        coarse = fine
    return coarse


def dd(a: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """Second-order centered first derivative with PERIODIC 3D boundaries."""
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2*dx)


def laplacian(a: np.ndarray, dx: float) -> np.ndarray:
    """Second-order 7-point Laplacian, independently for each vector component."""
    return sum((np.roll(a, -1, axis=i) + np.roll(a, 1, axis=i) - 2*a)/dx**2
               for i in range(3))


def curl(field: np.ndarray, dx: float) -> np.ndarray:
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError('expected 3D vector field shape (S,S,S,3)')
    return np.stack((dd(field[..., 2], 1, dx)-dd(field[..., 1], 2, dx),
                     dd(field[..., 0], 2, dx)-dd(field[..., 2], 0, dx),
                     dd(field[..., 1], 0, dx)-dd(field[..., 0], 1, dx)), axis=-1)


def div_outer(psi: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """[div(psi outer phi)]_i = sum_j d/dx_j (psi_i * phi_j)."""
    return np.stack([sum(dd(psi[..., i]*phi[..., j], j, dx) for j in range(3))
                     for i in range(3)], axis=-1)


def periodic_volume(c: Config) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    """3D volume, unlike the PDF's 2-parameter surface in 3D."""
    coords = np.linspace(-1.0, 1.0, c.side, endpoint=False)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    r2 = xx**2 + yy**2 + zz**2
    bump = np.exp(-8*r2)
    psi = np.stack((-yy*bump, xx*bump, zz*bump), axis=-1)
    vel = np.zeros_like(psi)
    phi = np.empty_like(psi)
    phi[...] = [0.2, 0.1, -0.1]  # chosen static feedback vector
    omega = np.stack((-yy*np.exp(-3*r2), xx*np.exp(-3*r2),
                      0*zz), axis=-1)
    return psi, vel, phi, omega, 2.0/c.side


def volume_acceleration(psi: np.ndarray, velocity: np.ndarray, phi: np.ndarray,
                        omega: np.ndarray, dx: float, c: Config) -> tuple[np.ndarray,dict]:
    """Well-typed EXTENSION, NOT the PDF's unspecified equation of motion:

      rho * psi_tt + nu * psi_t - kappa * laplacian(psi)
           + alpha * div(psi ⊗ phi)
           + beta * curl(g * curl(omega)) = 0.

    Gaussian g(x) = exp(-gamma ||x||^2) on normalized [-1,1)^3 grid.
    The periodic extension is an algorithmic boundary condition; the analytic
    Gaussian is not necessarily continuous at opposite faces.
    """
    n = psi.shape[0]
    axis = np.linspace(-1, 1, n, endpoint=False)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing='ij')
    g = np.exp(-c.gamma * (xx*xx+yy*yy+zz*zz))
    couple = div_outer(psi, phi, dx)
    rotate = curl(g[..., None] * curl(omega, dx), dx)
    diffuse = laplacian(psi, dx)
    force = c.diffusion*diffuse - c.damping*velocity - c.coupling*couple - c.rotation*rotate
    acceleration = force / c.density
    # Algebraic residual for the DISCRETIZED equation evaluated with this acceleration.
    residual = c.density*acceleration - force
    return acceleration, {'discrete_force_residual_max': float(np.max(np.abs(residual))),
                          'psi_norm': float(np.linalg.norm(psi)),
                          'accel_norm': float(np.linalg.norm(acceleration)),
                          'coupling_norm': float(np.linalg.norm(couple)),
                          'curl_force_norm': float(np.linalg.norm(rotate))}


def evolve_volume(c: Config) -> tuple[np.ndarray, dict]:
    """Semi-implicit Euler (velocity first); numerical stability must be measured."""
    psi, vel, phi, omega, dx = periodic_volume(c)
    first_norm = float(np.linalg.norm(psi))
    snapshots = []
    diagnostics = {}
    for k in range(c.steps):
        acceleration, diag = volume_acceleration(psi, vel, phi, omega, dx, c)
        vel += c.dt * acceleration
        psi += c.dt * vel
        if not np.isfinite(psi).all() or not np.isfinite(vel).all():
            raise FloatingPointError(f'nonfinite PDE state at step {k+1}')
        snapshots.append(float(np.linalg.norm(psi)))
        diagnostics = diag
    return psi, {'grid_shape': list(psi.shape), 'voxel_count': c.side**3,
                 'time_steps': c.steps, 'initial_state_norm': first_norm,
                 'final_state_norm': float(np.linalg.norm(psi)),
                 'max_time_series_norm': max(snapshots, default=first_norm),
                 'final_equation_diagnostics': diagnostics}


def synthetic_telemetry(seed: int = 123, count: int = 25) -> np.ndarray:
    """Illustrative PDF-formula series, not computed operator eigenvalues."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, count+1)
    return 10 * np.exp(-0.1*k) + rng.normal(0, .01, count)



def save_octree(path: Path, anchor: int, levels: list[dict]) -> None:
    """Write an independently decodable, non-pickle compressed NPZ container."""
    payload = {'anchor': np.array([anchor], dtype=np.int64),
               'num_levels': np.array([len(levels)], dtype=np.int64)}
    for lev, details in enumerate(levels):
        for offset in OFFSETS:
            payload[f'level_{lev}_{offset[0]}{offset[1]}{offset[2]}'] = details[offset]
    np.savez_compressed(path, **payload)


def load_octree(path: Path) -> np.ndarray:
    """Decode independently from stored integer residual layers."""
    with np.load(path, allow_pickle=False) as archive:
        anchor = int(archive['anchor'][0])
        depth = int(archive['num_levels'][0])
        if depth < 0 or depth > 25:  # reject absurd encoded dimensions
            raise ValueError('bad octree depth')
        levels = []
        for lev in range(depth):
            levels.append({offset: archive[f'level_{lev}_{offset[0]}{offset[1]}{offset[2]}']
                           for offset in OFFSETS})
    return octree_decode(anchor, levels)


def sha256_array(array: np.ndarray) -> str:
    """Stable digest for these explicitly little-endian signed integer cells."""
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes(order='C')).hexdigest()


def plot_3d(points: np.ndarray, graph_field: np.ndarray, volume: np.ndarray,
            destination: Path, seed: int = 123) -> None:
    """Optional 3D output: node surface coloured by computed fixed-point state,
    alongside a sparse sample of volumetric vector magnitudes. Visualization
    is not evidence of neural learning or a physical quantum phenomenon.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13,6), layout='constrained')
    ax = fig.add_subplot(121,projection='3d')
    ax.scatter(points[:,0],points[:,1],points[:,2],c=graph_field,cmap='viridis',
               s=4,alpha=.9,rasterized=True)
    ax.set_title('PDF-parametric 3D nodes | contractive graph state')
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    ax.view_init(elev=22, azim=35)
    ax2 = fig.add_subplot(122,projection='3d')
    n=volume.shape[0]
    sample=max(1,n//12)
    xi,yi,zi=np.mgrid[0:n:sample,0:n:sample,0:n:sample]
    mag=np.linalg.norm(volume[::sample,::sample,::sample],axis=-1)
    plot=ax2.scatter(xi.flatten(),yi.flatten(),zi.flatten(),c=mag.flatten(),
                     cmap='plasma',s=12,alpha=.65,rasterized=True)
    fig.colorbar(plot,ax=ax2,shrink=.65,label='||3D state vector||')
    ax2.set_title(f'True volume | {n}³ voxels (subsampled)')
    ax2.set_xlabel('i');ax2.set_ylabel('j');ax2.set_zlabel('k')
    ax2.view_init(elev=22,azim=35)
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def run_engine(c: Config, outdir: Path | None = None, plot: bool = False) -> dict:
    validate_config(c)
    t0 = time.perf_counter()
    base = surface_points(c)
    displaced = surface_pulse(base, t=1.0, c=c)
    P, edges, threshold = build_graph(displaced, c.gamma)
    forcing = np.linalg.norm(displaced, axis=1) / (10 * c.R0)
    state, graph_diagnostics = graph_fixed_point(P, forcing, c)
    field, volume_diagnostics = evolve_volume(c)
    # An integer 3D input: signed voxel values (NOT a float roundtrip claim).
    grid = np.indices((c.side,)*3, dtype=np.int64)
    original = 7*grid[0] - 5*grid[1] + 11*grid[2]
    anchor, levels = octree_encode(original)
    restored = octree_decode(anchor, levels)
    exact = bool(np.array_equal(original, restored))
    assert exact
    input_digest = sha256_array(original)
    restored_digest = sha256_array(restored)
    assert input_digest == restored_digest
    # Verify one full Morton address space for reasonable demo sides.
    bits = c.side.bit_length()-1
    if c.side <= 32:
        addresses = {morton_encode(int(x), int(y), int(z), bits)
                     for x in range(c.side) for y in range(c.side) for z in range(c.side)}
        address_test = len(addresses) == c.side**3 and addresses == set(range(c.side**3))
    else:
        address_test = all(morton_decode(morton_encode(x,y,z,bits),bits) == (x,y,z)
                           for x,y,z in ((0,0,0),(c.side-1,)*3,(1,2,3)))
    assert address_test
    stats = {'engine': 'Dr Moagi Hyper-Matrix 3D bounded reference implementation',
             'source_status': 'PDF geometry/threshold; all other mechanics explicitly extended',
             'surface_nodes': len(base), 'graph_undirected_edges': edges,
             'graph_threshold': threshold, 'graph_fixed_point': graph_diagnostics,
             'volume': volume_diagnostics,
             'octree': {'side': c.side, 'depth': len(levels), 'anchor': anchor,
                        'exact_integer_roundtrip': exact,
                        'sha256_original': input_digest,
                        'sha256_decoded': restored_digest,
                        'number_of_stored_integer_values': c.side**3},
             'morton_bijection_test': address_test,
             'telemetry_is_synthetic': True,
             'elapsed_seconds': float(time.perf_counter()-t0),
             'configuration': asdict(c)}
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        encoded_path = outdir/'octree_encoded.npz'
        save_octree(encoded_path, anchor, levels)
        reread = load_octree(encoded_path)
        disk_roundtrip = bool(np.array_equal(original, reread))
        if not disk_roundtrip:
            raise AssertionError('on-disk octree reconstruction failed')
        stats['octree']['independent_disk_roundtrip'] = disk_roundtrip
        stats['octree']['encoded_bytes_on_disk'] = encoded_path.stat().st_size
        if plot:
            plot_3d(displaced, state, field, outdir/'hypermatrix_3d.png', c.seed)
        with (outdir/'verification.json').open('w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, allow_nan=False)
        np.savez_compressed(outdir/'spatial_results.npz',
                            surface=base, displaced=displaced, graph_state=state,
                            volume_field=field, input_voxels=original,
                            reconstructed_voxels=restored,
                            synthetic_spectral_trace=synthetic_telemetry(c.seed))
    return stats


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.c = Config(side=8, steps=4, nu=8, nv=9, max_iterations=150)

    def test_surface_shape_and_finiteness(self):
        p = surface_points(self.c)
        self.assertEqual(p.shape, (72,3))
        self.assertTrue(np.isfinite(p).all())

    def test_pulse_zero_amplitude(self):
        from dataclasses import replace
        c = replace(self.c, pulse_amplitude=0.)
        p = surface_points(c)
        np.testing.assert_array_equal(surface_pulse(p, 1., c), p)

    def test_graph_symmetric_connections_and_row_sums(self):
        p = surface_points(self.c)
        P, edgecount, _ = build_graph(p, self.c.gamma)
        self.assertGreater(edgecount, 0)
        self.assertTrue(np.all((np.asarray(P.sum(axis=1)).ravel() <= 1.000000001)))

    def test_graph_contract_and_equation(self):
        p = surface_points(self.c)
        P, _, _ = build_graph(p, self.c.gamma)
        s, meta = graph_fixed_point(P, np.ones(len(p))*.1, self.c)
        self.assertLess(meta['q_bound'], 1.)
        self.assertLess(meta['fixed_point_residual'], 1e-9)
        self.assertEqual(s.shape, (len(p),))

    def test_graph_termination_flag(self):
        from dataclasses import replace
        c = replace(self.c, max_iterations=1, convergence_tolerance=1e-14)
        P = csr_matrix((2,2), dtype=float)
        _, meta = graph_fixed_point(P, np.ones(2), c)
        self.assertFalse(meta['stopped_at_tolerance'])

    def test_morton_exhaustive_4bits(self):
        bits=4
        codes=set()
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    m=morton_encode(x,y,z,bits)
                    self.assertEqual((x,y,z),morton_decode(m,bits))
                    codes.add(m)
        self.assertEqual(codes,set(range(16**3)))

    def test_morton_rejects_out_of_range(self):
        with self.assertRaises(ValueError): morton_encode(16,0,0,4)
        with self.assertRaises(ValueError): morton_decode(4096,4)

    def test_octree_random_roundtrip(self):
        rng=np.random.default_rng(678)
        for n in (1,2,4,8,16,32):
            original=rng.integers(-10000,10000,size=(n,n,n),dtype=np.int64)
            root,level=octree_encode(original)
            np.testing.assert_array_equal(original,octree_decode(root,level))
            self.assertEqual(1+sum(sum(arr.size for arr in d.values()) for d in level), n**3)

    def test_octree_disk_roundtrip(self):
        from tempfile import TemporaryDirectory
        original=np.arange(8**3,dtype=np.int64).reshape(8,8,8)-25
        a, lev=octree_encode(original)
        with TemporaryDirectory() as temp:
            path=Path(temp)/'octree.npz'
            save_octree(path,a,lev)
            np.testing.assert_array_equal(load_octree(path),original)
            self.assertEqual(sha256_array(original),sha256_array(load_octree(path)))

    def test_octree_rejects_overflow(self):
        arr=np.zeros((2,2,2),dtype=np.int64)
        arr[0,0,0]=np.iinfo(np.int64).min
        arr[1,0,0]=np.iinfo(np.int64).max
        with self.assertRaises(OverflowError): octree_encode(arr)

    def test_laplacian_constant_zero(self):
        v=np.ones((8,8,8,3))
        np.testing.assert_array_equal(laplacian(v,0.25),np.zeros_like(v))

    def test_curl_grad_zero_on_periodic_field(self):
        rng=np.random.default_rng(123)
        a=rng.normal(size=(8,8,8))
        grad=np.stack([dd(a,k,.25) for k in range(3)],axis=-1)
        np.testing.assert_allclose(curl(grad,.25),0,atol=2e-14)

    def test_div_outer_constant_zero(self):
        psi=np.ones((8,8,8,3))
        phi=np.ones_like(psi)
        np.testing.assert_array_equal(div_outer(psi,phi,.25),np.zeros_like(psi))

    def test_volumetric_solver_finite_and_typed(self):
        f,meta=evolve_volume(self.c)
        self.assertEqual(f.shape,(8,8,8,3))
        self.assertTrue(np.isfinite(f).all())
        self.assertLess(meta['final_equation_diagnostics']['discrete_force_residual_max'],1e-14)

    def test_full_engine(self):
        stats=run_engine(self.c)
        self.assertTrue(stats['octree']['exact_integer_roundtrip'])
        self.assertTrue(stats['morton_bijection_test'])
        self.assertEqual(stats['surface_nodes'],72)
        self.assertEqual(stats['volume']['voxel_count'],512)


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true',help='Run regression + invariance checks')
    parser.add_argument('--side',type=int,default=16,help='3D volume edge (power of two)')
    parser.add_argument('--steps',type=int,default=30,help='3D volumetric PDE time steps')
    parser.add_argument('--outdir',type=Path,default=None,help='Write JSON, numerical NPZ and encoded octree')
    parser.add_argument('--plot',action='store_true',help='Save 3D PNG (requires matplotlib and --outdir)')
    parser.add_argument('--nu',type=int,default=40,help='PDF parametric u samples')
    parser.add_argument('--nv',type=int,default=40,help='PDF parametric v samples')
    args=parser.parse_args(argv)
    if args.self_test:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(EngineTests)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
        return
    if args.plot and args.outdir is None:
        parser.error('--plot requires --outdir')
    c=Config(side=args.side,steps=args.steps,nu=args.nu,nv=args.nv)
    print(json.dumps(run_engine(c,args.outdir,args.plot),indent=2,allow_nan=False))


if __name__=='__main__':
    main()