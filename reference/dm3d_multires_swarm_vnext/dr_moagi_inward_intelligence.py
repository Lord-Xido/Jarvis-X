#!/usr/bin/env python3
"""Dr Moagi 3D sparse-virtual multimodal ANN swarm (NumPy reference).

A working geometric/arithmetical demonstration, NOT allocation of the virtual
10^(18,000,000) domain and NOT a trained general-purpose cognitive agent.

Equations for physical active tiles t:
    P_t = pool_2(X_t)                         # 3D 2x2x2 average
    H_t = tanh(P_t W_e + b_e + mu Omega_t)
    A_j = F_j(H_t)                           # toroidal, diffusion, inward warp
    a_j = softmax(q)_j
    Z_t = sum_j a_j A_j
    Y_t = upsample_2(Z_t W_d + b_d)
    L = mean((Y_t - X_t)^2)
    Omega_t' = beta Omega_t + (1-beta) H_t  # after accepted update

All parameter gradients below differentiate this exact model treating Omega_t
as fixed within one step (truncated temporal backpropagation). A CTR-style
verifier accepts a trial update only if the selected-batch loss does not rise.

No external services required. Optional Pillow for images; OpenCV for video.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

AXIS_EXPONENT = 6_000_000                    # (10^6)^(10^6) = 10^6,000,000
VOLUME_EXPONENT = 3 * AXIS_EXPONENT           # 10^18,000,000 sites
MODALITIES = ("text", "image", "audio", "video")
AGENT_NAMES = (
    "identity", "shift_x+", "shift_x-", "shift_y+", "shift_y-",
    "shift_z+", "shift_z-", "toroidal_diffusion", "inward_warp",
)


@dataclass(frozen=True)
class VirtualTile:
    """A symbolic decimal prefix of a 3D virtual region; no huge ints allocated."""
    x_prefix: str
    y_prefix: str
    z_prefix: str

    def __post_init__(self):
        prefixes = (self.x_prefix, self.y_prefix, self.z_prefix)
        if not all(p and p.isascii() and p.isdecimal() and len(p) <= AXIS_EXPONENT
                   for p in prefixes):
            raise ValueError("Prefix must be nonempty ASCII decimal digits")

    def region_exponent(self) -> int:
        """Number of site addresses in the represented region is 10^result."""
        return 3 * AXIS_EXPONENT - sum(map(len, (self.x_prefix,
                                                    self.y_prefix,
                                                    self.z_prefix)))

    def ranges_symbolic(self) -> dict:
        """Axis interval is [int(prefix)*10^r, (int(prefix)+1)*10^r)."""
        result = {}
        for axis, p in zip("xyz", (self.x_prefix, self.y_prefix, self.z_prefix)):
            r = AXIS_EXPONENT - len(p)
            result[axis] = {"prefix": p, "lower": f"{p} * 10^{r}",
                            "upper_exclusive": f"({int(p)}+1) * 10^{r}"}
        return result


def virtual_tile_ids(n: int) -> list[VirtualTile]:
    if not 1 <= n <= 1000:
        raise ValueError("n_tiles must be in [1, 1000]")
    return [VirtualTile(f"{i:04d}", f"{(17*i+211)%10000:04d}",
                        f"{(37*i+877)%10000:04d}") for i in range(n)]


def avg_pool2(x: np.ndarray) -> np.ndarray:
    """(B,2L,2L,2L,C)->(B,L,L,L,C), averaging every 2x2x2 block."""
    b, n1, n2, n3, c = x.shape
    if n1 != n2 or n2 != n3 or n1 % 2:
        raise ValueError("Expected even cubic 3D tiles")
    n = n1 // 2
    return x.reshape(b, n, 2, n, 2, n, 2, c).mean(axis=(2, 4, 6))


def upsample2(x: np.ndarray) -> np.ndarray:
    return x.repeat(2, axis=1).repeat(2, axis=2).repeat(2, axis=3)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class Geometry3D:
    """Nine real linear agents F_j and their exact Euclidean adjoints F_j^T."""
    def __init__(self, n: int, inward_ratio: float = 0.65):
        if n < 2:
            raise ValueError("Latent edge must be >= 2")
        if not 0 < inward_ratio <= 1:
            raise ValueError("inward_ratio must lie in (0,1]")
        self.n = n
        q = np.indices((n, n, n), dtype=np.float64)
        center = (n-1)/2
        sampled = np.rint(center + inward_ratio*(q-center)).astype(np.int64)
        sampled %= n
        self.inward_idx = np.ravel_multi_index(tuple(sampled), (n,n,n)).ravel()

    @staticmethod
    def diffusion(x: np.ndarray) -> np.ndarray:
        a = x.copy()
        for axis in (1,2,3):
            a += np.roll(x, 1, axis=axis) + np.roll(x, -1, axis=axis)
        return a / 7.0

    def apply(self, j: int, h: np.ndarray) -> np.ndarray:
        if j == 0:
            return h
        if 1 <= j <= 6:
            axis = (j - 1) // 2 + 1
            shift = 1 if j % 2 else -1
            return np.roll(h, shift, axis=axis)
        if j == 7:
            return self.diffusion(h)
        if j == 8:
            b, _, _, _, c = h.shape
            return h.reshape(b, -1, c)[:, self.inward_idx, :].reshape(h.shape)
        raise ValueError("Unknown agent")

    def adjoint(self, j: int, grad: np.ndarray) -> np.ndarray:
        if j == 0:
            return grad
        if 1 <= j <= 6:
            axis = (j - 1) // 2 + 1
            shift = -1 if j % 2 else 1
            return np.roll(grad, shift, axis=axis)
        if j == 7:
            return self.diffusion(grad)
        if j == 8:
            b, _, _, _, c = grad.shape
            flat_grad = grad.reshape(b, -1, c)
            result = np.zeros_like(flat_grad)
            for k in range(b):
                np.add.at(result[k], self.inward_idx, flat_grad[k])
            return result.reshape(grad.shape)
        raise ValueError("Unknown agent")


@dataclass
class Configuration:
    n_tiles: int = 8
    edge: int = 8
    hidden: int = 10
    steps: int = 150
    active_tiles: int = 4
    learning_rate: float = 1.0
    memory_gain: float = 0.05
    memory_decay: float = 0.94
    inward_ratio: float = 0.65
    seed: int = 7

    def validate(self):
        if not 1 <= self.n_tiles <= 1000:
            raise ValueError("n_tiles must be 1..1000")
        if self.edge < 4 or self.edge % 2:
            raise ValueError("edge must be even and >= 4")
        if self.hidden < 1:
            raise ValueError("hidden must be positive")
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if not 1 <= self.active_tiles <= self.n_tiles:
            raise ValueError("active_tiles must be 1..n_tiles")
        if self.learning_rate <= 0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if not 0 <= self.memory_gain <= 1:
            raise ValueError("memory_gain must be in [0,1]")
        if not 0 <= self.memory_decay < 1:
            raise ValueError("memory_decay must be in [0,1)")


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return v
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    m = max(1e-12, float(np.max(np.abs(v))))
    return v/m


def _sample_1d(y: np.ndarray, count: int) -> np.ndarray:
    y = np.asarray(y, np.float64).ravel()
    if len(y) == 0:
        return np.zeros(count)
    if len(y) == 1:
        return np.repeat(y[0], count)
    return np.interp(np.linspace(0, len(y)-1, count), np.arange(len(y)), y)


def text_volume(text: str, n: int) -> np.ndarray:
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float64)
    out = np.zeros(n**3, dtype=np.float64)
    k = min(len(raw), len(out))
    out[:k] = (raw[:k]-127.5)/127.5
    return out.reshape(n,n,n)


def image_volume(path: str, n: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to load images: pip install pillow") from exc
    with Image.open(path) as img:
        gray = img.convert("L").resize((n,n))
        frame = np.asarray(gray, dtype=np.float64)/127.5-1.0
    return np.repeat(frame[None,:,:], n, axis=0)


def audio_volume(path: str, n: int) -> np.ndarray:
    """Decode uncompressed PCM WAV (8/16/24/32-bit) to a sampled 3D field."""
    with wave.open(path, "rb") as fp:
        if fp.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV audio is supported")
        width, channels, count = fp.getsampwidth(), fp.getnchannels(), fp.getnframes()
        raw = fp.readframes(count)
    if channels < 1 or width not in (1,2,3,4):
        raise ValueError("Unsupported WAV layout")
    if width == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64)-128)/128
    elif width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float64)/(2**15)
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float64)/(2**31)
    else:
        t = np.frombuffer(raw, dtype=np.uint8)
        if t.size % 3:
            raise ValueError("Malformed 24-bit WAV")
        q = t.reshape(-1,3).astype(np.int32)
        vals = q[:,0] | q[:,1]<<8 | q[:,2]<<16
        vals = np.where(vals >= 2**23, vals - 2**24, vals)
        a = vals.astype(np.float64)/(2**23)
    if a.size % channels:
        raise ValueError("Incomplete WAV frame")
    mono = a.reshape(-1, channels).mean(axis=1)
    return _sample_1d(mono, n**3).reshape(n,n,n)


def video_volume(path: str, n: int) -> np.ndarray:
    """Decode up to n evenly sampled frames into (time,height,width) voxel grid."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to decode video: pip install opencv-python") from exc
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    try:
        if total > 0:
            desired = np.rint(np.linspace(0, total-1, n)).astype(int)
            for idx in desired:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out.append(cv2.resize(gray, (n,n)).astype(np.float64)/127.5-1)
        else:
            for _ in range(n):
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out.append(cv2.resize(gray, (n,n)).astype(np.float64)/127.5-1)
    finally:
        cap.release()
    if not out:
        raise ValueError("Video produced no decoded frames")
    indices = np.rint(np.linspace(0, len(out)-1, n)).astype(int)
    return np.asarray(out)[indices]


def demo_volumes(n: int) -> np.ndarray:
    """Four distinguishable fields: text pulses, image geometry, audio, motion."""
    q = np.indices((n,n,n), dtype=np.float64)
    z,y,x = q/(n-1)
    text = np.where(((q[1].astype(int)//2+q[2].astype(int)//2)%3 == 0) &
                    (q[0].astype(int)%4 < 2), 0.8, -0.2)
    image = 2*np.exp(-((x-.38)**2 + (y-.65)**2)/.065)-0.65
    image = image + 0.10*np.sin(2*np.pi*z)
    audio = np.sin(2*np.pi*(2.0*x+1.1*y+0.5*z))
    video = 2*np.exp(-((x-(.15+.7*z))**2+(y-.5)**2)/.055)-0.65
    return np.clip(np.stack([text,image,audio,video],axis=-1),-1,1)


def make_dataset(cfg: Configuration, text=None, image=None, audio=None, video=None):
    n = cfg.edge
    demo = demo_volumes(n)
    supplied = any(v is not None for v in (text, image, audio, video))
    if supplied:
        base = np.zeros_like(demo)
        if text is not None:
            base[...,0] = text_volume(text,n)
        if image is not None:
            base[...,1] = image_volume(image,n)
        if audio is not None:
            base[...,2] = audio_volume(audio,n)
        if video is not None:
            base[...,3] = video_volume(video,n)
    else:
        base = demo
    fields = []
    for i in range(cfg.n_tiles):
        # Virtual regions are independent samples of the input representation.
        # These cyclic variants do NOT map an entire media file losslessly.
        shifted = np.roll(base, (i%3,(i//3)%3,(i//5)%3), axis=(0,1,2))
        fields.append(shifted)
    return np.stack(fields), ("supplied_media" if supplied else "synthetic_four_modalities")


class MoagiSwarm3D:
    def __init__(self, cfg: Configuration):
        cfg.validate()
        self.cfg = cfg
        self.geometry = Geometry3D(cfg.edge//2, cfg.inward_ratio)
        self.rng = np.random.default_rng(cfg.seed)
        c, d = len(MODALITIES), cfg.hidden
        self.W_e = self.rng.normal(0, 0.35, (c,d))
        self.b_e = np.zeros(d)
        self.W_d = self.rng.normal(0, 0.35, (d,c))
        self.b_d = np.zeros(c)
        self.logits = np.array([3.6, -0.8, -0.8, -0.8, -0.8, -0.8,
                                -0.8, 0.5, -0.8], dtype=np.float64)
        l = cfg.edge//2
        self.omega = np.zeros((cfg.n_tiles,l,l,l,d), dtype=np.float64)
        self.last_visit = np.zeros(cfg.n_tiles, dtype=int)
        self.accepted = 0
        self.rejected = 0

    def parameters(self):
        return {k:getattr(self,k) for k in ("W_e","b_e","W_d","b_d","logits")}

    def forward(self, x, omega, with_cache=False):
        p = avg_pool2(x)
        h = np.tanh(p @ self.W_e + self.b_e + self.cfg.memory_gain * omega)
        a = softmax(self.logits)
        agents = [self.geometry.apply(j,h) for j in range(len(AGENT_NAMES))]
        z = sum(a[j]*agents[j] for j in range(len(AGENT_NAMES)))
        y_low = z @ self.W_d + self.b_d
        y = upsample2(y_low)
        mse = float(np.mean((y-x)**2))
        if not with_cache:
            return mse, y, z
        return mse, y, z, (p,h,a,agents,y_low)

    def gradients(self, x, omega):
        loss, y, z, (p,h,a,agents,y_low) = self.forward(x, omega, True)
        # Average pooling is orthogonal to the space of 2x upsampled outputs:
        # d(mean((upsample(y_low)-x)^2))/dy_low = 2*(y_low-p)/size(p)
        d_y = 2*(y_low-p) / p.size
        dW_d = z.reshape(-1,self.cfg.hidden).T @ d_y.reshape(-1,4)
        db_d = d_y.sum(axis=(0,1,2,3))
        d_z = d_y @ self.W_d.T
        d_logits = np.zeros(len(AGENT_NAMES))
        d_h = np.zeros_like(h)
        for j in range(len(AGENT_NAMES)):
            # Exact softmax Jacobian vector product; each agent is differentiable.
            d_logits[j] = a[j]*np.sum(d_z*(agents[j]-z))
            d_h += a[j] * self.geometry.adjoint(j, d_z)
        d_pre = d_h * (1-h*h)
        dW_e = p.reshape(-1,4).T @ d_pre.reshape(-1,self.cfg.hidden)
        db_e = d_pre.sum(axis=(0,1,2,3))
        grad = dict(W_e=dW_e,b_e=db_e,W_d=dW_d,b_d=db_d,logits=d_logits)
        return loss, grad

    def ctr_train_step(self, x, idx):
        """Transactional parameter update: rollback if evaluated loss rises."""
        xs = x[idx]
        om = self.omega[idx]
        before, grads = self.gradients(xs, om)
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        scale = min(1.0, 1.0/(norm+1e-12))
        old = {k:v.copy() for k,v in self.parameters().items()}
        accepted = False
        after = before
        for retry in range(12):
            rate = self.cfg.learning_rate * (0.5**retry) * scale
            for name, par in self.parameters().items():
                par[...] = old[name] - rate*grads[name]
            after, _, _ = self.forward(xs, om)
            if np.isfinite(after) and after <= before + 1e-12:
                accepted = True
                break
        if not accepted:
            for name, par in self.parameters().items():
                par[...] = old[name]
            after = before
            self.rejected += 1
        else:
            self.accepted += 1
            # Memory updates only after a verified weight update. This memory
            # change may affect subsequent losses; no global monotonicity claim.
            _, _, _, cache = self.forward(xs, om, True)
            h = cache[1]
            beta = self.cfg.memory_decay
            self.omega[idx] = beta*self.omega[idx] + (1-beta)*h
        return before, after, accepted, norm

    def metrics(self, x):
        mse, y, z = self.forward(x, self.omega)
        px = avg_pool2(x)
        y_low = avg_pool2(y)
        a = softmax(self.logits)
        return {"mse":mse, "coarse_mse":float(np.mean((y_low-px)**2)),
                "agent_entropy_nats":float(-np.sum(a*np.log(a+1e-30))),
                "agent_weights":{AGENT_NAMES[i]:float(a[i]) for i in range(len(a))}}, y,z

    def train(self, x, interval=25):
        history=[]
        m,_,_ = self.metrics(x)
        initial=m["mse"]
        for step in range(1,self.cfg.steps+1):
            # Scheduler: prioritize high residuals plus bounded recency fairness.
            individual = np.array([self.forward(x[i:i+1],self.omega[i:i+1])[0]
                                   for i in range(len(x))])
            score = individual + 0.015*np.minimum((step-self.last_visit)/20.0,1.0)
            idx = np.argsort(-score, kind="stable")[:self.cfg.active_tiles]
            before, after, ok, grad_norm = self.ctr_train_step(x,idx)
            self.last_visit[idx]=step
            m,_,_ = self.metrics(x)
            record = {"step":step, "mse":m["mse"],
                      "coarse_mse":m["coarse_mse"],
                      "selected_mse_before":before, "selected_mse_after":after,
                      "accepted":int(ok), "gradient_norm":grad_norm,
                      "entropy":m["agent_entropy_nats"]}
            history.append(record)
            if interval and (step==1 or step%interval==0 or step==self.cfg.steps):
                status = "ACCEPT" if ok else "REJECT"
                print(f"step {step:4d} | full MSE {m['mse']:.8f} | "
                      f"coarse MSE {m['coarse_mse']:.8f} | CTR {status}")
        return initial, history


def check_gradients():
    """Directional finite-difference check for encoder, decoder and gating."""
    cfg=Configuration(n_tiles=2,edge=4,hidden=3,steps=0,active_tiles=2,seed=21)
    model=MoagiSwarm3D(cfg)
    x=np.random.default_rng(1).normal(size=(2,4,4,4,4))*.3
    omega=np.random.default_rng(2).normal(size=model.omega.shape)*.05
    _, grad=model.gradients(x,omega)
    details={}
    for name, index in (("W_e",(2,1)),("W_d",(1,2)),("logits",(8,))):
        p=model.parameters()[name]
        saved=float(p[index]); eps=1e-5
        p[index]=saved+eps; plus=model.forward(x,omega)[0]
        p[index]=saved-eps; minus=model.forward(x,omega)[0]
        p[index]=saved
        finite=(plus-minus)/(2*eps)
        analytic=float(grad[name][index])
        relative=abs(finite-analytic)/max(1e-8,abs(finite),abs(analytic))
        details[name]={"finite_difference":finite,"analytic":analytic,"relative_error":relative}
        if relative>=1e-5:
            raise AssertionError(f"gradient check failed: {name}: {details[name]}")
    return details


def check_geometry():
    rng=np.random.default_rng(3)
    geom=Geometry3D(4)
    u=rng.normal(size=(2,4,4,4,3))
    v=rng.normal(size=u.shape)
    result={}
    for j,name in enumerate(AGENT_NAMES):
        left=np.sum(geom.apply(j,u)*v)
        right=np.sum(u*geom.adjoint(j,v))
        err=abs(left-right)
        if err>1e-10:
            raise AssertionError(f"Adjoint failed for {name}: {err}")
        result[name]=float(err)
    assert VOLUME_EXPONENT == 18_000_000
    assert abs(softmax(np.array([1.0,2.0,3.0])).sum()-1)<1e-12
    assert VirtualTile("0001","0002","0003").region_exponent()==17_999_988
    return result


def save_plot(x, y, z, filename):
    """3D scatter of a single active tile's decoded residual magnitude."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n=x.shape[1]
    residual=np.linalg.norm(x[0]-y[0],axis=-1)
    cutoff=np.quantile(residual,.70)
    coords=np.argwhere(residual>=cutoff)
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(111,projection="3d")
    value=residual[tuple(coords.T)]
    scatter=ax.scatter(coords[:,2],coords[:,1],coords[:,0],c=value,
                       cmap="viridis",s=14+70*value/(max(value.max(),1e-12)))
    fig.colorbar(scatter,ax=ax,shrink=.65,label="Euclidean reconstruction residual")
    ax.set(xlabel="x voxel",ylabel="y voxel",zlabel="z voxel",
           title="DM3D: physical active tile — 3D residual geometry")
    ax.set(xlim=(0,n-1),ylim=(0,n-1),zlim=(0,n-1))
    fig.tight_layout()
    fig.savefig(filename,dpi=160)
    plt.close(fig)


# ==== Recursive CTR-verified orchestration extension ====

from itertools import permutations
from typing import Any

# A septillion is 10^24 in the short scale. Do not construct 10**(24*10**24).
VIRTUAL_RECURSION_BUDGET = "(10^24)^(10^24) = 10^(24*10^24) (symbolic only)"


@dataclass
class RecursiveConfig:
    generations: int = 12
    inner_steps: int = 4
    latent_steps: int = 14
    latent_mix: float = 0.5
    ridge: float = 0.03
    tolerance: float = 1e-9
    patience: int = 6
    memory_feedback: float = 0.1
    permute_routing: bool = True
    maximum_bytes: int = 256 * 1024 * 1024

    def validate(self, cfg: Configuration):
        cfg.validate()
        if not 1 <= self.generations <= 100_000:
            raise ValueError("generations must be between 1 and 100000")
        if not 1 <= self.inner_steps <= 1000:
            raise ValueError("inner_steps must be between 1 and 1000")
        if not 1 <= self.latent_steps <= 1000:
            raise ValueError("latent_steps must be between 1 and 1000")
        if not 0 < self.latent_mix <= 1:
            raise ValueError("latent_mix must be in (0, 1]")
        if not self.ridge > 0 or not math.isfinite(self.ridge):
            raise ValueError("ridge must be positive and finite")
        if not 0 < self.tolerance < 1 or not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and in (0,1)")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not 0 <= self.memory_feedback <= 1:
            raise ValueError("memory_feedback must be in [0,1]")
        if cfg.edge > 48 or cfg.hidden > 128 or cfg.n_tiles > 64:
            raise ValueError("safety limits: edge<=48, hidden<=128, tiles<=64")
        # An illustrative bound for peak workspace including up to nine agents,
        # optimizer checkpoints, inputs, memory and reconstruction buffers.
        estimate = cfg.n_tiles * (cfg.edge**3*4 +
                  18*(cfg.edge//2)**3*cfg.hidden) * 8
        if estimate > self.maximum_bytes:
            raise ValueError(f"estimated workspace {estimate:,} bytes exceeds cap")


def _copy_state(model: MoagiSwarm3D) -> dict[str, Any]:
    return {
        "params": {name: p.copy() for name, p in model.parameters().items()},
        "omega": model.omega.copy(),
        "last_visit": model.last_visit.copy(),
        "accepted": model.accepted,
        "rejected": model.rejected,
        "learning_rate": model.cfg.learning_rate,
    }


def _restore_state(model: MoagiSwarm3D, snap: dict[str, Any]) -> None:
    for name, values in snap["params"].items():
        model.parameters()[name][...] = values
    model.omega[...] = snap["omega"]
    model.last_visit[...] = snap["last_visit"]
    model.accepted = snap["accepted"]
    model.rejected = snap["rejected"]
    model.cfg.learning_rate = snap["learning_rate"]


def _routing_permutation(logits: np.ndarray, permutation: tuple[int, int, int], reverse=False):
    """Reassign x/y/z directional agent weights; preserve other agents.

    This is a real finite architecture/routing search, not enumeration of
    any huge virtual permutation domain.
    """
    out = logits.copy()
    for dst_axis, src_axis in enumerate(permutation):
        pair = logits[1+2*src_axis:3+2*src_axis]
        out[1+2*dst_axis:3+2*dst_axis] = pair[::-1] if reverse else pair
    return out


class DrMoagiRecursiveIntelligence3D:
    """3D encoder/decoder with verified, finite recursive self-refinement.

    Outer loop: sparse high-residual tile selection; trial learning rates;
    best-of routing permutations; transactional global CTR; geometric latent
    fixed-point solve; verified feedback from latent refinement into memory.
    """
    def __init__(self, cfg: Configuration, recursion: RecursiveConfig):
        recursion.validate(cfg)
        self.net = MoagiSwarm3D(cfg)
        self.cfg = cfg
        self.rec = recursion
        self.history: list[dict[str, Any]] = []
        self.latent_history: list[dict[str, float | int]] = []
        self.inward_stats: dict[str, Any] = {}
        self.global_rejections = 0
        self.routing_changes = 0
        self.memory_accepts = 0

    @property
    def virtual_geometry(self):
        return {"axis": "10^6000000", "sites": "10^18000000",
                "virtual_recursive_permutations": VIRTUAL_RECURSION_BUDGET,
                "materialized_sites": self.cfg.n_tiles*self.cfg.edge**3}

    def encode(self, x: np.ndarray) -> np.ndarray:
        self._assert_input(x)
        return self.net.forward(x, self.net.omega)[2]

    def decode(self, z: np.ndarray) -> np.ndarray:
        want=(self.cfg.n_tiles,self.cfg.edge//2,self.cfg.edge//2,
              self.cfg.edge//2,self.cfg.hidden)
        if z.shape != want or not np.isfinite(z).all():
            raise ValueError(f"latent shape must be {want} and finite")
        return upsample2(z @ self.net.W_d + self.net.b_d)

    def _assert_input(self, x):
        expected=(self.cfg.n_tiles,self.cfg.edge,self.cfg.edge,self.cfg.edge,4)
        if x.shape != expected or not np.isfinite(x).all():
            raise ValueError(f"input shape must be {expected} and finite")

    def reconstruction_loss(self, x, z):
        return float(np.mean((x-self.decode(z))**2))

    def _fixed_point_target(self, x: np.ndarray, z_base: np.ndarray) -> np.ndarray:
        """Exact small-matrix ridge fixed point for the linear decoder.

        argmin_z (1/2)||z W_d+b_d-P(X)||^2 + (rho/2)||z-z_base||^2
        per latent cell. This is *not* a solution of full ANN optimization.
        """
        p=avg_pool2(x)
        wd=self.net.W_d
        a=wd@wd.T + self.rec.ridge*np.eye(self.cfg.hidden)
        rhs=(p-self.net.b_d)@wd.T + self.rec.ridge*z_base
        flat_rhs=rhs.reshape(-1, self.cfg.hidden)
        z_target=np.linalg.solve(a,flat_rhs.T).T.reshape(z_base.shape)
        if not np.isfinite(z_target).all():
            raise ArithmeticError("nonfinite fixed-point target")
        return z_target

    def inward_refine(self, x: np.ndarray, base: np.ndarray | None=None,
                      generation: int=0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        self._assert_input(x)
        z_base=self.encode(x) if base is None else base
        target=self._fixed_point_target(x,z_base)
        z=z_base.copy()
        loss=self.reconstruction_loss(x,z)
        initial=loss
        records=[]
        for k in range(1,self.rec.latent_steps+1):
            previous=loss
            candidate=z + self.rec.latent_mix*(target-z)
            new_loss=self.reconstruction_loss(x,candidate)
            # CTR transaction: observe real 3D decoded error, reject increase.
            if not math.isfinite(new_loss) or new_loss > loss+1e-12:
                break
            z=candidate
            loss=new_loss
            fixed_residual=float(np.sqrt(np.mean((target-z)**2)))
            records.append({"generation": generation, "iteration": k,
                            "mse": loss, "fixed_point_rms": fixed_residual,
                            "improvement": previous-loss})
            if fixed_residual < self.rec.tolerance:
                break
            if k>=3 and (previous-loss)<self.rec.tolerance*max(1.0,initial):
                break
        return z,self.decode(z),records

    def _schedule(self,x,gen):
        # Real reconstruction residual determines scheduling priority.
        residual=np.array([self.net.forward(x[i:i+1],self.net.omega[i:i+1])[0]
                           for i in range(self.cfg.n_tiles)])
        fairness=0.01*np.minimum((gen-self.net.last_visit)/10,1.0)
        score=residual+fairness
        return np.argsort(-score,kind="stable")[:self.cfg.active_tiles]

    def _search_routing(self,x):
        """Finite grid over axis permutations and chirality; full-data CTR."""
        before=self.net.metrics(x)[0]["mse"]
        old=self.net.logits.copy()
        best=old.copy()
        best_loss=before
        if self.rec.permute_routing:
            for p in permutations((0,1,2)):
                for reverse in (False, True):
                    trial=_routing_permutation(old,p,reverse)
                    self.net.logits[...] =trial
                    score=self.net.metrics(x)[0]["mse"]
                    if math.isfinite(score) and score<best_loss-1e-13:
                        best,best_loss=trial.copy(),score
        self.net.logits[...] = best
        if best_loss<before-1e-13:
            self.routing_changes+=1
        return best_loss

    def _feedback_memory(self,x,z_ref):
        """CTR-gated inward feedback; evaluated on identical input volume."""
        if self.rec.memory_feedback==0:
            return False
        previous=self.net.metrics(x)[0]["mse"]
        old=self.net.omega.copy()
        zbase=self.encode(x)
        correction=z_ref-zbase
        # Blend part of the verified latent error correction into Omega. Note
        # that Omega acts on pre-activations, so this is a heuristic proposal.
        for rate in (self.rec.memory_feedback, self.rec.memory_feedback*0.25):
            self.net.omega[...] = np.clip(old+rate*correction,-2,2)
            trial=self.net.metrics(x)[0]["mse"]
            if math.isfinite(trial) and trial<previous-1e-13:
                self.memory_accepts+=1
                return True
        self.net.omega[...] = old
        return False

    def fit(self,x: np.ndarray, verbose=True):
        self._assert_input(x)
        start_metrics=self.net.metrics(x)[0]
        initial_base=start_metrics["mse"]
        best=initial_base
        idle=0
        for generation in range(1,self.rec.generations+1):
            snapshot=_copy_state(self.net)
            selection=self._schedule(x,generation)
            best_snapshot=None
            best_trial=best
            best_factor=None
            for factor in (0.5,1.0,1.5):
                _restore_state(self.net,snapshot)
                self.net.cfg.learning_rate=snapshot["learning_rate"]*factor
                for _ in range(self.rec.inner_steps):
                    self.net.ctr_train_step(x,selection)
                loss=self.net.metrics(x)[0]["mse"]
                if math.isfinite(loss) and loss<best_trial-1e-13:
                    best_trial=loss
                    best_factor=factor
                    best_snapshot=_copy_state(self.net)
            if best_snapshot is None:
                _restore_state(self.net,snapshot)
                self.global_rejections+=1
                accepted=False
            else:
                _restore_state(self.net,best_snapshot)
                # Preserve base LR, tune trial steps without shifting default.
                self.net.cfg.learning_rate=snapshot["learning_rate"]
                accepted=True
                self.net.last_visit[selection]=generation
            routed=self._search_routing(x)
            zbase=self.encode(x)
            zref, yref, latent_records=self.inward_refine(x,zbase,generation)
            self.latent_history.extend(latent_records)
            fed_back=self._feedback_memory(x,zref)
            base_loss=self.net.metrics(x)[0]["mse"]
            # Model-level guarantee against the previous accepted full score.
            if base_loss>best+1e-12 or not math.isfinite(base_loss):
                _restore_state(self.net,snapshot)
                self.global_rejections+=1
                accepted=False
                fed_back=False
                zbase=self.encode(x)
                zref,yref,latent_records=self.inward_refine(x,zbase,generation)
                base_loss=self.net.metrics(x)[0]["mse"]
            delta=best-base_loss
            best=min(best,base_loss)
            if delta <= self.rec.tolerance*max(1,best):
                idle+=1
            else:
                idle=0
            rec={"generation":generation,"base_mse":base_loss,
                 "refined_mse":self.reconstruction_loss(x,zref),
                 "latent_iterations":len(latent_records),
                 "fixed_point_rms":(latent_records[-1]["fixed_point_rms"]
                                    if latent_records else 0.),
                 "selected_tiles":",".join(map(str,selection)),
                 "accepted_learning":int(accepted),
                 "learning_rate_factor":best_factor if best_factor else 0.0,
                 "feedback_accepted":int(fed_back),
                 "agent_entropy":self.net.metrics(x)[0]["agent_entropy_nats"]}
            self.history.append(rec)
            if verbose:
                print(f"generation {generation:3} | base {base_loss:.9f} | "
                      f"inward {rec['refined_mse']:.9f} | "
                      f"steps {rec['latent_iterations']:2} | "
                      f"CTR {'ACCEPT' if accepted else 'HOLD'}")
            if idle>=self.rec.patience:
                break
        zbase=self.encode(x)
        zref, yref, stats=self.inward_refine(x,zbase,len(self.history)+1)
        self.inward_stats={"initial_base_mse":initial_base,
                           "final_base_mse":self.net.metrics(x)[0]["mse"],
                           "final_inward_mse":self.reconstruction_loss(x,zref),
                           "latent_fixed_point_rms":stats[-1]["fixed_point_rms"] if stats else 0.,
                           "final_latent_iterations":len(stats),
                           "total_generations":len(self.history),
                           "routing_changes":self.routing_changes,
                           "memory_feedback_accepts":self.memory_accepts,
                           "global_rejections":self.global_rejections}
        return zref,yref

    def save(self,path: str | Path,inputs: np.ndarray,refined: np.ndarray):
        self._assert_input(inputs)
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,inputs=inputs,refined_latent=refined,
                            refined_reconstruction=self.decode(refined),
                            encoder=self.net.W_e,encoder_bias=self.net.b_e,
                            decoder=self.net.W_d,decoder_bias=self.net.b_d,
                            agent_logits=self.net.logits,omega=self.net.omega,
                            last_visit=self.net.last_visit,
                            config=json.dumps(vars(self.cfg)),
                            recursion_config=json.dumps(vars(self.rec)),
                            metrics=json.dumps(self.inward_stats))

    @classmethod
    def load(cls,path: str | Path):
        with np.load(path,allow_pickle=False) as d:
            cfg=Configuration(**json.loads(str(d["config"].item())))
            rec=RecursiveConfig(**json.loads(str(d["recursion_config"].item())))
            model=cls(cfg,rec)
            fields={"W_e":"encoder","b_e":"encoder_bias",
                    "W_d":"decoder","b_d":"decoder_bias",
                    "logits":"agent_logits","omega":"omega",
                    "last_visit":"last_visit"}
            for attr,key in fields.items():
                value=d[key]
                old=getattr(model.net,attr)
                if value.shape!=old.shape or not np.isfinite(value).all():
                    raise ValueError(f"checkpoint field {key} invalid")
                old[...] = value
            model.inward_stats=json.loads(str(d["metrics"].item()))
        return model


def save_recursive_visuals(x,y,history,out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 3D physical residual geometry; color is absolute residual magnitude.
    save_plot(x,y,None,out/"3d_inward_residual.png")
    if history:
        fig,ax=plt.subplots(figsize=(9,5))
        gx=[r["generation"] for r in history]
        ax.plot(gx,[r["base_mse"] for r in history],marker="o",label="encoder-decoder MSE")
        ax.plot(gx,[r["refined_mse"] for r in history],marker="s",label="inward refined MSE")
        ax.set(xlabel="outer recursive generation",ylabel="full-volume reconstruction MSE",
               title="CTR verified recursive 3D autoencoding")
        ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/"recursion_convergence.png",dpi=150)
        plt.close(fig)


def inward_main(argv=None):
    parser=argparse.ArgumentParser(description="Dr Moagi CTR-verified recursive 3D autoencoder")
    parser.add_argument("--generations",type=int,default=12)
    parser.add_argument("--inner-steps",type=int,default=4)
    parser.add_argument("--latent-steps",type=int,default=14)
    parser.add_argument("--mix",type=float,default=0.5)
    parser.add_argument("--ridge",type=float,default=0.03)
    parser.add_argument("--patience",type=int,default=6)
    parser.add_argument("--tiles",type=int,default=8)
    parser.add_argument("--active",type=int,default=4)
    parser.add_argument("--edge",type=int,default=8)
    parser.add_argument("--hidden",type=int,default=10)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--lr",type=float,default=1.)
    parser.add_argument("--text")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--video")
    parser.add_argument("--out",type=str,default="dm3d_recursive_output")
    parser.add_argument("--load",type=str,help="Load existing checkpoint and reconstruct saved inputs")
    parser.add_argument("--no-plot",action="store_true")
    parser.add_argument("--no-permutations",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    a=parser.parse_args(argv)
    if a.self_test:
        print(json.dumps({"geometry":check_geometry(),"gradients":check_gradients()},indent=2))
        return 0
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if a.load:
        model=DrMoagiRecursiveIntelligence3D.load(a.load)
        with np.load(a.load,allow_pickle=False) as saved:
            x=saved["inputs"]
        z,decoded,curve=model.inward_refine(x)
        mode="loaded_checkpoint"
        initial=model.net.metrics(x)[0]["mse"]
    else:
        cfg=Configuration(n_tiles=a.tiles,edge=a.edge,hidden=a.hidden,
                          steps=0,active_tiles=a.active,seed=a.seed,learning_rate=a.lr)
        rec=RecursiveConfig(generations=a.generations,inner_steps=a.inner_steps,
                            latent_steps=a.latent_steps,latent_mix=a.mix,
                            ridge=a.ridge,patience=a.patience,
                            permute_routing=not a.no_permutations)
        model=DrMoagiRecursiveIntelligence3D(cfg,rec)
        x,mode=make_dataset(cfg,a.text,a.image,a.audio,a.video)
        z,decoded=model.fit(x)
        initial=model.inward_stats["initial_base_mse"]
    model.save(out/"checkpoint.npz",x,z)
    base_mse=model.net.metrics(x)[0]["mse"]
    final_mse=float(np.mean((decoded-x)**2))
    irreducible_floor=float(np.mean((x-upsample2(avg_pool2(x)))**2))
    scalar_compression_ratio=(x.size/z.size)
    summary={"system":"Dr Moagi 3D inward recursive autoencoding NumPy reference",
             "virtual_geometry":model.virtual_geometry,
             "modalities":list(MODALITIES),"dataset":mode,
             "shape":list(x.shape),"initial_base_mse":initial,
             "final_base_mse":base_mse,"final_refined_mse":final_mse,
             "pooling_mse_lower_bound":irreducible_floor,
             "gap_to_pooling_floor":final_mse-irreducible_floor,
             "scalar_compression_ratio":scalar_compression_ratio,
             "improvement_base_fraction":(initial-base_mse)/initial if initial else 0.,
             "improvement_refined_fraction":(initial-final_mse)/initial if initial else 0.,
             "recursion":model.inward_stats,
             "agent_weights":model.net.metrics(x)[0]["agent_weights"],
             "precision_note":"Lossy 2x pooling; MSE cannot generally reach zero",
             "capability_note":"Computes finite 3D tiles; no full astronomical enumeration or general AI"}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    if model.history:
        with (out/"generations.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(model.history[0]));w.writeheader();w.writerows(model.history)
    if model.latent_history:
        with (out/"latent_iterations.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(model.latent_history[0]));w.writeheader();w.writerows(model.latent_history)
    if not a.no_plot:
        save_recursive_visuals(x,decoded,model.history,out)
    print("\n"+json.dumps({"initial_base_mse":initial,"final_base_mse":base_mse,
                           "final_inward_mse":final_mse,
                           "generations":len(model.history),"out":str(out)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(inward_main())
