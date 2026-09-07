from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ProofStatus(str, Enum):
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"


class Opcode(IntEnum):
    NOP = 0x00
    AXIOM_LOAD = 0x01
    GEOM_INIT = 0x10
    PERMEATE = 0x20
    FOLD_INWARD = 0x30
    VERIFY = 0x40
    SELF_EXEC = 0x50
    MDL_SCORE = 0x60
    TIME_BLOCK = 0x70
    OPTIMIZE = 0x80
    HALT = 0xF0


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    a: int = 0
    b: int = 0
    c: int = 0
    length: int = 4


class BytecodeCodec:
    WIDTH = 4

    @staticmethod
    def encode(instructions: Sequence[Instruction]) -> bytearray:
        out = bytearray()
        for ins in instructions:
            out.extend((int(ins.opcode), ins.a & 0xFF, ins.b & 0xFF, ins.c & 0xFF))
        return out

    @staticmethod
    def decode(program: Sequence[int], pc: int) -> Instruction:
        if pc < 0 or pc + BytecodeCodec.WIDTH > len(program):
            raise ValueError(f"pc={pc} outside program length={len(program)}")
        raw = program[pc : pc + BytecodeCodec.WIDTH]
        try:
            opcode = Opcode(raw[0])
        except ValueError as exc:
            raise ValueError(f"unknown opcode 0x{raw[0]:02X} at pc={pc}") from exc
        return Instruction(opcode, int(raw[1]), int(raw[2]), int(raw[3]))

    @staticmethod
    def decode_all(program: Sequence[int]) -> List[Instruction]:
        if len(program) % BytecodeCodec.WIDTH:
            raise ValueError("program length must be a multiple of 4 bytes")
        return [BytecodeCodec.decode(program, pc) for pc in range(0, len(program), 4)]

    @staticmethod
    def semantic_signature(program: Sequence[int]) -> Tuple[Tuple[int, int, int, int], ...]:
        return tuple(
            (int(ins.opcode), ins.a, ins.b, ins.c)
            for ins in BytecodeCodec.decode_all(program)
            if ins.opcode != Opcode.NOP
        )


class Clifford4:
    """Minimal Cl(1,3) geometric product over 16 basis blades."""

    def __init__(self, signature: Tuple[int, int, int, int] = (1, -1, -1, -1)):
        self.signature = signature

    def blade_product(self, a_mask: int, b_mask: int) -> Tuple[float, int]:
        sign = 1.0
        for i in range(4):
            if (a_mask >> i) & 1:
                lower_in_b = b_mask & ((1 << i) - 1)
                if lower_in_b.bit_count() % 2:
                    sign *= -1.0
        common = a_mask & b_mask
        for i in range(4):
            if (common >> i) & 1:
                sign *= self.signature[i]
        return sign, a_mask ^ b_mask

    def geometric_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != (16,) or b.shape != (16,):
            raise ValueError("Clifford multivectors must have 16 coefficients")
        out = np.zeros(16, dtype=float)
        for i, ai in enumerate(a):
            if ai == 0.0:
                continue
            for j, bj in enumerate(b):
                if bj == 0.0:
                    continue
                sign, mask = self.blade_product(i, j)
                out[mask] += sign * ai * bj
        return out

    def reverse(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for mask in range(16):
            grade = mask.bit_count()
            out[mask] *= (-1) ** (grade * (grade - 1) // 2)
        return out

    def rotor_action(self, rotor: np.ndarray, psi: np.ndarray) -> np.ndarray:
        return self.geometric_product(
            self.geometric_product(rotor, psi), self.reverse(rotor)
        )


@dataclass
class Theta:
    w_enc: np.ndarray
    w_dec: np.ndarray
    epoch: int = 0

    def clone(self) -> "Theta":
        return Theta(self.w_enc.copy(), self.w_dec.copy(), self.epoch)

    def parameter_count(self) -> int:
        return int(self.w_enc.size + self.w_dec.size)


@dataclass
class MachineMemory:
    registers: Dict[str, float] = field(default_factory=dict)
    trace: List[Dict[str, object]] = field(default_factory=list)
    rollback_count: int = 0

    def clone(self) -> "MachineMemory":
        return MachineMemory(dict(self.registers), copy.deepcopy(self.trace), self.rollback_count)


@dataclass
class ProofBundle:
    status: ProofStatus = ProofStatus.UNKNOWN
    checks: Dict[str, Optional[bool]] = field(default_factory=dict)
    certificates: List[str] = field(default_factory=list)

    def clone(self) -> "ProofBundle":
        return ProofBundle(self.status, dict(self.checks), list(self.certificates))


@dataclass
class HGAState:
    pc: int
    bytecode: bytearray
    gamma: set[str]
    clifford_metric: np.ndarray
    psi: np.ndarray
    z: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    theta: Theta
    proof: ProofBundle
    memory: MachineMemory
    cycle: int = 0
    halted: bool = False
    last_loss: float = math.inf
    last_mdl: float = math.inf

    def clone(self) -> "HGAState":
        return HGAState(
            pc=self.pc,
            bytecode=bytearray(self.bytecode),
            gamma=set(self.gamma),
            clifford_metric=self.clifford_metric.copy(),
            psi=self.psi.copy(),
            z=self.z.copy(),
            v=self.v.copy(),
            omega=self.omega.copy(),
            theta=self.theta.clone(),
            proof=self.proof.clone(),
            memory=self.memory.clone(),
            cycle=self.cycle,
            halted=self.halted,
            last_loss=self.last_loss,
            last_mdl=self.last_mdl,
        )


@dataclass
class HGAConfig:
    psi_dim: int = 16
    latent_dim: int = 8
    dt: float = 0.08
    damping: float = 0.35
    mass: float = 1.0
    learning_rate: float = 0.02
    rho_memory: float = 0.85
    error_gain: float = 0.30
    memory_gain: float = 0.08
    lambda_rec: float = 1.0
    lambda_cycle: float = 0.15
    lambda_geo: float = 0.01
    lambda_mdl: float = 1e-5
    rec_epsilon: float = 0.14
    unknown_rec_limit: float = 0.75
    energy_max: float = 50.0
    hard_energy_max: float = 1e6
    contraction_q: float = 0.92
    mdl_beta: float = 100.0
    mdl_gamma: float = 0.02
    mdl_mu: float = 0.01
    seed: int = 7


class HGAVM:
    AXIOMS = {0: "ZFC", 1: "PA", 2: "CT_META"}

    def __init__(self, config: Optional[HGAConfig] = None, bytecode: Optional[bytearray] = None):
        self.cfg = config or HGAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.latent_dim > self.cfg.psi_dim:
            raise ValueError("latent_dim must not exceed psi_dim")

        w_enc = np.zeros((self.cfg.latent_dim, self.cfg.psi_dim), dtype=float)
        w_enc[:, : self.cfg.latent_dim] = np.eye(self.cfg.latent_dim)
        w_dec = w_enc.T.copy()
        x = np.linspace(0.0, 2.0 * np.pi, self.cfg.psi_dim, endpoint=False)
        psi = np.sin(x) * np.exp(-np.arange(self.cfg.psi_dim) / 9.0)
        theta = Theta(w_enc=w_enc, w_dec=w_dec)
        z = w_enc @ psi

        self.state = HGAState(
            pc=0,
            bytecode=bytearray(bytecode or self.default_program()),
            gamma=set(),
            clifford_metric=np.diag([1.0, -1.0, -1.0, -1.0]),
            psi=psi,
            z=z,
            v=np.zeros(self.cfg.latent_dim, dtype=float),
            omega=np.zeros(self.cfg.psi_dim, dtype=float),
            theta=theta,
            proof=ProofBundle(),
            memory=MachineMemory(),
        )
        self.clifford = Clifford4((1, -1, -1, -1))

    @staticmethod
    def default_program() -> bytearray:
        return BytecodeCodec.encode([
            Instruction(Opcode.AXIOM_LOAD, 0),
            Instruction(Opcode.AXIOM_LOAD, 1),
            Instruction(Opcode.AXIOM_LOAD, 2),
            Instruction(Opcode.GEOM_INIT),
            Instruction(Opcode.PERMEATE, 1),
            Instruction(Opcode.FOLD_INWARD),
            Instruction(Opcode.VERIFY),
            Instruction(Opcode.SELF_EXEC),
            Instruction(Opcode.MDL_SCORE),
            Instruction(Opcode.TIME_BLOCK, 4, 3),
            Instruction(Opcode.OPTIMIZE),
            Instruction(Opcode.HALT),
        ])

    def encode(self, psi: np.ndarray, theta: Optional[Theta] = None) -> np.ndarray:
        t = theta or self.state.theta
        return np.tanh(t.w_enc @ psi)

    def decode(self, z: np.ndarray, theta: Optional[Theta] = None) -> np.ndarray:
        t = theta or self.state.theta
        return t.w_dec @ z

    def cycle_loss(self, state: Optional[HGAState] = None) -> float:
        s = state or self.state
        z = self.encode(s.psi, s.theta)
        psi_hat = self.decode(z, s.theta)
        z_cycle = self.encode(psi_hat, s.theta)
        rec = np.mean((s.psi - psi_hat) ** 2)
        cyc = np.mean((z - z_cycle) ** 2)
        geo = np.mean(z**2)
        mdl = s.theta.parameter_count()
        return float(
            self.cfg.lambda_rec * rec
            + self.cfg.lambda_cycle * cyc
            + self.cfg.lambda_geo * geo
            + self.cfg.lambda_mdl * mdl
        )

    def grad_potential_z(self, state: HGAState) -> np.ndarray:
        z = state.z
        psi_hat = self.decode(z, state.theta)
        rec_grad = (2.0 / self.cfg.psi_dim) * (state.theta.w_dec.T @ (psi_hat - state.psi))
        z_cycle = self.encode(psi_hat, state.theta)
        cycle_grad = (2.0 / self.cfg.latent_dim) * (z - z_cycle)
        geo_grad = (2.0 / self.cfg.latent_dim) * z
        return (
            self.cfg.lambda_rec * rec_grad
            + self.cfg.lambda_cycle * cycle_grad
            + self.cfg.lambda_geo * geo_grad
        )

    def permeate(self, state: HGAState, steps: int = 1) -> HGAState:
        out = state.clone()
        for _ in range(max(1, steps)):
            grad_u = self.grad_potential_z(out)
            denom = 1.0 + self.cfg.dt * self.cfg.damping / self.cfg.mass
            control = 0.05 * self.encode(out.omega, out.theta)
            v_next = (
                out.v
                - self.cfg.dt * grad_u / self.cfg.mass
                + self.cfg.dt * control / self.cfg.mass
            ) / denom
            z_next = out.z + self.cfg.dt * v_next
            out.v = v_next
            out.z = z_next
        return out

    def fold_inward(self, state: HGAState, train: bool = True) -> HGAState:
        out = state.clone()
        z = self.encode(out.psi, out.theta)
        psi_hat = self.decode(z, out.theta)
        error = out.psi - psi_hat
        omega_next = self.cfg.rho_memory * out.omega + (1.0 - self.cfg.rho_memory) * error
        corrected = psi_hat + self.cfg.error_gain * error + self.cfg.memory_gain * omega_next

        if train:
            n = self.cfg.psi_dim
            d_hat = (2.0 / n) * (psi_hat - out.psi)
            grad_w_dec = np.outer(d_hat, z)
            dz = out.theta.w_dec.T @ d_hat
            pre = out.theta.w_enc @ out.psi
            dz_pre = dz * (1.0 - np.tanh(pre) ** 2)
            grad_w_enc = np.outer(dz_pre, out.psi)
            for grad in (grad_w_dec, grad_w_enc):
                norm = np.linalg.norm(grad)
                if norm > 1.0:
                    grad /= norm
            out.theta.w_dec -= self.cfg.learning_rate * grad_w_dec
            out.theta.w_enc -= self.cfg.learning_rate * grad_w_enc
            out.theta.epoch += 1

        out.z = self.encode(corrected, out.theta)
        out.psi = corrected
        out.omega = omega_next
        out.last_loss = self.cycle_loss(out)
        return out

    def state_energy(self, state: HGAState) -> float:
        kinetic = 0.5 * self.cfg.mass * float(np.dot(state.v, state.v))
        return kinetic + self.cycle_loss(state)

    def contraction_proxy(self, state: HGAState) -> float:
        damping_factor = 1.0 / (1.0 + self.cfg.dt * self.cfg.damping / self.cfg.mass)
        dec_norm = float(np.linalg.norm(state.theta.w_dec, ord=2))
        return min(2.0, damping_factor * (0.5 + 0.5 * min(dec_norm, 1.5)))

    def verify(self, state: Optional[HGAState] = None) -> ProofBundle:
        s = state or self.state
        checks: Dict[str, Optional[bool]] = {}
        try:
            decoded = BytecodeCodec.decode_all(s.bytecode)
            checks["bytecode_well_formed"] = bool(decoded and decoded[-1].opcode == Opcode.HALT)
        except Exception:
            checks["bytecode_well_formed"] = False

        arrays = [s.psi, s.z, s.v, s.omega, s.theta.w_enc, s.theta.w_dec]
        checks["finite_state"] = all(np.isfinite(a).all() for a in arrays)
        rec = float(np.mean((s.psi - self.decode(self.encode(s.psi, s.theta), s.theta)) ** 2))
        energy = self.state_energy(s) if checks["finite_state"] else math.inf
        q = self.contraction_proxy(s) if checks["finite_state"] else math.inf
        checks["reconstruction"] = True if rec <= self.cfg.rec_epsilon else False if rec > self.cfg.unknown_rec_limit else None
        checks["energy"] = True if energy <= self.cfg.energy_max else False if energy > self.cfg.hard_energy_max else None
        checks["local_contraction"] = True if q < 1.0 else None
        checks["logic_loaded"] = {"ZFC", "PA", "CT_META"}.issubset(s.gamma)

        if any(v is False for v in checks.values()):
            status = ProofStatus.DISPROVED
        elif all(v is True for v in checks.values()):
            status = ProofStatus.PROVED
        else:
            status = ProofStatus.UNKNOWN

        payload = {
            "cycle": s.cycle,
            "checks": checks,
            "rec": round(rec, 12),
            "energy": round(energy, 12) if math.isfinite(energy) else "inf",
            "q": round(q, 12) if math.isfinite(q) else "inf",
            "bytecode_sha256": hashlib.sha256(bytes(s.bytecode)).hexdigest(),
        }
        cert = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return ProofBundle(status=status, checks=checks, certificates=[cert])

    def mdl_score(self, state: Optional[HGAState] = None) -> float:
        s = state or self.state
        rec = float(np.mean((s.psi - self.decode(self.encode(s.psi, s.theta), s.theta)) ** 2))
        estimated_ops = len(s.bytecode) / 4 + s.theta.parameter_count()
        energy = self.state_energy(s)
        return float(
            len(s.bytecode)
            + self.cfg.mdl_beta * rec
            + self.cfg.mdl_gamma * estimated_ops
            + self.cfg.mdl_mu * energy
        )

    def propose_optimized_bytecode(self, program: Optional[bytearray] = None) -> bytearray:
        source = program or self.state.bytecode
        instructions = BytecodeCodec.decode_all(source)
        return BytecodeCodec.encode([ins for ins in instructions if ins.opcode != Opcode.NOP])

    def verify_refinement(self, old: Sequence[int], new: Sequence[int]) -> ProofBundle:
        checks: Dict[str, Optional[bool]] = {}
        try:
            checks["semantic_signature"] = BytecodeCodec.semantic_signature(old) == BytecodeCodec.semantic_signature(new)
            checks["nonexpanding"] = len(new) <= len(old)
            checks["well_formed"] = BytecodeCodec.decode_all(new)[-1].opcode == Opcode.HALT
        except Exception:
            checks["semantic_signature"] = False
            checks["nonexpanding"] = False
            checks["well_formed"] = False
        status = ProofStatus.PROVED if all(v is True for v in checks.values()) else ProofStatus.DISPROVED
        cert = hashlib.sha256((bytes(old).hex() + "->" + bytes(new).hex() + json.dumps(checks, sort_keys=True)).encode()).hexdigest()
        return ProofBundle(status, checks, [cert])

    def transactional_self_optimize(self, state: HGAState) -> Tuple[HGAState, ProofBundle]:
        out = state.clone()
        candidate = self.propose_optimized_bytecode(out.bytecode)
        proof = self.verify_refinement(out.bytecode, candidate)
        if proof.status == ProofStatus.PROVED:
            out.bytecode = candidate
        else:
            out.memory.rollback_count += 1
        return out, proof

    @staticmethod
    def scheduler_levels() -> List[List[str]]:
        deps = {
            "decode": set(),
            "logic": {"decode"},
            "geometry": {"decode"},
            "permeate": {"geometry"},
            "fold": {"permeate"},
            "verify_pre": {"fold", "logic"},
            "self_exec": {"verify_pre"},
            "mdl": {"self_exec"},
            "time_block": {"self_exec"},
            "verify_post": {"mdl", "time_block"},
            "commit": {"verify_post"},
        }
        completed: set[str] = set()
        levels: List[List[str]] = []
        while len(completed) < len(deps):
            ready = sorted(name for name, pred in deps.items() if name not in completed and pred.issubset(completed))
            if not ready:
                raise RuntimeError("dependency graph contains a cycle")
            levels.append(ready)
            completed.update(ready)
        return levels

    def _latent_map(self, z: np.ndarray, state: HGAState) -> np.ndarray:
        tmp = state.clone()
        tmp.z = z.copy()
        tmp.v = np.zeros_like(z)
        return self.permeate(tmp, steps=1).z

    def time_block_solve(self, state: HGAState, horizon: int = 4, iterations: int = 4) -> Tuple[List[np.ndarray], List[float]]:
        horizon = max(2, int(horizon))
        iterations = max(1, int(iterations))
        trajectory = [state.z.copy() for _ in range(horizon + 1)]
        residuals: List[float] = []
        for _ in range(iterations):
            targets = [trajectory[0]]
            for t in range(horizon):
                targets.append(self._latent_map(trajectory[t], state))
            new_traj = [trajectory[0]]
            for t in range(1, horizon + 1):
                new_traj.append(0.25 * trajectory[t] + 0.75 * targets[t])
            trajectory = new_traj
            residual = 0.0
            for t in range(horizon):
                predicted = self._latent_map(trajectory[t], state)
                residual += float(np.mean((trajectory[t + 1] - predicted) ** 2))
            residuals.append(residual)
        return trajectory, residuals

    def execute_bytecode(self, state: Optional[HGAState] = None) -> HGAState:
        out = (state or self.state).clone()
        out.pc = 0
        out.halted = False
        while not out.halted:
            ins = BytecodeCodec.decode(out.bytecode, out.pc)
            next_pc = out.pc + ins.length
            if ins.opcode == Opcode.NOP:
                pass
            elif ins.opcode == Opcode.AXIOM_LOAD:
                if ins.a in self.AXIOMS:
                    out.gamma.add(self.AXIOMS[ins.a])
                else:
                    out.proof = ProofBundle(ProofStatus.DISPROVED, {"known_axiom": False}, [])
                    out.halted = True
            elif ins.opcode == Opcode.GEOM_INIT:
                out.clifford_metric = np.diag([1.0, -1.0, -1.0, -1.0])
            elif ins.opcode == Opcode.PERMEATE:
                out = self.permeate(out, steps=max(1, ins.a))
            elif ins.opcode == Opcode.FOLD_INWARD:
                out = self.fold_inward(out, train=True)
            elif ins.opcode == Opcode.VERIFY:
                out.proof = self.verify(out)
                if out.proof.status == ProofStatus.DISPROVED:
                    out.halted = True
            elif ins.opcode == Opcode.SELF_EXEC:
                out.memory.registers["self_program_bytes"] = float(len(out.bytecode))
                out.memory.registers["self_program_hash32"] = float(int(hashlib.sha256(bytes(out.bytecode)).hexdigest()[:8], 16))
            elif ins.opcode == Opcode.MDL_SCORE:
                out.last_mdl = self.mdl_score(out)
            elif ins.opcode == Opcode.TIME_BLOCK:
                _, residuals = self.time_block_solve(out, horizon=max(2, ins.a), iterations=max(1, ins.b))
                out.memory.registers["time_block_residual"] = residuals[-1]
            elif ins.opcode == Opcode.OPTIMIZE:
                out, refinement = self.transactional_self_optimize(out)
                out.memory.registers["self_opt_proved"] = 1.0 if refinement.status == ProofStatus.PROVED else 0.0
            elif ins.opcode == Opcode.HALT:
                out.halted = True
            out.pc = next_pc
            if out.pc > len(out.bytecode):
                out.halted = True
                out.proof = ProofBundle(ProofStatus.DISPROVED, {"pc_bounds": False}, [])
        return out

    def closed_cycle(self) -> HGAState:
        previous = self.state.clone()
        candidate = self.execute_bytecode(previous)
        candidate.cycle = previous.cycle + 1
        post = self.verify(candidate)
        candidate.proof = post
        if post.status == ProofStatus.PROVED:
            committed = candidate
        elif post.status == ProofStatus.UNKNOWN:
            committed = previous
            committed.cycle += 1
            committed.proof = post
            committed.memory.trace.append({"cycle": committed.cycle, "event": "preserve", "reason": "UNKNOWN"})
        else:
            committed = previous
            committed.cycle += 1
            committed.memory.rollback_count += 1
            committed.proof = post
            committed.memory.trace.append({"cycle": committed.cycle, "event": "rollback", "reason": "DISPROVED"})
        committed.last_loss = self.cycle_loss(committed)
        committed.last_mdl = self.mdl_score(committed)
        committed.memory.trace.append(self.telemetry(committed))
        self.state = committed
        return self.state

    def telemetry(self, state: Optional[HGAState] = None) -> Dict[str, object]:
        s = state or self.state
        rec = float(np.mean((s.psi - self.decode(self.encode(s.psi, s.theta), s.theta)) ** 2))
        return {
            "cycle": s.cycle,
            "proof": s.proof.status.value,
            "loss": float(self.cycle_loss(s)),
            "reconstruction_mse": rec,
            "energy": float(self.state_energy(s)),
            "contraction_proxy": float(self.contraction_proxy(s)),
            "mdl": float(self.mdl_score(s)),
            "bytecode_bytes": len(s.bytecode),
            "theta_epoch": s.theta.epoch,
            "rollback_count": s.memory.rollback_count,
            "gamma": sorted(s.gamma),
        }

    def run(self, cycles: int = 8) -> List[Dict[str, object]]:
        history = []
        for _ in range(max(1, cycles)):
            self.closed_cycle()
            history.append(self.telemetry())
        return history


def _print_human(history: List[Dict[str, object]], vm: HGAVM) -> None:
    print("HGA proof-gated geometric VM")
    print("=" * 78)
    for row in history:
        print(
            f"cycle={row['cycle']:03d} proof={row['proof']:10s} "
            f"loss={row['loss']:.8f} rec={row['reconstruction_mse']:.8f} "
            f"E={row['energy']:.8f} q={row['contraction_proxy']:.6f} "
            f"MDL={row['mdl']:.4f} B={row['bytecode_bytes']}B"
        )
    print("-" * 78)
    print("scheduler:", " -> ".join("+".join(level) for level in vm.scheduler_levels()))
    print("final certificate:", vm.state.proof.certificates[-1] if vm.state.proof.certificates else "none")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Operational HGA proof-gated geometric VM emulator")
    parser.add_argument("command", nargs="?", choices=["run", "inspect", "time-block"], default="run")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    vm = HGAVM(HGAConfig(seed=args.seed))

    if args.command == "inspect":
        data = {
            "bytecode_hex": bytes(vm.state.bytecode).hex(),
            "instructions": [{"opcode": ins.opcode.name, "a": ins.a, "b": ins.b, "c": ins.c} for ins in BytecodeCodec.decode_all(vm.state.bytecode)],
            "scheduler_levels": vm.scheduler_levels(),
            "initial": vm.telemetry(),
        }
        print(json.dumps(data, indent=2))
        return 0

    if args.command == "time-block":
        traj, residuals = vm.time_block_solve(vm.state, horizon=8, iterations=8)
        print(json.dumps({"horizon": len(traj) - 1, "residuals": residuals, "final_norm": float(np.linalg.norm(traj[-1]))}, indent=2))
        return 0

    history = vm.run(args.cycles)
    if args.as_json:
        print(json.dumps(history, indent=2))
    else:
        _print_human(history, vm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
