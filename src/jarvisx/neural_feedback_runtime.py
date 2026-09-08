from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import socket
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RuntimeConfig:
    input_dim: int = 64
    latent_dim: int = 16
    learning_rate: float = 0.03
    batch_size: int = 16
    epochs_per_cycle: int = 8
    simulations_per_observation: int = 3
    replay_capacity: int = 512
    verification_threshold: float = 0.58
    promotion_margin: float = 1e-6
    validation_fraction: float = 0.25
    gradient_clip: float = 5.0
    max_feed_bytes: int = 1_000_000
    feed_timeout_seconds: float = 8.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.input_dim < 8:
            raise ValueError("input_dim must be >= 8")
        if not 1 <= self.latent_dim < self.input_dim:
            raise ValueError("latent_dim must satisfy 1 <= latent_dim < input_dim")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        if self.epochs_per_cycle < 1:
            raise ValueError("epochs_per_cycle must be >= 1")
        if self.simulations_per_observation < 0:
            raise ValueError("simulations_per_observation must be >= 0")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if not 0.0 <= self.verification_threshold <= 1.0:
            raise ValueError("verification_threshold must be in [0, 1]")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")


@dataclass(frozen=True)
class WebObservation:
    claim: str
    content: str
    source: str
    source_group: str
    authority: float = 0.5
    counterevidence: float = 0.0
    timestamp: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WebObservation":
        required = ("claim", "content", "source")
        missing = [name for name in required if not str(value.get(name, "")).strip()]
        if missing:
            raise ValueError(f"observation missing required fields: {', '.join(missing)}")
        authority = float(value.get("authority", 0.5))
        counterevidence = float(value.get("counterevidence", 0.0))
        if not 0.0 <= authority <= 1.0:
            raise ValueError("authority must be in [0, 1]")
        if not 0.0 <= counterevidence <= 1.0:
            raise ValueError("counterevidence must be in [0, 1]")
        source = str(value["source"]).strip()
        return cls(
            claim=str(value["claim"]).strip(),
            content=str(value["content"]).strip(),
            source=source,
            source_group=str(value.get("source_group") or source).strip(),
            authority=authority,
            counterevidence=counterevidence,
            timestamp=str(value["timestamp"]) if value.get("timestamp") is not None else None,
        )


@dataclass(frozen=True)
class EvidenceScore:
    claim: str
    score: float
    authority: float
    independence: float
    support: float
    simulation_consistency: float
    predictive_agreement: float
    counterevidence: float
    accepted: bool


@dataclass(frozen=True)
class CycleReport:
    cycle: int
    observations: int
    claims: int
    accepted_claims: int
    accepted_vectors: int
    replay_size: int
    mean_verification: float
    baseline_loss: float | None
    candidate_loss: float | None
    promoted: bool
    production_generation: int


class HashingFeatureEncoder:
    """Deterministic text-to-vector boundary for untrusted web observations."""

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def encode(self, observation: WebObservation) -> np.ndarray:
        vector = np.zeros(self.input_dim, dtype=np.float64)
        text = f"{observation.claim} {observation.content}".lower()
        tokens = [token for token in text.replace("/", " ").split() if token]
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.input_dim
            sign = 1.0 if digest[8] & 1 else -1.0
            magnitude = 1.0 + (digest[9] / 255.0)
            vector[index] += sign * magnitude
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector


class DenseAutoencoder:
    """Small NumPy autoencoder with explicit forward/backward passes."""

    def __init__(self, input_dim: int, latent_dim: int, rng: np.random.Generator) -> None:
        scale_e = math.sqrt(2.0 / (input_dim + latent_dim))
        scale_d = math.sqrt(2.0 / (latent_dim + input_dim))
        self.w_e = rng.normal(0.0, scale_e, size=(input_dim, latent_dim))
        self.b_e = np.zeros(latent_dim, dtype=np.float64)
        self.w_d = rng.normal(0.0, scale_d, size=(latent_dim, input_dim))
        self.b_d = np.zeros(input_dim, dtype=np.float64)

    def clone(self) -> "DenseAutoencoder":
        clone = object.__new__(DenseAutoencoder)
        clone.w_e = self.w_e.copy()
        clone.b_e = self.b_e.copy()
        clone.w_d = self.w_d.copy()
        clone.b_d = self.b_d.copy()
        return clone

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w_e + self.b_e)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.w_d + self.b_d

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

    def loss(self, x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        error = self.reconstruct(x) - x
        return float(np.mean(error * error))

    def train_batch(self, x: np.ndarray, learning_rate: float, gradient_clip: float) -> float:
        z = self.encode(x)
        y = self.decode(z)
        diff = y - x
        loss = float(np.mean(diff * diff))
        grad_y = (2.0 / diff.size) * diff
        grad_w_d = z.T @ grad_y
        grad_b_d = np.sum(grad_y, axis=0)
        grad_z = (grad_y @ self.w_d.T) * (1.0 - z * z)
        grad_w_e = x.T @ grad_z
        grad_b_e = np.sum(grad_z, axis=0)
        gradients = (grad_w_e, grad_b_e, grad_w_d, grad_b_d)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        scale = min(1.0, gradient_clip / max(norm, 1e-12))
        self.w_e -= learning_rate * scale * grad_w_e
        self.b_e -= learning_rate * scale * grad_b_e
        self.w_d -= learning_rate * scale * grad_w_d
        self.b_d -= learning_rate * scale * grad_b_d
        return loss


class Simulator:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def variants(self, x: np.ndarray, count: int) -> list[np.ndarray]:
        variants: list[np.ndarray] = []
        for index in range(count):
            dropout = min(0.45, 0.08 + 0.06 * index)
            mask = self.rng.random(x.shape) >= dropout
            noise = self.rng.normal(0.0, 0.015 + 0.01 * index, size=x.shape)
            variant = x * mask + noise
            norm = float(np.linalg.norm(variant))
            if norm > 1.0:
                variant = variant / norm
            variants.append(variant.astype(np.float64, copy=False))
        return variants

    def consistency(self, model: DenseAutoencoder, x: np.ndarray, count: int) -> float:
        if count <= 0:
            return 1.0
        variants = self.variants(x, count)
        losses = [float(np.mean((model.reconstruct(v[None, :])[0] - x) ** 2)) for v in variants]
        return 1.0 / (1.0 + 24.0 * float(np.mean(losses)))


class VerificationGate:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.clip((a @ b) / denom, -1.0, 1.0))

    def score_group(
        self,
        claim: str,
        observations: Sequence[WebObservation],
        vectors: Sequence[np.ndarray],
        model: DenseAutoencoder,
        simulator: Simulator,
        simulations_per_observation: int,
    ) -> EvidenceScore:
        authority = float(np.mean([item.authority for item in observations]))
        independence = min(1.0, len({item.source_group for item in observations}) / 3.0)
        if len(vectors) > 1:
            pairwise = [
                max(0.0, self._cosine(vectors[i], vectors[j]))
                for i in range(len(vectors))
                for j in range(i + 1, len(vectors))
            ]
            support = float(np.mean(pairwise)) if pairwise else 0.5
        else:
            support = 0.5
        simulation_consistency = float(
            np.mean(
                [simulator.consistency(model, vector, simulations_per_observation) for vector in vectors]
            )
        )
        predictive_loss = model.loss(np.stack(vectors))
        predictive_agreement = 1.0 / (1.0 + 20.0 * predictive_loss)
        counterevidence = float(np.mean([item.counterevidence for item in observations]))
        score = float(
            np.clip(
                0.32 * authority
                + 0.24 * independence
                + 0.16 * support
                + 0.16 * simulation_consistency
                + 0.12 * predictive_agreement
                - 0.35 * counterevidence,
                0.0,
                1.0,
            )
        )
        return EvidenceScore(
            claim=claim,
            score=score,
            authority=authority,
            independence=independence,
            support=support,
            simulation_consistency=simulation_consistency,
            predictive_agreement=predictive_agreement,
            counterevidence=counterevidence,
            accepted=score >= self.threshold,
        )


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._vectors: list[np.ndarray] = []

    def add_many(self, vectors: Iterable[np.ndarray]) -> None:
        for vector in vectors:
            self._vectors.append(np.asarray(vector, dtype=np.float64).copy())
        overflow = len(self._vectors) - self.capacity
        if overflow > 0:
            del self._vectors[:overflow]

    def matrix(self) -> np.ndarray:
        if not self._vectors:
            return np.empty((0, 0), dtype=np.float64)
        return np.stack(self._vectors)

    def __len__(self) -> int:
        return len(self._vectors)


class NeuralFeedbackRuntime:
    """Proof-gated autoencoding/decoding runtime with shadow-model promotion."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.features = HashingFeatureEncoder(config.input_dim)
        self.production = DenseAutoencoder(config.input_dim, config.latent_dim, self.rng)
        self.candidate = self.production.clone()
        self.simulator = Simulator(self.rng)
        self.gate = VerificationGate(config.verification_threshold)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.production_generation = 0
        self.cycle = 0

    def encode(self, observation: WebObservation) -> np.ndarray:
        return self.production.encode(self.features.encode(observation)[None, :])[0]

    def decode(self, latent: np.ndarray) -> np.ndarray:
        z = np.asarray(latent, dtype=np.float64)
        if z.shape != (self.config.latent_dim,):
            raise ValueError(f"latent shape must be ({self.config.latent_dim},)")
        return self.production.decode(z[None, :])[0]

    def process(self, observations: Sequence[WebObservation]) -> CycleReport:
        self.cycle += 1
        grouped: dict[str, list[WebObservation]] = {}
        for observation in observations:
            grouped.setdefault(observation.claim, []).append(observation)
        scores: list[EvidenceScore] = []
        accepted_vectors: list[np.ndarray] = []
        for claim, group in grouped.items():
            vectors = [self.features.encode(item) for item in group]
            score = self.gate.score_group(
                claim,
                group,
                vectors,
                self.production,
                self.simulator,
                self.config.simulations_per_observation,
            )
            scores.append(score)
            if score.accepted:
                for vector in vectors:
                    accepted_vectors.append(vector)
                    accepted_vectors.extend(
                        self.simulator.variants(vector, self.config.simulations_per_observation)
                    )
        self.replay.add_many(accepted_vectors)
        baseline_loss: float | None = None
        candidate_loss: float | None = None
        promoted = False
        if len(self.replay) >= self.config.batch_size:
            baseline_loss, candidate_loss, promoted = self._evolve()
        return CycleReport(
            cycle=self.cycle,
            observations=len(observations),
            claims=len(grouped),
            accepted_claims=sum(score.accepted for score in scores),
            accepted_vectors=len(accepted_vectors),
            replay_size=len(self.replay),
            mean_verification=float(np.mean([score.score for score in scores])) if scores else 0.0,
            baseline_loss=baseline_loss,
            candidate_loss=candidate_loss,
            promoted=promoted,
            production_generation=self.production_generation,
        )

    def _evolve(self) -> tuple[float, float, bool]:
        matrix = self.replay.matrix()
        matrix = matrix[self.rng.permutation(len(matrix))]
        validation_count = max(1, int(round(len(matrix) * self.config.validation_fraction)))
        validation = matrix[:validation_count]
        training = matrix[validation_count:]
        if len(training) < 2:
            training = matrix
        baseline_loss = self.production.loss(validation)
        best_model = self.production.clone()
        best_loss = baseline_loss
        for multiplier in (0.5, 1.0, 1.5):
            trial = self.production.clone()
            lr = self.config.learning_rate * multiplier
            for _ in range(self.config.epochs_per_cycle):
                order = self.rng.permutation(len(training))
                for start in range(0, len(training), self.config.batch_size):
                    batch = training[order[start : start + self.config.batch_size]]
                    if len(batch):
                        trial.train_batch(batch, lr, self.config.gradient_clip)
            trial_loss = trial.loss(validation)
            if trial_loss < best_loss:
                best_model = trial
                best_loss = trial_loss
        self.candidate = best_model.clone()
        promoted = best_loss + self.config.promotion_margin < baseline_loss
        if promoted:
            self.production = best_model
            self.production_generation += 1
        return baseline_loss, best_loss, promoted

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": asdict(self.config),
            "cycle": self.cycle,
            "production_generation": self.production_generation,
        }
        np.savez_compressed(
            target,
            w_e=self.production.w_e,
            b_e=self.production.b_e,
            w_d=self.production.w_d,
            b_d=self.production.b_d,
            metadata=np.array(json.dumps(metadata)),
        )
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "NeuralFeedbackRuntime":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            runtime = cls(RuntimeConfig(**metadata["config"]))
            runtime.production.w_e = data["w_e"].copy()
            runtime.production.b_e = data["b_e"].copy()
            runtime.production.w_d = data["w_d"].copy()
            runtime.production.b_d = data["b_d"].copy()
            runtime.candidate = runtime.production.clone()
            runtime.cycle = int(metadata.get("cycle", 0))
            runtime.production_generation = int(metadata.get("production_generation", 0))
            return runtime


class JsonWebFeed:
    """Fetch bounded JSON observations while blocking local/private network targets."""

    def __init__(self, max_bytes: int, timeout_seconds: float, allowed_domains: Sequence[str] = ()):
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.allowed_domains = tuple(domain.lower().strip(".") for domain in allowed_domains if domain)

    @staticmethod
    def _validate_public_url(url: str, allowed_domains: Sequence[str] = ()) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("feed URL must use http or https")
        if parsed.username or parsed.password:
            raise ValueError("feed URL credentials are not allowed")
        hostname = parsed.hostname.lower().strip(".")
        normalized = tuple(domain.lower().strip(".") for domain in allowed_domains if domain)
        if normalized and not any(hostname == domain or hostname.endswith("." + domain) for domain in normalized):
            raise ValueError(f"feed host {hostname!r} is outside the configured domain allowlist")
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, parsed.port or 443):
            if family not in (socket.AF_INET, socket.AF_INET6):
                continue
            address = ipaddress.ip_address(sockaddr[0])
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_multicast
                or address.is_reserved
                or address.is_unspecified
            ):
                raise ValueError("feed URL resolves to a non-public address")
        return url

    def fetch(self, url: str) -> list[WebObservation]:
        safe_url = self._validate_public_url(url, self.allowed_domains)
        request = urllib.request.Request(
            safe_url,
            headers={"Accept": "application/json", "User-Agent": "Jarvis-X-Neural-Feedback/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read(self.max_bytes + 1)
        if len(payload) > self.max_bytes:
            raise ValueError("feed exceeded configured maximum size")
        parsed = json.loads(payload.decode("utf-8"))
        if isinstance(parsed, Mapping):
            parsed = parsed.get("observations")
        if not isinstance(parsed, list):
            raise ValueError("feed must be a JSON list or {'observations': [...]} object")
        return [WebObservation.from_mapping(item) for item in parsed]


class SyntheticWebFeed:
    """Deterministic offline feed used for smoke tests and demonstrations."""

    def __init__(self, seed: int = 17) -> None:
        self.seed = seed

    def observations(self, cycle: int) -> list[WebObservation]:
        phase = (self.seed + cycle) % 11
        facts = [
            ("orbital-signal", f"signal phase {phase} follows bounded sinusoidal drift"),
            ("thermal-signal", f"thermal index {20 + phase} remains inside simulated tolerance"),
        ]
        sources = [
            ("lab-a.example", "lab-a", 0.92),
            ("lab-b.example", "lab-b", 0.88),
            ("archive.example", "archive", 0.82),
        ]
        observations: list[WebObservation] = []
        for claim, fact in facts:
            for source, group, authority in sources:
                observations.append(
                    WebObservation(
                        claim=claim,
                        content=f"{fact}; independent measurement cycle {cycle}",
                        source=f"https://{source}/record/{cycle}/{claim}",
                        source_group=group,
                        authority=authority,
                        counterevidence=0.02,
                    )
                )
        return observations
