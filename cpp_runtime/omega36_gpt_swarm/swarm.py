"""Bounded Ω36 task-swarm adapter. No network calls by default."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import tempfile

L = 10**12
Coord = tuple[int, int, int]
OFFSETS = ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))
MAX_MESSAGE = 256
MAX_ACTIVE = 64

def address(p: Coord) -> int:
    if len(p) != 3 or any(type(x) is not int or not 0 <= x < L for x in p):
        raise ValueError("coordinate outside logical lattice")
    x,y,z = p
    return x + L*y + L*L*z

def coordinate(a: int) -> Coord:
    if type(a) is not int or not 0 <= a < L**3:
        raise ValueError("invalid address")
    return a % L, (a // L) % L, a // (L*L)

def neighbors(p: Coord) -> tuple[Coord, ...]:
    address(p)
    return tuple(tuple((x+d)%L for x,d in zip(p,off)) for off in OFFSETS)

def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")

@dataclass(frozen=True)
class Node:
    value: str
    memory: tuple[str, ...] = ()

@dataclass(frozen=True)
class Request:
    position: Coord
    tick: int
    value: str
    memory: tuple[str, ...]
    adjacent: tuple[tuple[Coord, str], ...]

Backend = Callable[[Request], str]
Score = Callable[[Coord, str], int]

def fixture_backend(request: Request) -> str:
    """Deterministic arithmetic fixture, NOT GPT inference."""
    n = int(request.value)
    return str(max(0,n-1))

def fixture_score(position: Coord, value: str) -> int:
    n = int(value)
    if str(n) != value or not 0 <= n <= 255:
        raise ValueError("invalid fixture output")
    return n*n

def bounded_message(s: str) -> bool:
    return isinstance(s,str) and 0 < len(s.encode("utf-8")) <= MAX_MESSAGE

def verify_archive(svm: Path, payload: bytes) -> dict:
    """Use actual C++ spatial bytecode programs for the round trip."""
    side = 1
    while side**3 < len(payload)+4:
        side *= 2
    if side > 128:
        raise ValueError("active tile budget exceeded")
    raw = len(payload).to_bytes(4,"little") + payload
    raw += bytes(side**3-len(raw))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p/"in").write_bytes(raw)
        def run(*args):
            subprocess.run([str(svm.resolve()), *map(str,args)], check=True,
                           capture_output=True, timeout=120)
        run("emit","encode",p/"encode.bc")
        run("emit","decode",p/"decode.bc")
        run("encode",side,p/"in",p/"archive",p/"encode.bc")
        run("decode",p/"archive",p/"out",p/"decode.bc")
        if (p/"out").read_bytes() != raw:
            raise RuntimeError("bytecode round-trip mismatch")
        return {"side":side,"raw_bytes":len(raw),"archive_bytes":(p/"archive").stat().st_size,
                "payload_sha256":hashlib.sha256(payload).hexdigest()}

class Swarm:
    def __init__(self, states: dict[Coord,str], backend: Backend,
                 score: Score, *, workers: int = 4, memory_limit: int = 4):
        if not 1 <= len(states) <= MAX_ACTIVE or not 1 <= workers <= 8 or not 1 <= memory_limit <= 8:
            raise ValueError("physical budget exceeded")
        for p,v in states.items():
            address(p)
            if not bounded_message(v):
                raise ValueError("bad input message")
            if type(score(p,v)) is not int or score(p,v)<0:
                raise ValueError("invalid initial score")
        self.nodes = {p:Node(v) for p,v in states.items()}
        self.backend,self.score = backend,score
        self.workers,self.memory_limit = workers,memory_limit
        self.tick = 0

    def snapshot(self) -> bytes:
        return canonical([[list(p),n.value,list(n.memory)]
                          for p,n in sorted(self.nodes.items())])

    def step(self, *, svm: Path | None = None) -> dict:
        """Synchronous snapshot; bounded concurrent calls; per-node CTR.
        Archive verification failure aborts the whole state transaction.
        Backend callbacks must enforce their own deadlines and external budgets.
        """
        old = dict(self.nodes)
        positions = sorted(old)
        req = [Request(p,self.tick,old[p].value,old[p].memory,
                       tuple((q,old[q].value) for q in neighbors(p) if q in old))
               for p in positions]
        def propose(r):
            try:
                return self.backend(r)
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            proposals = list(pool.map(propose, req))
        nxt = dict(old)
        commits = 0
        for p,v in zip(positions,proposals):
            try:
                if not bounded_message(v):
                    continue
                cost = self.score(p,v)
                prior = self.score(p,old[p].value)
                if type(cost) is not int or cost < 0 or cost >= prior:
                    continue
            except Exception:
                continue
            nxt[p] = Node(v,(old[p].memory+(old[p].value,))[-self.memory_limit:])
            commits += 1
        payload = canonical([[list(p),n.value,list(n.memory)]
                             for p,n in sorted(nxt.items())])
        receipt = verify_archive(svm,payload) if svm is not None else None
        self.nodes = nxt
        self.tick += 1
        return {"tick":self.tick,"backend_calls":len(req),"active_nodes":len(req),
                "neighbor_messages":sum(len(r.adjacent) for r in req),
                "commits":commits,"rollbacks":len(req)-commits,
                "score":sum(self.score(p,n.value) for p,n in nxt.items()),
                "state_sha256":hashlib.sha256(payload).hexdigest(),"archive":receipt}
