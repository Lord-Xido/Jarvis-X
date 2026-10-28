"""Sparse kinetic 3D ROM: reversible addressing and damped wave retrieval."""
import numpy as np

class Kinetic3DROM:
    def __init__(self, grid_size=64, alpha=0.09, damping=0.06, stiffness=0.10):
        if grid_size < 2 or not (0 < alpha <= 1/3) or not (0 <= damping < 1) or stiffness < 0:
            raise ValueError("Invalid grid or wave parameters")
        self.n = int(grid_size)
        self.alpha, self.damping, self.stiffness = alpha, damping, stiffness
        self.rom = set()

    def encode(self, packet):
        if not isinstance(packet, (int, np.integer)) or not 0 <= packet < self.n**3:
            raise ValueError("Packet outside virtual ROM address space")
        return tuple(map(int, np.unravel_index(int(packet), (self.n,)*3)))

    def decode(self, xyz):
        x, y, z = xyz
        if any(not 0 <= v < self.n for v in xyz):
            raise ValueError("Invalid coordinate")
        return int(x*self.n*self.n+y*self.n+z)

    def etch_rom(self, packets):
        self.rom.update(self.encode(p) for p in packets)

    def run(self, query, steps=4, amplitude=100., threshold=1.):
        if steps < 1:
            raise ValueError("steps must be positive")
        origin = self.encode(query)
        radius = steps + 2
        side = 2*radius+1
        current = np.zeros((side,)*3, dtype=np.float64)
        previous = np.zeros_like(current)
        current[(radius,)*3] = amplitude
        mask = np.zeros_like(current)
        candidates = {}
        for xyz in self.rom:
            delta = tuple(((v-o+self.n//2)%self.n)-self.n//2 for v,o in zip(xyz,origin))
            if all(abs(d) <= radius for d in delta):
                idx = tuple(radius+d for d in delta)
                mask[idx] = 1.
                candidates[idx] = xyz
        scores = {idx: 0. for idx in candidates}
        for _ in range(steps):
            for idx in candidates:
                scores[idx] += abs(current[idx])
            lap = sum(np.roll(current, shift, axis=axis)
                      for axis in range(3) for shift in (-1,1)) - 6*current
            nxt = ((2-self.damping)*current-(1-self.damping)*previous
                   +self.alpha*lap-self.stiffness*mask*current)
            previous, current = current, nxt
        for idx in candidates:
            scores[idx] += abs(current[idx])
        if not scores:
            return None
        idx = max(scores, key=scores.get)
        return {"packet": self.decode(candidates[idx]), "coordinate": candidates[idx],
                "score": float(scores[idx])} if scores[idx] > threshold else None

if __name__ == "__main__":
    e = Kinetic3DROM()
    e.etch_rom([12345,54321,99999])
    print(e.run(54320))
