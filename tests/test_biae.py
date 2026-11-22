import hashlib
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from jarvisx import biae

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "research" / "3d_biae_omega"


def scalar_packet(side, field):
    """Single-voxel oracle using coordinates and modular array indexing."""
    count = side**3
    source = [(field[m // 8] >> (m % 8)) & 1 for m in range(count)]
    coordinates = [0] * count
    for x in range(side):
        for y in range(side):
            for z in range(side):
                morton = 0
                for bit in range(side.bit_length() - 1):
                    for axis, coord in enumerate((x, y, z)):
                        morton |= ((coord >> bit) & 1) << (3 * bit + axis)
                coordinates[morton] = (x ^ y ^ z) & 1
    latent = [b ^ source[(m - 7) % count] ^ source[(m + 13) % count] for m, b in enumerate(source)]
    drift = [b ^ coordinates[m] ^ source[(m - 3) % count] for m, b in enumerate(source)]
    warped = [z ^ d for z, d in zip(latent, drift)]
    residual = [
        b ^ warped[m] ^ warped[(m - 3) % count] ^ warped[(m + 11) % count]
        for m, b in enumerate(source)
    ]

    def pack(bits):
        return bytes(sum(bits[i + k] << k for k in range(8)) for i in range(0, count, 8))

    return biae.Packet(pack(latent), pack(residual), pack(drift))


class TestBIAE(unittest.TestCase):
    def test_scalar_oracle_and_source_free_decode(self):
        for side in (4, 8):
            size = side**3 // 8
            for field in (bytes(size), b"\xff" * size, biae.seeded_field(side, 11)):
                with self.subTest(side=side, first_byte=field[0]):
                    packet = biae.encode(side, field)
                    self.assertEqual(packet, scalar_packet(side, field))
                    self.assertEqual(biae.decode(side, packet), field)

    def test_chunk_seams_and_global_wrap(self):
        side = 128
        size = side**3 // 8
        count = size * 8
        seam = (1 << 16) * 8
        positions = (0, 3, 9, 63, 65, seam - 1, seam + 2, 3 * seam, count - 2)
        value = sum(1 << bit for bit in positions)
        field = value.to_bytes(size, "little")
        packet = biae.encode(side, field)
        mask = (1 << count) - 1

        def rot(value, distance):
            distance %= count
            return ((value << distance) | (value >> (count - distance))) & mask

        latent = value ^ rot(value, 7) ^ rot(value, -13)
        coords = int.from_bytes(b"\x96" * size, "little")
        drift = value ^ rot(value, 3) ^ coords
        warped = latent ^ drift
        residual = value ^ warped ^ rot(warped, 3) ^ rot(warped, -11)
        planes = (n.to_bytes(size, "little") for n in (latent, residual, drift))
        self.assertEqual(packet, biae.Packet(*planes))
        self.assertEqual(biae.decode(side, packet), field)

    def test_shear_counterexamples(self):
        one = (1).to_bytes(8, "little")
        zero = bytes(8)

        def shear(field):
            return biae.decode(4, biae.Packet(field, zero, zero))

        composed = shear(biae.encode(4, one).latent)
        self.assertEqual(int.from_bytes(composed, "little"), 0x1068010000000489)
        self.assertEqual(int.from_bytes(shear(shear(one)), "little"), 0x0000040000000041)

    def test_corruption_rolls_back_all_committed_fields(self):
        field = biae.seeded_field(8)
        good = biae.encode(8, field)
        state, _ = biae.commit(biae.State(8, field), good)
        saved = (state.tick, state.field, state.packet, state.omega)

        def flip(plane):
            return plane[:-1] + bytes([plane[-1] ^ 0x80])

        concealed = replace(good, latent=flip(good.latent), drift=flip(good.drift))
        self.assertEqual(biae.decode(8, concealed), field)
        for proposal, reason in (
            (replace(good, drift=flip(good.drift)), "drift"),
            (replace(good, residual=flip(good.residual)), "reconstruction"),
            (concealed, "latent"),
            (replace(good, residual=good.residual[:-1]), "packed bytes"),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                biae.commit(state, proposal)
            self.assertEqual((state.tick, state.field, state.packet, state.omega), saved)
        retry, _ = biae.commit(state, good)
        self.assertEqual(retry.tick, 2)
        self.assertNotEqual(state.omega, retry.omega)

    def test_geometry_shape_seed_and_counter_boundaries(self):
        self.assertEqual(biae.packed_size(1024), 134217728)
        for side in (0, 2, 6, 2048, True, 4.0):
            with self.assertRaises(ValueError):
                biae.packed_size(side)
        for bad_seed in (-1, 1 << 32, False):
            with self.assertRaises(ValueError):
                biae.seeded_field(4, bad_seed)
        self.assertNotEqual(biae.seeded_field(4, 0), biae.seeded_field(4, (1 << 32) - 1))
        for bad in (b"", bytes(9), bytearray(8)):
            with self.assertRaises(ValueError):
                biae.encode(4, bad)
        for tick in (-1, 1 << 64, True):
            with self.assertRaises(ValueError):
                biae.State(4, bytes(8), tick=tick)
        with self.assertRaises(ValueError):
            biae.State(4, bytes(8), omega=bytes(31))
        last = biae.State(4, bytes(8), tick=(1 << 64) - 1)
        with self.assertRaisesRegex(biae.GateRejected, "counter"):
            biae.commit(last, biae.encode(4, last.field))

    def test_seeded_replay_and_prior_ledger_binding(self):
        golden = [json.loads(row) for row in (RECORD / "replay.jsonl").read_text().splitlines()]
        ticks = [(r["side"], r["input_tick"]) for r in golden]
        self.assertEqual(ticks, [(4, 0), (4, 1), (1024, 0), (1024, 1)])
        field = biae.seeded_field(4)
        packet = biae.encode(4, field)
        for _ in range(2):
            state = biae.State(4, field)
            for expected in golden[:2]:
                state, receipt = biae.commit(state, packet)
                self.assertEqual({"seed": biae.SEED, **receipt}, expected)
                self.assertEqual(state.field, field)
        different, _ = biae.commit(biae.State(4, field, omega=b"\xff" * 32), packet)
        self.assertNotEqual(different.omega.hex(), golden[0]["omega_out"])
        digest = hashlib.sha3_256((RECORD / "DECLARATION.md").read_bytes()).hexdigest()
        self.assertEqual(digest, "67cc47ebc1f0e6265fa460ef37f89f4f0580e00befc19e41e921bd05a530f03f")

    def test_toroidal_diffusion_probe(self):
        impulse = (1).to_bytes(8, "little")
        neighbors = {(1, 0, 0), (3, 0, 0), (0, 1, 0), (0, 3, 0), (0, 0, 1), (0, 0, 3)}
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    actual = biae.diffusion_bit(4, impulse, x, y, z)
                    self.assertEqual(actual, int((x, y, z) in neighbors))
        with self.assertRaises(ValueError):
            biae.diffusion_bit(4, impulse, 4, 0, 0)

    def test_module_entry_point(self):
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        command = [sys.executable, "-m", "jarvisx.biae", "--side", "4"]
        result = subprocess.run(
            command + ["--ticks", "2"], env=env, capture_output=True, check=True
        )
        expected = (RECORD / "replay.jsonl").read_bytes().splitlines()[:2]
        self.assertEqual(result.stdout.splitlines(), expected)
        for option, value in (("--ticks", "0"), ("--seed", "-1"), ("--side", "3")):
            proc = subprocess.run(command + [option, value], env=env, capture_output=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn(b"error", proc.stderr)


if __name__ == "__main__":
    unittest.main()
