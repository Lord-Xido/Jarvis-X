"""Independent functional, boundary, tamper and policy tests for the 1 MB DM3D engine."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
import sys

SRC=Path(__file__).with_name('Dr_Moagi_1MB_3D_Engine.py')
spec=importlib.util.spec_from_file_location('dm3d_standalone',SRC)
dm=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=dm
spec.loader.exec_module(dm)

class OperationTests(unittest.TestCase):
    def test_all_four_rotations(self):
        randomizer=random.Random(23)
        raw=bytes(randomizer.randrange(256) for _ in range(dm.TILE_BYTES))
        for index in range(4):
            with self.subTest(rotation=index):
                rot=dm.cube_rotate(raw,index)
                self.assertEqual(dm.cube_rotate(rot,index,inverse=True),raw)
                self.assertEqual(sorted(rot),sorted(raw))

    def test_reversible_modes_and_weights(self):
        r=random.Random(40)
        blob=bytes(r.randrange(256) for _ in range(dm.TILE_BYTES))
        w=(128,130,87,-15)
        for rot in range(4):
            transformed=dm.cube_rotate(blob,rot)
            data=[transformed,dm.zlib.compress(transformed),
                  dm.zlib.compress(dm.xyz_delta_encode(transformed)),
                  dm.zlib.compress(dm.predict_encode(transformed,w))]
            for k,payload in enumerate(data):
                self.assertEqual(dm.decode_payload(4*rot+k,payload,w),blob)

    def test_diverse_input_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rng=random.Random(4)
            for size in (0,1,31,100,2048,32767,32768,32769,65536,65545):
                content=bytes(rng.randrange(256) for _ in range(size))
                src=root/'input.bin';arc=root/'file.dm3d';dst=root/'output.bin'
                src.write_bytes(content)
                dm.encode_file(src,arc)
                dm.decode_file(arc,dst)
                self.assertEqual(hashlib.sha256(dst.read_bytes()).digest(),hashlib.sha256(content).digest())
                self.assertEqual(dst.read_bytes(),content)

    def test_incompatible_model_state_does_not_break_old_packets(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            src=root/'input';arc=root/'arc';dst=root/'dst';state=root/'model.json'
            src.write_bytes(dm.synthetic_tile(3,7)+dm.synthetic_tile(4,7))
            dm.encode_file(src,arc,state_file=state)
            obj=json.loads(state.read_text())
            obj['weights']=[0,0,256,0]
            state.write_text(json.dumps(obj))
            dm.decode_file(arc,dst)
            self.assertEqual(dst.read_bytes(),src.read_bytes())

    def test_untrusted_state_rejects_resource_overload(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            src=root/'input';src.write_bytes(b'ok')
            state=root/'state';state.write_bytes(b'X'*4097)
            with self.assertRaises(ValueError):
                dm.encode_file(src,root/'archive',state_file=state)

    def test_tamper_payload_and_header_cannot_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            src=root/'input';src.write_bytes(dm.synthetic_tile(2,8))
            arc=root/'archive';dm.encode_file(src,arc)
            for name,data in [('body',bytearray(arc.read_bytes())),('header',bytearray(arc.read_bytes()))]:
                if name=='body': data[-10]^=3
                else: data[20]^=1
                broken=root/'broken';broken.write_bytes(data)
                out=root/'not_committed'
                with self.assertRaises(dm.InvalidArchive):
                    dm.decode_file(broken,out)
                self.assertFalse(out.exists())

    def test_zlib_bomb_rejected(self):
        oversized=dm.zlib.compress(b'\x00'*(dm.TILE_BYTES+1))
        with self.assertRaises(dm.InvalidArchive):
            dm.guarded_decompress(oversized)

    def test_arbitrary_precision_morton_near_edge(self):
        top=dm.VIRTUAL_AXIS_BITS-1
        point=(top,top-1000,top-9000)
        self.assertEqual(dm.unmorton3(dm.morton3(*point)),point)
        with self.assertRaises(ValueError): dm.morton3(top+1,0,0)

    def test_spatial_extent_overrun(self):
        with self.assertRaises(ValueError):
            dm.validate_spatial_extent(8,2,(dm.VIRTUAL_AXIS_BITS//dm.SIDE-1,0,0))

    def test_https_host_allowlist(self):
        for url in ('http://api.github.com/repos/x/y/events',
                    'https://localhost/repos/x/y/events',
                    'https://api.github.com.evil.invalid/repos/x/y/events',
                    'https://api.github.com/repos/x/y/contents'):
            with self.assertRaises(ValueError):
                dm.no_redirect_http(url,1024,0.1)

    def test_full_embedded_vector_catalog(self):
        # The bundled 1MB file is itself a deterministic reference vector corpus.
        report=dm.selftest(check_vectors=True)
        self.assertGreater(report['tests_passed'],10000)

if __name__=='__main__':
    unittest.main(verbosity=2)
