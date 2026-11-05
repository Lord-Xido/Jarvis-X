"""Bounded correctness tests for the isolated DM3D v5 numerical reference."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from reportlab.pdfgen.canvas import Canvas
import dm3d_septillion_inward_v4 as v4
import dm3d_septillion_inward_v5 as v5

class DM3DV5Tests(unittest.TestCase):
    def test_reference_and_spatial_axis_operations(self):
        cube=np.random.default_rng(37).integers(0,256,(12,12,12),dtype=np.uint8)
        restored=v4.decode3d(v4.encode3d(cube,v4.BASELINE).raw)
        self.assertTrue(np.array_equal(restored,cube))
        for gene in [(0,0,0),(2,4,9),(7,0,11)]:
            decoded=v4.decode3d(v4.encode3d(cube,gene).raw)
            self.assertEqual(decoded.shape,cube.shape)
            self.assertEqual(decoded.dtype,np.uint8)

    def test_multiscale_contracts_under_fixed_target(self):
        r=np.random.default_rng(71)
        a=r.normal(100,20,(12,12,12)).astype(np.float32)
        b=r.normal(100,20,(12,12,12)).astype(np.float32)
        target=np.ones_like(a)*42
        fa,_=v5.multiscale_field(a,target,steps=6)
        fb,_=v5.multiscale_field(b,target,steps=6)
        self.assertLessEqual(np.max(np.abs(fa-fb)),(.7201**6)*np.max(np.abs(a-b))+.002)

    def test_archive_chain_and_rejection_of_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            seed=root/'seed.pdf'
            pdf=Canvas(str(seed))
            pdf.drawString(40,700,'DM3D v5 CI smoke seed')
            pdf.save()
            output=root/'out'
            metrics=v5.run_engine(seed,None,output,generations=2,candidates=5,
                                  trials=1,psnr_floor=10.0,n=12,seed=4)
            self.assertEqual(metrics['generations'],2)
            self.assertGreaterEqual(metrics['evaluations'],6)
            self.assertEqual(v5.verify(output,seed),2)
            self.assertEqual(metrics['cardinality'],'(10^24)^(10^24)')
            self.assertEqual(metrics['physical_genotype_count'],768)
            with (output/'generation_02.pdf').open('ab') as stream:
                stream.write(b'\nCORRUPTED')
            with self.assertRaises(AssertionError):
                v5.verify(output,seed)

if __name__=='__main__':
    unittest.main()
