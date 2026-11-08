import unittest, tempfile
from pathlib import Path
import numpy as np
from omega_ann_codec import (train_model,build_model,get_params,pack,unpack,parse_bitstream,run_trial,
                             MODEL_BYTES,HEADER)
from omega_stream import stream_encode,stream_verify_full,stream_decode_to_npy
from omega_controller import negotiate
from omega_codec_probe import generate_volume

class TestOmegaANN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params,cls.train=train_model(steps=220)
        cls.affine=generate_volume('affine',64,seed=11)
    def test_learned_and_held_out_affine(self):
        self.assertLess(self.train['loss_after'],self.train['loss_before']/100)
        self.assertTrue(np.max(np.abs(self.params['factors'] - get_params(build_model())['factors']))>1e-5)
        trial=run_trial(self.affine,self.params,'affine')
        self.assertTrue(trial['gates']['quality_accepted'])
        self.assertGreater(trial['quality']['min_q'],.98)
        self.assertEqual(trial['embedded_model_bytes'],MODEL_BYTES)
        self.assertEqual(trial['latent_payload_bytes'], 8*(64//16)**3)
    def test_model_included_and_standalone_decode(self):
        stream,_=pack(self.affine,self.params)
        dim,model,scales,codes=parse_bitstream(stream)
        self.assertEqual(tuple(dim),self.affine.shape)
        self.assertEqual(len(codes),64)
        self.assertEqual(unpack(stream).shape,self.affine.shape)
        self.assertTrue(np.isfinite(model['factors']).all())
    def test_tamper_truncation_bad_metadata(self):
        stream,_=pack(self.affine,self.params)
        for i in (HEADER.size+1,len(stream)-1):
            bad=bytearray(stream);bad[i] ^= 0x80
            with self.assertRaises(ValueError):unpack(bytes(bad))
        with self.assertRaises(ValueError):unpack(stream[:-1])
        with self.assertRaises(ValueError):unpack(stream+b'x')
        with self.assertRaises(ValueError):unpack(b'bad')
    def test_rare_noisy_tile_cannot_pass(self):
        mixed=generate_volume('mixed',64,seed=11)
        trial=run_trial(mixed,self.params,'mixed')
        self.assertGreater(trial['quality']['mean_q'],.98)
        self.assertLess(trial['quality']['min_q'],.98)
        self.assertFalse(trial['gates']['quality_accepted'])
    def test_random_noise_cannot_pass(self):
        noise=generate_volume('noise',32,seed=11)
        trial=run_trial(noise,self.params,'noise')
        self.assertFalse(trial['gates']['quality_accepted'])
    def test_file_streaming_full_coverage_and_file_decode(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t)
            original=t/'vol.npy'; encoded=t/'vol.onn3'; reconstructed=t/'reconstructed.npy'
            np.save(original,self.affine)
            mapped=np.load(original,mmap_mode='r',allow_pickle=False)
            r=stream_encode(mapped,self.params,encoded,batch_tiles=5)
            checks=stream_verify_full(mapped,encoded,batch_tiles=7)
            self.assertEqual(checks['coverage_fraction'],1.)
            self.assertEqual(checks['tiles_verified'],64)
            self.assertTrue(checks['mean_quality_pass'])
            self.assertTrue(checks['minimum_quality_pass'])
            self.assertEqual(r['model_bytes'],MODEL_BYTES)
            stream_decode_to_npy(encoded,reconstructed,batch_tiles=11)
            fromfile=np.load(reconstructed,mmap_mode='r',allow_pickle=False)
            self.assertEqual(fromfile.shape,self.affine.shape)
            self.assertLess(float(np.max(np.abs(fromfile-self.affine))),1e-3)
            damaged=bytearray(encoded.read_bytes());damaged[-1] ^= 0x80
            encoded.write_bytes(damaged)
            with self.assertRaises(ValueError):stream_verify_full(mapped,encoded)
    def test_controller_commits_only_verified_target(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            small=negotiate(generate_volume('affine',80,seed=11),self.params,root,'small')
            self.assertEqual(small['status'],'COMMITTED')
            self.assertEqual(small['codec'],'deterministic_polynomial_v1')
            self.assertTrue((root/'small.om3d').exists())
            self.assertFalse((root/'small.pending.onn3').exists())
            noise=generate_volume('noise',32,seed=11)
            failed=negotiate(noise,self.params,root,'noise')
            self.assertEqual(failed['status'],'REJECTED')
            self.assertFalse((root/'noise.onn3').exists())
            self.assertFalse((root/'noise.om3d').exists())

if __name__ == '__main__':unittest.main()
