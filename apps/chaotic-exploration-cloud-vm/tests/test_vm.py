"""Executable conformance tests for the JARVIS X native Phi3D transition."""
import http.client
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import Handler, validate


def run(*args, voxels=None):
    return subprocess.run([str(ROOT / 'jarvisx_vm'), *args], input=voxels,
                          capture_output=True, text=True, timeout=10)


def state(*args, voxels=None):
    result = run(*args, voxels=voxels)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class NativeMathematicsTests(unittest.TestCase):
    def test_bounded_native_execution_and_bytecode(self):
        a = state('--side', '8', '--steps', '5', '--emit-field', '1')
        self.assertEqual(a['voxels'], 512)
        self.assertTrue(a['verified_finite'])
        self.assertTrue(a['spatial_invariants_verified'])
        self.assertFalse(a['asymptotic_convergence_proven'])
        self.assertEqual(len(a['trace']), a['iterations'])
        self.assertEqual(a['program_hex'], '10 20 30 40 50 60 ff')
        self.assertEqual(len(a['field_zyx']), 8**3)
        self.assertEqual(len(a['xy_slice']), 8**2)
        self.assertEqual(len(a['opcodes']), 7)
        self.assertTrue(0 <= a['path_id'] <= 3)
        self.assertTrue(a['max_abs'] <= 1.0 + 1e-10)

    def test_finite_states_and_fold_mass_invariance_across_scales(self):
        for n in (4, 8, 16, 32):
            with self.subTest(side=n):
                a = state('--side', str(n), '--steps', '4',
                          '--alpha', '0.125', '--beta', '0.25', '--epsilon', '0')
                self.assertEqual(a['voxels'], n**3)
                self.assertAlmostEqual(a['stability_margin'], 0.0, places=9)
                for record in a['trace']:
                    self.assertLessEqual(record['max_abs'], 1.0000000001)
                    self.assertLessEqual(record['fold_mean_error'], 1e-10)
                    self.assertTrue(math.isfinite(record['residual_rms']))

    def test_same_input_produces_same_state_and_trace(self):
        a = state('--seed', '19', '--steps', '11')
        b = state('--seed', '19', '--steps', '11')
        for item in ('fingerprint64', 'path_id', 'residual_rms', 'trace', 'xy_slice'):
            self.assertEqual(a[item], b[item])
        self.assertFalse(a['fingerprint_cryptographic'])

    def test_explicit_3d_zero_field_stationary(self):
        values = ' '.join(['0']*(4**3))
        a = state('--side', '4', '--steps', '10', '--chaos', '0',
                  '--input-stdin', '1', '--emit-field', '1', voxels=values)
        self.assertEqual(a['input_mode'], 'explicit_voxels')
        self.assertTrue(a['stationary_observed'])
        self.assertEqual(a['iterations'], 1)
        self.assertEqual(a['residual_rms'], 0.0)
        self.assertEqual(a['path_cost'], 0.0)
        self.assertTrue(all(v == 0 for v in a['field_zyx']))

    def test_explicit_nonuniform_field_roundtrip_path(self):
        vals = [(.5 if (x+y+z)%2 else -.5) for z in range(8)
                for y in range(8) for x in range(8)]
        a = state('--side', '8', '--steps', '2', '--chaos', '0',
                  '--input-stdin', '1', '--emit-field', '1',
                  voxels=' '.join(map(str, vals)))
        self.assertTrue(all(-1 <= v <= 1 for v in a['field_zyx']))
        self.assertEqual(a['iterations'], 2)
        self.assertNotEqual(a['field_zyx'], vals)
        mid = 8//2
        self.assertEqual(a['xy_slice'], a['field_zyx'][mid*64:(mid+1)*64])

    def test_chaos_is_spatial_not_uniform(self):
        zeros = ' '.join(['0']*(8**3))
        a = state('--side','8','--chaos','0.2','--epsilon','0',
                  '--steps','1','--input-stdin','1','--emit-field','1',voxels=zeros)
        self.assertNotEqual(a['forcing_amplitude'], 0)
        self.assertGreater(max(a['field_zyx'])-min(a['field_zyx']), 1e-6)

    def test_invalid_configuration_and_nonfinite_values_rejected(self):
        bad = [('--steps','-1'),('--side','200'),('--alpha','nan'),
               ('--beta','0.9'),('--alpha','0.16','--beta','0.5'),
               ('--decay','1.001'),('--emit-field','1','--side','32'),
               ('--no-such-op','1')]
        for args in bad:
            with self.subTest(args=args):
                self.assertNotEqual(run(*args).returncode,0)
        for v in ('nan','inf','1.00001','oops'):
            wrong = ' '.join([v]+['0']*63)
            self.assertNotEqual(run('--side','4','--input-stdin','1',voxels=wrong).returncode,0)
        self.assertNotEqual(run('--side','4','--input-stdin','1',voxels='0 1').returncode,0)
        self.assertNotEqual(run('--side','4','--input-stdin','1',voxels=' '.join(['0']*65)).returncode,0)

    def test_gateway_validation(self):
        for bad in ({'steps':-1},{'side':999},{'alpha':float('nan')},
                    {'seed':True},{'steps':'4'},{'command':'rm -rf /'},
                    {'alpha':.166,'beta':.4},{'values':[0]},
                    {'emit_field':True,'side':32},{'trace':1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(bad)
        args,values = validate({})
        self.assertEqual(len(args),22)
        self.assertIsNone(values)
        args,values = validate({'side':4,'values':[0.0]*64,'emit_field':True})
        self.assertIn('--input-stdin',args)
        self.assertEqual(len(values.split()),64)


class HttpIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.httpd.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def send(self, method,route,payload=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.httpd.server_port,timeout=10)
        body=json.dumps(payload) if payload is not None else None
        head={'Content-Type':'application/json'} if headers is None else headers
        try:
            conn.request(method,route,body,headers=head)
            res=conn.getresponse()
            status=res.status
            result=json.loads(res.read())
            return status,result
        finally:
            conn.close()

    def test_status_and_health(self):
        self.assertEqual(self.send('GET','/healthz')[0],200)
        status,response=self.send('GET','/api/status')
        self.assertEqual(status,200)
        self.assertFalse(response['trained_model'])
        self.assertIsNone(response['hardware_vms'])

    def test_real_3d_explicit_input_through_http(self):
        status,out=self.send('POST','/api/execute',{
            'side':4,'steps':4,'chaos':0,'values':[0.0]*64,
            'emit_field':True,'trace':True})
        self.assertEqual(status,200,out)
        self.assertEqual(len(out['field_zyx']),64)
        self.assertEqual(out['residual_rms'],0)
        self.assertTrue(out['spatial_invariants_verified'])
        self.assertEqual(len(out['trace']),1)
        self.assertGreaterEqual(out['gateway_ms'],0)

    def test_rejects_malformed_or_unsafe_request(self):
        self.assertEqual(self.send('POST','/api/execute',{'alpha':.16,'beta':.5})[0],400)
        self.assertEqual(self.send('POST','/api/execute',{'values':[2]*64,'side':4})[0],400)
        self.assertEqual(self.send('POST','/api/execute',{'epsilon':float('nan')})[0],400)
        self.assertEqual(self.send('POST','/api/nope',{})[0],404)
        self.assertEqual(self.send('POST','/api/execute',{},headers={'Content-Type':'text/plain'})[0],415)


if __name__ == '__main__':
    unittest.main()
