import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from serve import Core, Handler

class NativeHttpIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core=Core(d=10,seed=20261104)
        with cls.core.lock:cls.core.auto=False
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.server.core=cls.core
        cls.server.verbose=False
        cls.base='http://127.0.0.1:'+str(cls.server.server_port)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=1)
        cls.core.close()

    def request(self,method,path,payload=None,binary=False):
        data=(payload if binary else json.dumps(payload or {}).encode()) if method=='POST' else None
        req=urllib.request.Request(self.base+path,data=data,method=method,
            headers={'Content-Type':'application/octet-stream' if binary else 'application/json'})
        with urllib.request.urlopen(req,timeout=5) as r:
            content=r.read()
            return content if path in ('/api/export','/api/ann/export','/') else json.loads(content)

    def test_01_health_and_ui(self):
        self.assertTrue(self.request('GET','/api/health')['ok'])
        self.assertIn(b'JARVIS-X',self.request('GET','/'))

    def test_02_native_step_and_verification(self):
        state=self.request('GET','/api/state')
        self.assertEqual(len(state['values']),1000)
        self.assertEqual(state['native_backend'],'C++20 / CPU (ctypes shared library)')
        start=state['ticks']
        self.request('POST','/api/step',{'steps':8})
        self.assertEqual(self.request('GET','/api/state')['ticks'],start+8)
        verified=self.request('POST','/api/verify')
        self.assertTrue(verified['verified'])

    def test_03_path_and_healing(self):
        self.request('POST','/api/reset',{'seed':20261104})
        route=self.request('POST','/api/route',{'from':0,'to':999})
        self.assertEqual(route['hops'],27)
        self.request('POST','/api/fault',{'index':0})
        route=self.request('POST','/api/route',{'from':0,'to':999})
        self.assertFalse(route['found'])
        self.request('POST','/api/step',{'steps':17})
        self.assertIn(0,self.request('GET','/api/state')['faults'])
        self.request('POST','/api/step',{'steps':1})
        self.assertNotIn(0,self.request('GET','/api/state')['faults'])
        self.assertTrue(self.request('POST','/api/route',{'from':0,'to':999})['found'])

    def test_04_binary_export_import_and_corruption(self):
        data=self.request('GET','/api/export')
        self.assertTrue(data.startswith(b'JX3D'))
        a=self.request('GET','/api/state')['values']
        self.request('POST','/api/step',{'steps':6})
        self.assertTrue(self.request('POST','/api/import',data,True)['imported'])
        b=self.request('GET','/api/state')['values']
        self.assertEqual(a,b)
        corrupt=bytearray(data)
        corrupt[8]^=1
        with self.assertRaises(urllib.error.HTTPError) as ex:
            self.request('POST','/api/import',bytes(corrupt),True)
        self.assertEqual(ex.exception.code,400)
        self.assertEqual(b,self.request('GET','/api/state')['values'])

    def test_06_neural_ann_encode_decode_and_bitplanes(self):
        self.request('POST','/api/reset',{'seed':20261104})
        bits=self.request('GET','/api/bitplanes')
        self.assertEqual(len(bits['packed_hex']),8)
        self.assertEqual(len(bits['packed_hex'][0]),250)
        initial=self.request('GET','/api/state')
        original=initial['values']
        self.assertEqual(len(initial['reconstructed']),1000)
        self.assertEqual(sum(bits['bits_set']),sum(int(x).bit_count() for x in original))
        train=self.request('POST','/api/ann/train',{'epochs':100,'learning_rate':1.0})
        self.assertEqual(train['trainable_parameters'],76)
        self.assertGreater(train['accepted_updates'],0)
        self.assertLess(train['mse_after'],train['mse_before'])
        encoded=self.request('GET','/api/ann/export')
        self.assertEqual(encoded[:4],b'JXA1')
        self.request('POST','/api/step',{'steps':12})
        imported=self.request('POST','/api/ann/import',encoded,True)
        self.assertTrue(imported['imported'])
        self.assertEqual(self.request('GET','/api/state')['values'],original)
        corrupt=bytearray(encoded);corrupt[-1]^=1
        with self.assertRaises(urllib.error.HTTPError) as ex:
            self.request('POST','/api/ann/import',bytes(corrupt),True)
        self.assertEqual(ex.exception.code,400)
        self.assertEqual(self.request('GET','/api/state')['values'],original)

    def test_05_utf8_codec(self):
        payload='Ξ⃝ Jarvis-X 3D 3D '*10
        r=self.request('POST','/api/codec',{'text':payload})
        self.assertEqual(r['decoded_text'],payload)
        self.assertTrue(r['verified'])

if __name__=='__main__':unittest.main()
