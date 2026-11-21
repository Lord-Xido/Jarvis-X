"""Reproducible machine-readable local / CI audit of Phi3D correctness gates."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'jarvisx_vm'

def sample(args):
    p = subprocess.run([str(ENGINE), *args], text=True,capture_output=True, timeout=15)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout)

def main():
    cases = []
    for n in (4,8,16,32):
        args = ['--side',str(n),'--steps','8','--epsilon','0','--seed','101',
                '--alpha','0.1','--beta','0.2']
        a,b=sample(args),sample(args)
        assert a['fingerprint64']==b['fingerprint64']
        assert a['trace']==b['trace']
        assert all(s['max_abs']<=1.0000000001 and s['fold_mean_error']<1e-10
                   for s in a['trace'])
        assert a['stability_margin']>=0 and a['verified_finite']
        assert a['voxels']==n**3
        canonical=json.dumps({'trace':a['trace'],'slice':a['xy_slice'],
                              'fingerprint':a['fingerprint64']},sort_keys=True,
                             separators=(',',':'))
        cases.append({'side':n,'iterations':a['iterations'],
                      'residual_rms':a['residual_rms'],
                      'mean_error':a['fold_mean_error'],
                      'max_abs':a['max_abs'],
                      'fingerprint64':a['fingerprint64'],
                      'sha256_of_serialized_evidence':hashlib.sha256(canonical.encode()).hexdigest(),
                      'invariants':'PASS','seed_determinism':'PASS'})
    print(json.dumps({'engine':'jarvisx-phi3d-native-cpu-v2',
                      'host':platform.platform(), 'compiler_target':'C++17',
                      'hardware_gpu_used':False,'case_count':len(cases),
                      'cases':cases},indent=2))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'AUDIT FAILURE: {exc}',file=sys.stderr)
        raise
