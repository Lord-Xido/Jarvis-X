"""Run one bounded native 3D computation in a Cloud Run Job or any container.

Request is JSON in JX_JOB_REQUEST; output is machine-readable JSON on stdout.
No implicit GCP calls, storage access, or unbounded agent automation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from server import validate

ROOT = Path(__file__).resolve().parent


def main():
    request = json.loads(os.environ.get('JX_JOB_REQUEST',
                                      '{"side":12,"steps":20,"trace":true}'))
    if not isinstance(request, dict):
        raise ValueError('JX_JOB_REQUEST must be a JSON object')
    task_index = int(os.environ.get('CLOUD_RUN_TASK_INDEX', '0'))
    if task_index < 0:
        raise ValueError('negative task index is invalid')
    if os.environ.get('JX_JOB_SHARDED', '0') == '1':
        if 'values' in request:
            raise ValueError('sharded jobs require seeded inputs')
        seed = request.get('seed', 42)
        if type(seed) is not int or seed < 0 or seed + task_index > 2147483647:
            raise ValueError('task seed would exceed native VM limits')
        request['seed'] = seed + task_index
    argv, voxels = validate(request)
    result = subprocess.run([str(ROOT / 'jarvisx_vm'), *argv], input=voxels,
                            text=True, capture_output=True, timeout=30,
                            cwd=ROOT, check=True)
    report = json.loads(result.stdout)
    if not report.get('spatial_invariants_verified'):
        raise RuntimeError('spatial invariant not verified')
    # Reproducible cryptographic digest of the *serialized report's result data*;
    # this is NOT a signature, not a proof of code provenance or a GPU benchmark.
    subset = {k: v for k, v in report.items() if k not in ('runtime_ms',)}
    canonical = json.dumps(subset, sort_keys=True, separators=(',', ':'), allow_nan=False)
    output = {'mode': 'bounded-native-batch', 'task_index': task_index,
              'result_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
              'result': report}
    print(json.dumps(output, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
