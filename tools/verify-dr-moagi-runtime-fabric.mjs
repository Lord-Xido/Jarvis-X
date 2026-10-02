import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const manifestPath = resolve(repoRoot, 'apps/dr-moagi-platform-java/runtime-fabric.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

const fail = (message) => {
  console.error(`runtime-fabric: ${message}`);
  process.exitCode = 1;
};

if (manifest.schema !== 'jarvisx.dr-moagi-runtime-fabric/v1') fail('unsupported schema');
if (!manifest.platform?.id || !Array.isArray(manifest.platform?.surfaces)) fail('invalid platform declaration');
if (!Array.isArray(manifest.runtimes) || manifest.runtimes.length === 0) fail('runtime list is empty');

const ids = new Set();
for (const runtime of manifest.runtimes ?? []) {
  if (!runtime.id || !/^[a-z0-9][a-z0-9-]*$/.test(runtime.id)) {
    fail(`invalid runtime id: ${String(runtime.id)}`);
    continue;
  }
  if (ids.has(runtime.id)) fail(`duplicate runtime id: ${runtime.id}`);
  ids.add(runtime.id);

  if (!runtime.path || runtime.path.startsWith('/') || runtime.path.includes('..')) {
    fail(`${runtime.id}: unsafe path`);
    continue;
  }
  const runtimePath = resolve(repoRoot, runtime.path);
  if (!runtimePath.startsWith(repoRoot + '/')) fail(`${runtime.id}: path escapes repository`);
  if (!existsSync(runtimePath) || !statSync(runtimePath).isDirectory()) fail(`${runtime.id}: runtime directory missing`);

  if (!Array.isArray(runtime.capabilities) || runtime.capabilities.length === 0) {
    fail(`${runtime.id}: capabilities missing`);
  } else if (new Set(runtime.capabilities).size !== runtime.capabilities.length) {
    fail(`${runtime.id}: duplicate capability`);
  }

  if (!runtime.transport?.kind) fail(`${runtime.id}: transport kind missing`);
  if (!runtime.authorityBoundary?.trim()) fail(`${runtime.id}: authority boundary missing`);

  for (const relative of runtime.requiredFiles ?? []) {
    if (!relative || relative.startsWith('/') || relative.includes('..')) {
      fail(`${runtime.id}: unsafe required file ${relative}`);
      continue;
    }
    if (!existsSync(join(runtimePath, relative))) fail(`${runtime.id}: required file missing: ${relative}`);
  }

  const env = runtime.transport?.endpointEnv;
  if (env && !/^MOAGI_[A-Z0-9_]+$/.test(env)) fail(`${runtime.id}: endpoint env must use MOAGI_* namespace`);
}

const surfaceSet = new Set(manifest.platform?.surfaces ?? []);
if (surfaceSet.size !== (manifest.platform?.surfaces ?? []).length) fail('duplicate platform surface');
if (manifest.policy?.hostExecutionAuthority !== 'none-from-fabric') fail('fabric must not grant implicit host execution');
if (manifest.policy?.provisionalIsAuthoritative !== false) fail('provisional state must not be authoritative');

const workerFabric = manifest.workerFabric;
if (!workerFabric || workerFabric.id !== 'dr-moagi-6400-cube') {
  fail('canonical 6400-cube worker fabric missing');
} else {
  const dims = workerFabric.logicalDimensions ?? [];
  const brickDims = workerFabric.brickDimensions ?? [];
  const exactDims = Array.isArray(dims) && dims.length === 3 && dims.every((v) => v === 6400);
  const exactBrickDims = Array.isArray(brickDims) && brickDims.length === 3 && brickDims.every((v) => v === 100);
  if (!exactDims) fail('worker fabric logical dimensions must be exactly 6400^3');
  if (workerFabric.logicalWorkers !== 262144000000) fail('worker fabric logical worker count mismatch');
  if (workerFabric.brickSide !== 64) fail('worker fabric brick side must be 64');
  if (!exactBrickDims) fail('worker fabric brick dimensions must be exactly 100^3');
  if (workerFabric.logicalBricks !== 1000000) fail('worker fabric logical brick count mismatch');
  if (workerFabric.workersPerBrick !== 262144) fail('worker fabric workers-per-brick mismatch');
  if (workerFabric.logicalWorkers !== workerFabric.logicalBricks * workerFabric.workersPerBrick) {
    fail('worker fabric partition arithmetic does not close');
  }
  if (workerFabric.executionModel !== 'sparse-virtualized') fail('worker fabric must declare sparse-virtualized execution');

  for (const key of ['implementation', 'specification']) {
    const relative = workerFabric[key];
    if (!relative || relative.startsWith('/') || relative.includes('..')) {
      fail(`worker fabric has unsafe ${key} path`);
      continue;
    }
    if (!existsSync(resolve(repoRoot, relative))) fail(`worker fabric ${key} path missing: ${relative}`);
  }
  if (!workerFabric.authorityBoundary?.trim()) fail('worker fabric authority boundary missing');
}

const annIde = (manifest.runtimes ?? []).find((runtime) => runtime.id === 'dr-moagi-ann-ide');
if (!annIde) {
  fail('dr-moagi-ann-ide runtime registration missing');
} else {
  for (const capability of ['6400-cube-worker-fabric', 'sparse-scheduling', 'telemetry']) {
    if (!annIde.capabilities?.includes(capability)) fail(`dr-moagi-ann-ide missing capability: ${capability}`);
  }
}

if (!process.exitCode) {
  console.log(`runtime-fabric: PASS (${ids.size} specialist runtimes, ${surfaceSet.size} platform surfaces)`);
}
