export interface AE3DConfig {
  grid?: number;
  dim?: number;
  beta?: number;
  recursionDepth?: number;
  foldGain?: number;
  learningRate?: number;
  seed?: number;
  maxFrame?: number;
  quantMode?: 'int8';
}

export interface AE3DMetrics {
  byteAccuracy: number;
  ber: number;
  bitErrors: number;
  mse: number;
  psnr: number;
  activeVoxels: number;
  latentDelta: number;
  stability: number;
  quantMSE: number;
  iterations: number;
  elapsedMs: number;
  bytesPerSecond: number;
}

export interface AE3DRunResult {
  input: Uint8Array;
  output: Uint8Array;
  metrics: AE3DMetrics;
  clock: number;
}

export declare class RecursiveByteANN3D {
  constructor(options?: AE3DConfig);
  grid: number;
  voxels: number;
  dim: number;
  beta: number;
  recursionDepth: number;
  foldGain: number;
  learningRate: number;
  clock: number;
  embedding: Float32Array;
  decoder: Float32Array;
  latent: Float32Array;
  dequant: Float32Array;
  qLatent: Int8Array;
  scales: Float32Array;
  activeMask: Uint8Array;
  voxelEnergy: Float32Array;
  run(input: string | Uint8Array | ArrayBuffer | number[], options?: {recursionDepth?: number}): AE3DRunResult;
  step(input: string | Uint8Array | ArrayBuffer | number[]): AE3DRunResult;
  train(input: string | Uint8Array | ArrayBuffer | number[], options?: {epochs?: number; learningRate?: number}): {epochs:number; loss:number; elapsedMs:number};
  encode(input: string | Uint8Array | ArrayBuffer | number[]): Float32Array;
  quantize(): number;
  decode(length?: number): Uint8Array;
  verify(): AE3DMetrics;
  setConfig(patch: Partial<AE3DConfig>): AE3DConfig;
  getConfig(): AE3DConfig;
  exportModel(): object;
  importModel(model: object): this;
  snapshot(): object;
  reset(seed?: number): this;
}
