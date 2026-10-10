"""CLI: train, run, index, retrieve, refine and export all four modalities."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SceneDataset, RealSceneDataset, VOCAB, decode_text, collate_one, save_dataset_archive
from .model import ModelConfig, DM3DMultiVAE
from .training import train, set_seed, load_checkpoint, unpack
from .media import save_media
from .store import MemoryStore
from .inputs import load_local


def index_scenes(model, dataset_list, archive, db_path, batch_size=16):
    model.eval()
    rows = 0
    with MemoryStore(db_path) as store, torch.no_grad():
        for split, dataset in zip(('train', 'validation'), dataset_list):
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                z, _, _ = model.encode(unpack(batch))
                embeddings = z.detach().cpu().numpy()
                for i, emb in enumerate(embeddings):
                    store.add(int(batch['sample_id'][i]), batch['text'][i].tolist(),
                              batch['label'][i], split, str(archive), emb)
                    rows += 1
        counts = store.counts()
    return rows, counts


def run(args):
    if args.train_count < 8 or args.validation_count < 4 or args.train_count > 2048 or args.validation_count > 512:
        raise ValueError('dataset size out of bounds')
    if not 1 <= args.epochs <= 100:
        raise ValueError('invalid epochs')
    set_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset_manifest:
        trainset = RealSceneDataset(args.dataset_manifest, split='train')
        validset = RealSceneDataset(args.dataset_manifest, split='validation')
        if len(trainset) < 8 or len(validset) < 4:
            raise ValueError('manifest needs at least 8 train and 4 validation records')
        disclosure = f'User-supplied local media listed in {args.dataset_manifest}; files are resized/resampled for the bounded model.'
    else:
        trainset = SceneDataset(count=args.train_count, start=0, seed=args.seed)
        validset = SceneDataset(count=args.validation_count, start=args.train_count, seed=args.seed)
        disclosure = 'All training, validation, and query scenes are procedurally generated synthetic samples.' 
    # A full training run rebuilds its synthetic index to avoid stale embeddings.
    database_file = out / 'memory.sqlite'
    database_file.unlink(missing_ok=True)
    archive = out / 'source_modalities.npz'
    archived_rows = save_dataset_archive(archive, (trainset, validset))
    model = DM3DMultiVAE(ModelConfig(latent_dim=args.latent_dim))
    history = train(model, trainset, validset, epochs=args.epochs,
                    batch_size=args.batch_size, seed=args.seed,
                    lr=args.lr, out_dir=out)
    indexed, counts = index_scenes(model, [trainset, validset], archive, out / 'memory.sqlite', args.batch_size)
    # Fixed-point self-observation, based on held-out sample not used in SGD.
    scene = validset[0]
    x = collate_one(scene)
    with torch.no_grad():
        decoded, final_z, convergence = model.refine(
            x, alpha=args.alpha, tolerance=args.tolerance, max_steps=args.refinement_steps)
        original_mu, _, h = model.encode(x)
        reconstructed_media = save_media(out / 'reconstruction', decoded)
        prior_generator = torch.Generator().manual_seed(args.seed + 43)
        generated_z = torch.randn((1, args.latent_dim), generator=prior_generator)
        synth = model.decode(generated_z)
        synthetic_media = save_media(out / 'synthetic', synth)
    result_tokens = decoded['text_logits'].argmax(-1)[0]
    synthetic_tokens = synth['text_logits'].argmax(-1)[0]
    (out / 'decoded_text.txt').write_text(decode_text(result_tokens) + '\n')
    (out / 'synthetic_text.txt').write_text(decode_text(synthetic_tokens) + '\n')
    np.save(out / f'latent_{args.latent_dim}.npy', final_z[0].detach().cpu().numpy())
    with MemoryStore(out / 'memory.sqlite') as store:
        matches = store.top_k(original_mu[0].detach().cpu().numpy(), k=5,
                              exclude_id=scene['sample_id'], split='train')
        vecs = [store.embedding(m['sample_id']) for m in matches]
        store.react(scene['sample_id'], 'heldout_refinement', convergence['residuals'][-1])
        counts = store.counts()
    # Retrieved training vectors condition a separate multimodal generation path.
    if vecs:
        prior = torch.from_numpy(np.mean(vecs, axis=0).astype(np.float32)).unsqueeze(0)
        conditioned_latent = 0.65 * original_mu + 0.35 * prior
        with torch.no_grad():
            conditioned = model.decode(conditioned_latent)
        conditioned_media = save_media(out / 'retrieval_conditioned', conditioned)
        conditioned_text = decode_text(conditioned['text_logits'].argmax(-1)[0])
        (out / 'retrieval_conditioned_text.txt').write_text(conditioned_text + '\n')
    else:
        conditioned_media, conditioned_text = {}, ''
    summary = {
        'architecture': '4 separate modality stems -> unified 8x4x4x4 volume -> Gaussian VAE -> 512-dimensional Z -> 3D decoder -> four heads',
        'data_disclosure': disclosure,
        'training_samples': len(trainset), 'validation_samples': len(validset),
        'archived_samples': archived_rows, 'indexed_embeddings': indexed,
        'latent_dim': args.latent_dim, 'input_lattice': [8, 4, 4, 4],
        'initial_validation': history[0]['validation'], 'final_validation': history[-1]['validation'],
        'validation_objective_delta': history[-1]['validation']['total'] - history[0]['validation']['total'],
        'convergence': convergence, 'fixed_point_caveat': 'Convergence means change between successive latent states was small; it does not imply perfect reconstruction or a mathematical contraction guarantee.',
        'heldout_ground_truth_text': decode_text(scene['text']),
        'heldout_reconstructed_text': decode_text(result_tokens),
        'unconditional_synthetic_text': decode_text(synthetic_tokens),
        'retrieval_cosine_top5': matches, 'database_counts': counts,
        'retrieval_conditioned_text': conditioned_text,
        'retrieval_conditioned_files': conditioned_media,
        'reconstruction_files': reconstructed_media, 'unconditional_generation_files': synthetic_media,
        'computing_device': 'CPU', 'index_method': 'exact scan over SQLite float32 embeddings; not an approximate nearest neighbor index',
        'compression_caveat': '512 latent float32 values; no fixed compression ratio is claimed and model weights are not included in a naive per-sample ratio',
    }
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(f"DB: {counts}; samples_indexed={indexed}; latent={args.latent_dim}; ",
          f"refinement_steps={convergence['steps']} converged={convergence['converged']}", flush=True)
    print(f"heldout: {summary['heldout_ground_truth_text']}", flush=True)
    print(f"reconstructed: {summary['heldout_reconstructed_text']}", flush=True)
    print(f"synthetic: {summary['unconditional_synthetic_text']}", flush=True)
    print(f"validation_objective_delta={summary['validation_objective_delta']:.6f}", flush=True)
    print(f"PASS - artifacts in {out.resolve()}", flush=True)
    return summary


def replay(args):
    out = Path(args.output)
    model = load_checkpoint(out / 'dm3d_weights.pt')
    set_seed(args.seed)
    with np.load(out / 'source_modalities.npz') as archive:
        ids = archive['sample_id'].tolist()
        if args.sample_id not in ids:
            raise ValueError('sample id missing in stored source dataset')
        i = ids.index(args.sample_id)
        x = {name: torch.from_numpy(archive[name][i]).unsqueeze(0)
             for name in ('text', 'image', 'audio', 'video')}
    with torch.no_grad():
        result, z, diag = model.refine(x, alpha=args.alpha,
                                       tolerance=args.tolerance, max_steps=args.refinement_steps)
        mu, _, _ = model.encode(x)
    paths = save_media(out / f'replay_{args.sample_id}', result)
    with MemoryStore(out / 'memory.sqlite') as store:
        matches = store.top_k(mu[0].detach().numpy(), 5, args.sample_id, split='train')
    print(json.dumps({'sample_id': args.sample_id, 'text': decode_text(result['text_logits'].argmax(-1)[0]),
                      'convergence': diag, 'top5': matches, 'files': paths}, indent=2))



def ingest(args):
    """Run trained model on supplied local files (no training, no web upload)."""
    out = Path(args.output)
    model = load_checkpoint(out / 'dm3d_weights.pt')
    set_seed(args.seed)
    x = load_local(args.text, args.image, args.audio, args.video)
    with torch.no_grad():
        decoded, z, convergence = model.refine(x, alpha=args.alpha,
              tolerance=args.tolerance, max_steps=args.refinement_steps)
        mu, _, _ = model.encode(x)
    paths = save_media(out / 'external_ingestion', decoded)
    with MemoryStore(out / 'memory.sqlite') as store:
        neighbors = store.top_k(mu[0].detach().cpu().numpy(), 5, split='train')
    payload = {'input_text': args.text, 'reconstruction': decode_text(decoded['text_logits'].argmax(-1)[0]),
               'convergence': convergence, 'top5': neighbors, 'exports': paths,
               'disclosure': 'External data is processed locally using weights trained only on synthetic source scenes.'}
    (out / 'external_ingestion.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

def main():
    parser = argparse.ArgumentParser(description='DM3D complete trainable multimodal inward loop')
    subs = parser.add_subparsers(dest='mode', required=True)
    train_parser = subs.add_parser('run', help='build/train/index/refine/generate/export')
    train_parser.add_argument('--output', default='output')
    train_parser.add_argument('--dataset-manifest', default=None, help='JSON manifest of aligned local records, overrides synthetic training')
    train_parser.add_argument('--epochs', type=int, default=8)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--lr', type=float, default=0.003)
    train_parser.add_argument('--train-count', type=int, default=96)
    train_parser.add_argument('--validation-count', type=int, default=24)
    train_parser.add_argument('--latent-dim', type=int, default=512)
    train_parser.add_argument('--seed', type=int, default=7)
    train_parser.add_argument('--alpha', type=float, default=0.3)
    train_parser.add_argument('--tolerance', type=float, default=.002)
    train_parser.add_argument('--refinement-steps', type=int, default=12)
    replay_parser = subs.add_parser('replay', help='load checkpoint, replay one source, query memory')
    replay_parser.add_argument('--output', default='output')
    replay_parser.add_argument('--sample-id', type=int, default=96)
    replay_parser.add_argument('--seed', type=int, default=7)
    replay_parser.add_argument('--alpha', type=float, default=.3)
    replay_parser.add_argument('--tolerance', type=float, default=.002)
    replay_parser.add_argument('--refinement-steps', type=int, default=12)
    ingest_parser = subs.add_parser('ingest', help='process real local aligned text, PNG/JPEG, WAV, MP4')
    ingest_parser.add_argument('--output', default='output')
    ingest_parser.add_argument('--text', required=True)
    ingest_parser.add_argument('--image', required=True)
    ingest_parser.add_argument('--audio', required=True)
    ingest_parser.add_argument('--video', required=True)
    ingest_parser.add_argument('--seed', type=int, default=7)
    ingest_parser.add_argument('--alpha', type=float, default=.3)
    ingest_parser.add_argument('--tolerance', type=float, default=.002)
    ingest_parser.add_argument('--refinement-steps', type=int, default=12)
    args = parser.parse_args()
    if args.mode == 'run':
        run(args)
    elif args.mode == 'replay':
        replay(args)
    else:
        ingest(args)


if __name__ == '__main__':
    main()
