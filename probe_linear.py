"""
Linear probing experiment for ImageNet shortcut model.

Method B: Train probe on ImageNet train split, evaluate on val split.
Extracts intermediate DiT features at each (t, dt_base) pair via pmap (multi-GPU).

Usage:
    python probe_linear.py \
        --checkpoint_dir "Shortcut Model Checkpoints/imagenet-shortcut2-b-fulldata800001" \
        --save_dir /tmp/probe_feats_imagenet \
        --output_dir /tmp/probe_results_imagenet \
        --model_size b \
        --num_train_samples 50000 \
        --num_val_samples 50000 \
        --batch_size 64 \
        --layers last_4 \
        --dt_base_max 5
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
import tqdm


# ─── Model Configs ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    'b':  {'hidden_size': 768,  'depth': 12, 'num_heads': 12, 'patch_size': 2, 'mlp_ratio': 4.0},
    'xl': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16, 'patch_size': 2, 'mlp_ratio': 4.0},
}


# ─── Multi-GPU helpers ────────────────────────────────────────────────────────
def _pad_to_n(arr, n):
    """Pad axis-0 to a multiple of n. Returns (padded_arr, original_B)."""
    B = arr.shape[0]
    pad = (-B) % n
    if pad:
        arr = np.concatenate([arr, arr[:pad]], axis=0)
    return arr, B

def _shard(arr, n_dev):
    """(B, ...) → (n_dev, B//n_dev, ...)."""
    return arr.reshape(n_dev, -1, *arr.shape[1:])

def _unshard(sharded, orig_B):
    """(n_dev, local_B, ...) → (orig_B, ...)."""
    return np.array(sharded).reshape(-1, *np.array(sharded).shape[2:])[:orig_B]


# ─── Argument Parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Linear probing on shortcut models (Method B)')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save extracted features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results (plots, CSV)')
    parser.add_argument('--model_size', type=str, choices=['b', 'xl'], default='b')
    parser.add_argument('--num_train_samples', type=int, default=50000,
                        help='# ImageNet train images to use for probe fitting')
    parser.add_argument('--num_val_samples', type=int, default=50000,
                        help='# ImageNet val images to use for probe eval')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dt_base_max', type=int, default=5,
                        help='Max dt_base (0~7). dt_base_max=5 → 63 pairs')
    parser.add_argument('--layers', type=str, default='last_4',
                        help='"all", "last", or "last_N" (e.g. "last_4")')
    parser.add_argument('--probe_c', type=float, default=0.316,
                        help='Regularization C for LogisticRegression')
    parser.add_argument('--skip_extraction', action='store_true')
    parser.add_argument('--skip_probe', action='store_true')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--imagenet_root', type=str,
                        default='/131_data/datasets/ImageNetLDM',
                        help='Root of ImageNetLDM folder '
                             '(expects ILSVRC2012_train/train/ and '
                             'ILSVRC2012_validation/validation/)')
    return parser.parse_args()


# ─── Pair Grid ────────────────────────────────────────────────────────────────
def get_all_pairs(dt_base_max=5):
    pairs = []
    for dt_base in range(dt_base_max + 1):
        for k in range(2 ** dt_base):
            pairs.append((dt_base, k / (2 ** dt_base)))
    return pairs


def feat_path(save_dir, dt_base, t, split):
    """split: 'train' or 'val'"""
    ts = f"{t:.6f}".replace('.', 'p')
    return os.path.join(save_dir, split, f"feat_dtbase{dt_base}_t{ts}.npy")


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_model(checkpoint_dir, model_size, num_classes=1000):
    from model import DiT
    from utils.train_state import TrainStateEma
    from utils.checkpoint import Checkpoint

    cfg = MODEL_CONFIGS[model_size]
    model_def = DiT(
        patch_size=cfg['patch_size'],
        hidden_size=cfg['hidden_size'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        out_channels=4,
        class_dropout_prob=0.0,
        num_classes=num_classes,
        dropout=0.0,
        ignore_dt=False,
    )
    rng = jax.random.PRNGKey(0)
    model_rngs = {'params': rng, 'label_dropout': rng, 'dropout': rng}
    params = model_def.init(
        model_rngs,
        jnp.zeros((1, 32, 32, 4)),
        jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32)
    )['params']
    train_state = TrainStateEma.create(model_def, params, rng=rng)

    cp = Checkpoint(checkpoint_dir)
    replace_dict = cp.load_as_dict()['train_state']
    del replace_dict['opt_state']
    train_state = train_state.replace(**replace_dict)
    print(f"Loaded checkpoint from {checkpoint_dir}")
    return train_state


# ─── Dataset & VAE Encoding ───────────────────────────────────────────────────
def _build_imagenet_tf_dataset(split_dir, filelist_path, num_samples, batch_size, is_train):
    """
    Build a tf.data pipeline over raw ImageNet folder structure.
    split_dir: e.g. /131_data/datasets/ImageNetLDM/ILSVRC2012_train/train
    filelist_path: e.g. .../ILSVRC2012_train/filelist.txt  (lines: synset/fname.JPEG)
    Returns (tf.data.Dataset yielding (images_bhwc float32 [-1,1], labels_int64)),
            synset_to_idx dict.
    """
    import tensorflow as tf

    # Build synset→label mapping (alphabetical = standard ImageNet order)
    with open(filelist_path) as f:
        rel_paths = [l.strip() for l in f if l.strip()]
    synsets_sorted = sorted(set(p.split('/')[0] for p in rel_paths))
    synset_to_idx = {s: i for i, s in enumerate(synsets_sorted)}

    # Subsample
    if num_samples < len(rel_paths):
        rng = np.random.RandomState(42)
        rel_paths = rng.choice(rel_paths, num_samples, replace=False).tolist()

    abs_paths = [os.path.join(split_dir, p) for p in rel_paths]
    labels    = [synset_to_idx[p.split('/')[0]] for p in rel_paths]

    path_ds  = tf.data.Dataset.from_tensor_slices(abs_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    def load_and_resize(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        # centre-crop to square then resize
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        side = tf.minimum(h, w)
        img = tf.image.resize_with_crop_or_pad(img, side, side)
        img = tf.image.resize(img, (256, 256), antialias=True)
        if is_train:
            img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float32) / 127.5 - 1.0  # → [-1, 1]
        return img

    img_ds = path_ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((img_ds, label_ds))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, synset_to_idx


def setup_vae_pmap():
    """Create VAE + pmap'd encode once. Returns (params_rep, encode_pmap_fn)."""
    from utils.stable_vae import StableVAE
    n_dev = jax.local_device_count()
    print(f"Setting up VAE on {n_dev} device(s)...")
    vae = StableVAE.create()
    scaling_factor = vae.module.config.scaling_factor

    @jax.pmap
    def _p_encode(params, key, images_bchw):
        latents = vae.module.apply(
            {'params': params}, images_bchw, method=vae.module.encode
        ).latent_dist.sample(key)
        return latents * scaling_factor

    params_rep = jax.device_put_replicated(vae.params, jax.local_devices())
    return params_rep, _p_encode


def encode_imagenet_split(is_train, num_samples, batch_size, imagenet_root,
                          params_rep, encode_pmap_fn):
    """Encode one ImageNet split to VAE latents. Returns (latents, labels)."""
    n_dev = jax.local_device_count()
    split_name = 'train' if is_train else 'val'

    if is_train:
        split_dir     = os.path.join(imagenet_root, 'ILSVRC2012_train', 'train')
        filelist_path = os.path.join(imagenet_root, 'ILSVRC2012_train', 'filelist.txt')
    else:
        split_dir     = os.path.join(imagenet_root, 'ILSVRC2012_validation', 'validation')
        filelist_path = os.path.join(imagenet_root, 'ILSVRC2012_validation', 'filelist.txt')

    vae_rng = jax.random.PRNGKey(42 + int(is_train))

    print(f"Loading ImageNet [{split_name}] from {split_dir} ...")
    # is_train=False for both splits: no random flip during encoding
    ds, _ = _build_imagenet_tf_dataset(split_dir, filelist_path,
                                        num_samples, batch_size, is_train=False)

    all_latents, all_labels = [], []
    for batch_idx, (imgs_np, lbls_np) in enumerate(
            tqdm.tqdm(ds.as_numpy_iterator(), desc=f'Encoding [{split_name}]')):
        B = imgs_np.shape[0]
        imgs_bchw = imgs_np.transpose(0, 3, 1, 2)
        imgs_padded, orig_B = _pad_to_n(imgs_bchw, n_dev)
        imgs_sharded = _shard(imgs_padded, n_dev)
        batch_rng = jax.random.fold_in(vae_rng, batch_idx)
        rngs = jnp.stack([jax.random.fold_in(batch_rng, d) for d in range(n_dev)])
        latents_sharded = encode_pmap_fn(params_rep, rngs, imgs_sharded)
        latents = _unshard(latents_sharded, orig_B)
        all_latents.append(latents)
        all_labels.append(lbls_np)

    latents = np.concatenate(all_latents, axis=0)
    labels  = np.concatenate(all_labels, axis=0)
    print(f"  [{split_name}] Encoded {len(latents)} images. Shape: {latents.shape}")
    return latents, labels


# ─── Feature Extraction ───────────────────────────────────────────────────────
def make_extract_fn(train_state):
    """Returns a pmap'd extract function."""
    n_dev = jax.local_device_count()
    params_ema_rep = jax.device_put_replicated(train_state.params_ema, jax.local_devices())

    @jax.pmap
    def _p_extract(params_ema, x_noisy, t_vec, dt_base_vec, label_vec):
        _, _, activations = train_state.model_def.apply(
            {'params': params_ema}, x_noisy, t_vec, dt_base_vec, label_vec,
            train=False, return_activations=True
        )
        return activations

    def extract(x_t_np, labels_np, t, dt_base, orig_B):
        imgs_padded, _ = _pad_to_n(x_t_np, n_dev)
        lbls_padded, _ = _pad_to_n(labels_np.astype(np.int32), n_dev)
        local_B = imgs_padded.shape[0] // n_dev
        x_sharded  = _shard(imgs_padded, n_dev)
        lbl_sharded = _shard(lbls_padded, n_dev)
        t_sharded  = np.full((n_dev, local_B), t, dtype=np.float32)
        dt_sharded = np.full((n_dev, local_B), float(dt_base), dtype=np.float32)
        acts = _p_extract(params_ema_rep, x_sharded, t_sharded, dt_sharded, lbl_sharded)
        return acts, orig_B

    return extract


def get_selected_layers(layers_arg, depth):
    if layers_arg == 'all':
        return [f'dit_block_{i}' for i in range(depth)]
    elif layers_arg == 'last':
        return [f'dit_block_{depth - 1}']
    elif layers_arg.startswith('last_'):
        n = int(layers_arg.split('_')[1])
        return [f'dit_block_{i}' for i in range(depth - n, depth)]
    raise ValueError(f"Unknown layers: {layers_arg}")


def extract_all_features(train_state, latents, labels, pairs, batch_size,
                         save_dir, selected_layers, split):
    """
    split: 'train' or 'val' — features saved to {save_dir}/{split}/.
    """
    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    noise_rng = jax.random.PRNGKey(42)
    fixed_noise = np.array(jax.random.normal(noise_rng, latents.shape))

    labels_path = os.path.join(split_dir, 'labels.npy')
    if not os.path.exists(labels_path):
        np.save(labels_path, labels)

    extract_fn = make_extract_fn(train_state)

    print(f"[{split}] Extracting: {len(pairs)} pairs × {len(selected_layers)} layers "
          f"× {len(latents)} images  on {jax.local_device_count()} GPU(s)")
    for dt_base, t in tqdm.tqdm(pairs, desc=f'Pairs [{split}]'):
        path = feat_path(save_dir, dt_base, t, split)
        if os.path.exists(path):
            continue

        all_feats = []
        for i in range(0, len(latents), batch_size):
            x_clean = latents[i:i + batch_size]
            x_noise = fixed_noise[i:i + batch_size]
            lbl_batch = labels[i:i + batch_size]
            B = x_clean.shape[0]

            x_t = (1 - (1 - 1e-5) * t) * x_noise + t * x_clean
            acts, orig_B = extract_fn(x_t, lbl_batch, t, dt_base, B)

            batch_feats = []
            for lk in selected_layers:
                feat = _unshard(jnp.mean(acts[lk], axis=2), orig_B)  # (B, hidden)
                batch_feats.append(feat.astype(np.float16))
            all_feats.append(np.stack(batch_feats, axis=1))  # (B, L, H)

        np.save(path, np.concatenate(all_feats, axis=0))  # (N, L, H)


# ─── Linear Probe ─────────────────────────────────────────────────────────────
def _run_pair_layer(dt_base, t, layer_idx, layer_name, save_dir, probe_c, n_classes):
    """Worker: fit probe on train features, score on val features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_feats  = np.load(feat_path(save_dir, dt_base, t, 'train'))   # (N_tr, L, H)
    val_feats    = np.load(feat_path(save_dir, dt_base, t, 'val'))     # (N_val, L, H)
    train_labels = np.load(os.path.join(save_dir, 'train', 'labels.npy'))
    val_labels   = np.load(os.path.join(save_dir, 'val',   'labels.npy'))

    X_tr  = train_feats[:, layer_idx, :].astype(np.float32)
    X_val = val_feats[:,   layer_idx, :].astype(np.float32)

    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)

    clf = LogisticRegression(C=probe_c, max_iter=200, solver='liblinear')
    clf.fit(X_tr, train_labels)
    top1 = clf.score(X_val, val_labels)

    if n_classes >= 5:
        proba = clf.predict_proba(X_val)
        top5_preds = np.argsort(proba, axis=1)[:, -5:]
        top5 = float(np.any(top5_preds == val_labels[:, None], axis=1).mean())
    else:
        top5 = top1

    return {(dt_base, t, layer_name): {'top1': top1, 'top5': top5}}


def run_all_probes(pairs, save_dir, selected_layers, probe_c=0.316, n_classes=1000):
    """Train on train-split features, eval on val-split. Parallelises jobs."""
    from joblib import Parallel, delayed

    jobs = [
        (dt_base, t, li, ln)
        for dt_base, t in pairs
        if (os.path.exists(feat_path(save_dir, dt_base, t, 'train')) and
            os.path.exists(feat_path(save_dir, dt_base, t, 'val')))
        for li, ln in enumerate(selected_layers)
    ]

    print(f"Running {len(jobs)} probe jobs in parallel (n_jobs=-1)...")
    results_list = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_run_pair_layer)(db, t, li, ln, save_dir, probe_c, n_classes)
        for db, t, li, ln in tqdm.tqdm(jobs, desc='Probing')
    )

    results = {}
    for d in results_list:
        results.update(d)
    return results


# ─── Save & Plot Results ──────────────────────────────────────────────────────
def save_results_csv(results, output_dir):
    import csv
    csv_path = os.path.join(output_dir, 'probe_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dt_base', 't', 'layer', 'top1', 'top5'])
        for (dt_base, t, layer), acc in results.items():
            writer.writerow([dt_base, f"{t:.6f}", layer,
                             f"{acc['top1']:.4f}", f"{acc['top5']:.4f}"])
    print(f"Results saved to {csv_path}")


def plot_results(results, pairs, selected_layers, dt_base_max, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    n_rows = dt_base_max + 1
    n_cols = 2 ** dt_base_max

    for layer_name in selected_layers:
        grid = np.full((n_rows, n_cols), np.nan)
        for dt_base, t in pairs:
            key = (dt_base, t, layer_name)
            if key not in results:
                continue
            k = round(t * (2 ** dt_base))
            col = k * (n_cols // (2 ** dt_base))
            grid[dt_base, col] = results[key]['top1']

        fig, ax = plt.subplots(figsize=(max(14, n_cols // 4), 5))
        vmax = grid[~np.isnan(grid)].max() if grid[~np.isnan(grid)].size > 0 else 1
        im = ax.imshow(grid, aspect='auto', origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Top-1 Accuracy')
        ax.set_xlabel('t  (0 = pure noise, 1 = clean)')
        ax.set_ylabel('dt_base  (step size = 1 / 2^dt_base)')
        ax.set_title(f'Linear Probe Accuracy — {layer_name}')
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'{b}  (d=1/{2**b})' for b in range(n_rows)])
        xtick_ts = [0.0, 0.25, 0.5, 0.75, 1.0]
        xtick_cols = [round(tv * (n_cols - 1)) for tv in xtick_ts]
        ax.set_xticks(xtick_cols)
        ax.set_xticklabels([f'{tv:.2f}' for tv in xtick_ts])
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'heatmap_{layer_name}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved heatmap: {out_path}")

    last_layer = selected_layers[-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    for dt_base in range(dt_base_max + 1):
        ts = sorted([t for (db, t) in pairs if db == dt_base])
        accs = [results.get((dt_base, t, last_layer), {}).get('top1', np.nan) for t in ts]
        ax.plot(ts, accs, marker='o', label=f'dt_base={dt_base} (d=1/{2**dt_base})')
    ax.set_xlabel('t (noise level)')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title(f'Linear Probe vs t — {last_layer}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'lineplot_{last_layer}.png'), dpi=150)
    plt.close()

    dt_bases = list(range(dt_base_max + 1))
    accs_t0 = [results.get((db, 0.0, last_layer), {}).get('top1', 0.0) for db in dt_bases]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f'dt={b}\n(d=1/{2**b})' for b in dt_bases], accs_t0)
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title(f'Linear Probe at t=0 (pure noise) vs step size — {last_layer}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'barplot_t0_{last_layer}.png'), dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = get_all_pairs(args.dt_base_max)
    print(f"Total (t, dt_base) pairs: {len(pairs)}  (dt_base_max={args.dt_base_max})")

    cfg = MODEL_CONFIGS[args.model_size]
    selected_layers = get_selected_layers(args.layers, cfg['depth'])
    print(f"Probing layers: {selected_layers}")

    # ── Feature extraction ────────────────────────────────────────────────
    if not args.skip_extraction:
        params_rep, encode_pmap_fn = setup_vae_pmap()

        train_latents, train_labels = encode_imagenet_split(
            is_train=True, num_samples=args.num_train_samples,
            batch_size=args.batch_size, imagenet_root=args.imagenet_root,
            params_rep=params_rep, encode_pmap_fn=encode_pmap_fn
        )
        val_latents, val_labels = encode_imagenet_split(
            is_train=False, num_samples=args.num_val_samples,
            batch_size=args.batch_size, imagenet_root=args.imagenet_root,
            params_rep=params_rep, encode_pmap_fn=encode_pmap_fn
        )

        print("Loading model...")
        train_state = load_model(args.checkpoint_dir, args.model_size, args.num_classes)

        extract_all_features(
            train_state, train_latents, train_labels, pairs,
            args.batch_size, args.save_dir, selected_layers, split='train'
        )
        extract_all_features(
            train_state, val_latents, val_labels, pairs,
            args.batch_size, args.save_dir, selected_layers, split='val'
        )
    else:
        print("Skipping feature extraction (--skip_extraction).")

    # ── Linear probing ────────────────────────────────────────────────────
    if not args.skip_probe:
        results = run_all_probes(pairs, args.save_dir, selected_layers,
                                 args.probe_c, args.num_classes)
        save_results_csv(results, args.output_dir)
    else:
        import csv
        results = {}
        csv_path = os.path.join(args.output_dir, 'probe_results.csv')
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                key = (int(row['dt_base']), float(row['t']), row['layer'])
                results[key] = {'top1': float(row['top1']), 'top5': float(row['top5'])}
        print(f"Loaded {len(results)} results from {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(results, pairs, selected_layers, args.dt_base_max, args.output_dir)
    print("All done.")


if __name__ == '__main__':
    main()
