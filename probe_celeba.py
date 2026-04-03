"""
Linear probing experiment for CelebA shortcut model.

Uses CelebA's 40 binary face attributes (Smiling, Male, Young, etc.) as labels.
For each (t, dt_base) pair, extracts DiT intermediate features and trains
a binary logistic regression probe per attribute.

Method B: trains probe on CelebA train split (partition 0), evaluates on val split (partition 1).

Usage (inside Docker):
    python probe_celeba.py \
        --checkpoint_dir "/workspace/checkpoints/celeba-shortcut2-every4400001" \
        --save_dir /tmp/probe_feats_celeba \
        --output_dir /tmp/probe_results_celeba \
        --num_train_samples 50000 \
        --num_val_samples 19867 \
        --batch_size 32 \
        --layers last_4 \
        --dt_base_max 5
"""

import argparse
import math
import os
import numpy as np
import jax
import jax.numpy as jnp
import tqdm


# ─── Multi-GPU helpers ────────────────────────────────────────────────────────
def _pad_to_n(arr, n):
    """Pad axis-0 to a multiple of n. Returns (padded_arr, original_B)."""
    B = arr.shape[0]
    pad = (-B) % n
    if pad:
        arr = np.concatenate([arr, arr[:pad]], axis=0)
    return arr, B

def _shard(arr, n_dev):
    """(B, ...) → (n_dev, B//n_dev, ...)  — pure numpy reshape."""
    return arr.reshape(n_dev, -1, *arr.shape[1:])

def _unshard(sharded, orig_B):
    """(n_dev, local_B, ...) → (orig_B, ...)  — device→host + unpad."""
    return np.array(sharded).reshape(-1, *np.array(sharded).shape[2:])[:orig_B]

# CelebA 40 attributes (in TFDS order)
CELEBA_ATTRS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young',
]

# Subset of most visually salient / balanced attributes for quick reporting
KEY_ATTRS = ['Male', 'Smiling', 'Young', 'Wavy_Hair', 'Wearing_Hat',
             'Eyeglasses', 'Bald', 'Attractive', 'Heavy_Makeup', 'Blond_Hair']

MODEL_CONFIGS = {
    'b':  {'hidden_size': 768,  'depth': 12, 'num_heads': 12, 'patch_size': 2, 'mlp_ratio': 4.0},
    'xl': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16, 'patch_size': 2, 'mlp_ratio': 4.0},
}


# ─── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--model_size', type=str, choices=['b', 'xl'], default='b')
    p.add_argument('--num_train_samples', type=int, default=50000,
                   help='# training images for probe fitting (max 162770)')
    p.add_argument('--num_val_samples', type=int, default=19867,
                   help='# val images for probe eval (max 19867)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--dt_base_max', type=int, default=5)
    p.add_argument('--layers', type=str, default='last_4')
    p.add_argument('--probe_c', type=float, default=1.0)
    p.add_argument('--skip_extraction', action='store_true')
    p.add_argument('--skip_probe', action='store_true')
    p.add_argument('--tfds_data_dir', type=str, default=None)
    return p.parse_args()


# ─── Pairs ────────────────────────────────────────────────────────────────────
def get_all_pairs(dt_base_max):
    pairs = []
    for dt_base in range(dt_base_max + 1):
        for k in range(2 ** dt_base):
            pairs.append((dt_base, k / (2 ** dt_base)))
    return pairs

def feat_path(save_dir, dt_base, t, split):
    """split: 'train' or 'val'"""
    ts = f"{t:.6f}".replace('.', 'p')
    return os.path.join(save_dir, split, f"feat_dtbase{dt_base}_t{ts}.npy")


# ─── Dataset: CelebA via local files ─────────────────────────────────────────
CELEBA_ROOT = '/131_data/datasets/CelebA'

def load_celeba_attrs(attr_file=None):
    """
    Parse the official CelebA list_attr_celeba.txt.
    Returns (attr_names, attrs) where attrs is (N, 40) int32 {0, 1}.
    """
    if attr_file is None:
        attr_file = os.path.join(CELEBA_ROOT, 'list_attr_celeba.txt')
    with open(attr_file) as f:
        lines = f.read().splitlines()
    # line 0: count, line 1: attribute names, line 2+: per-image data
    attr_names = lines[1].split()
    rows = []
    for line in lines[2:]:
        parts = line.split()
        vals = [1 if int(v) == 1 else 0 for v in parts[1:]]
        rows.append(vals)
    return attr_names, np.array(rows, dtype=np.int32)


def make_celeba_dataset(val_filenames, val_attrs, batch_size,
                        celeba_root='/131_data/datasets/CelebA'):
    """
    tf.data pipeline over local CelebA files.
    Returns an iterator of (images_np, attrs_np) batches,
    where images_np is float32 (B, 256, 256, 3) in [-1, 1].
    """
    import tensorflow as tf

    img_dir = os.path.join(celeba_root, 'img_align_celeba')
    full_paths = [os.path.join(img_dir, fn) for fn in val_filenames]

    path_ds  = tf.data.Dataset.from_tensor_slices(full_paths)
    attrs_ds = tf.data.Dataset.from_tensor_slices(val_attrs)

    def load_and_resize(path):
        raw  = tf.io.read_file(path)
        img  = tf.image.decode_jpeg(raw, channels=3)
        img  = tf.image.resize(img, [256, 256], antialias=True)
        img  = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img

    img_ds = path_ds.map(load_and_resize,
                         num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((img_ds, attrs_ds))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def setup_vae_pmap(celeba_root=CELEBA_ROOT):
    """Create VAE + pmap'd encode. Returns (params_rep, encode_pmap_fn)."""
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


def encode_celeba_split(partition_id, num_samples, batch_size,
                        params_rep, encode_pmap_fn,
                        celeba_root=CELEBA_ROOT):
    """
    Encode one CelebA partition (0=train, 1=val, 2=test) with the VAE.
    Returns (latents, attrs) where latents: (N, 32, 32, 4), attrs: (N, 40).
    """
    split_name = {0: 'train', 1: 'val', 2: 'test'}[partition_id]
    attr_file = os.path.join(celeba_root, 'list_attr_celeba.txt')
    part_file = os.path.join(celeba_root, 'list_eval_partition.txt')
    n_dev = jax.local_device_count()
    vae_rng = jax.random.PRNGKey(42 + partition_id)

    _, all_attrs_arr = load_celeba_attrs(attr_file)

    # Collect indices for this partition
    indices = []
    with open(part_file) as f:
        for i, line in enumerate(f):
            if int(line.strip().split()[1]) == partition_id:
                indices.append(i)
    indices = indices[:num_samples]
    attrs = all_attrs_arr[indices]

    # Filenames
    with open(attr_file) as f:
        lines = f.read().splitlines()
    all_fnames = [lines[i + 2].split()[0] for i in range(len(all_attrs_arr))]
    fnames = [all_fnames[i] for i in indices]
    print(f"  [{split_name}] {len(fnames)} images")

    ds = make_celeba_dataset(fnames, attrs, batch_size, celeba_root)

    all_latents, all_attrs_out = [], []
    pbar = tqdm.tqdm(total=len(fnames), desc=f'Encoding [{split_name}]')
    for batch_idx, (imgs_np, attrs_np) in enumerate(ds.as_numpy_iterator()):
        B = imgs_np.shape[0]
        imgs_bchw = imgs_np.transpose(0, 3, 1, 2)
        imgs_padded, orig_B = _pad_to_n(imgs_bchw, n_dev)
        imgs_sharded = _shard(imgs_padded, n_dev)
        batch_rng = jax.random.fold_in(vae_rng, batch_idx)
        rngs = jnp.stack([jax.random.fold_in(batch_rng, d) for d in range(n_dev)])
        latents_sharded = encode_pmap_fn(params_rep, rngs, imgs_sharded)
        latents = _unshard(latents_sharded, orig_B)
        all_latents.append(latents)
        all_attrs_out.append(attrs_np)
        pbar.update(B)
    pbar.close()

    latents = np.concatenate(all_latents, axis=0)
    attrs_out = np.concatenate(all_attrs_out, axis=0)
    print(f"  [{split_name}] Encoded {len(latents)} images. Shape: {latents.shape}")
    return latents, attrs_out


# ─── Model ────────────────────────────────────────────────────────────────────
def load_model(checkpoint_dir, model_size):
    from model import DiT
    from utils.train_state import TrainStateEma
    from utils.checkpoint import Checkpoint

    cfg = MODEL_CONFIGS[model_size]
    # CelebA model: num_classes=1, unconditional
    dit_args = dict(
        patch_size=cfg['patch_size'],
        hidden_size=cfg['hidden_size'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        out_channels=4,
        class_dropout_prob=0.0,
        num_classes=1,
        dropout=0.0,
        ignore_dt=False,
    )
    model_def = DiT(**dit_args)
    rng = jax.random.PRNGKey(0)
    example_x = jnp.zeros((1, 32, 32, 4))
    model_rngs = {'params': rng, 'label_dropout': rng, 'dropout': rng}
    params = model_def.init(
        model_rngs, example_x,
        jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32)
    )['params']

    train_state = TrainStateEma.create(model_def, params, rng=rng)
    cp = Checkpoint(checkpoint_dir)
    replace_dict = cp.load_as_dict()['train_state']
    del replace_dict['opt_state']
    train_state = train_state.replace(**replace_dict)
    print(f"Loaded checkpoint from {checkpoint_dir}")
    return train_state


# ─── Feature Extraction ───────────────────────────────────────────────────────
def make_extract_fn(train_state):
    """Returns a pmap'd extract function that shards a batch across all GPUs."""
    n_dev = jax.local_device_count()
    params_ema_rep = jax.device_put_replicated(train_state.params_ema, jax.local_devices())

    @jax.pmap
    def _p_extract(params_ema, x_noisy, t_vec, dt_base_vec):
        label_vec = jnp.zeros(x_noisy.shape[0], dtype=jnp.int32)
        _, _, activations = train_state.model_def.apply(
            {'params': params_ema}, x_noisy, t_vec, dt_base_vec, label_vec,
            train=False, return_activations=True
        )
        return activations

    def extract(x_t_np, t, dt_base, orig_B):
        """
        x_t_np : np.ndarray (B, 32, 32, 4)
        Returns activations dict with values (orig_B, num_patches, hidden_size).
        """
        imgs_padded, _ = _pad_to_n(x_t_np, n_dev)
        x_sharded  = _shard(imgs_padded, n_dev)               # (n_dev, local_B, 32, 32, 4)
        local_B    = x_sharded.shape[1]
        t_sharded  = np.full((n_dev, local_B), t,           dtype=np.float32)
        dt_sharded = np.full((n_dev, local_B), float(dt_base), dtype=np.float32)
        acts = _p_extract(params_ema_rep, x_sharded, t_sharded, dt_sharded)
        # acts: dict of (n_dev, local_B, num_patches, hidden_size)
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


def extract_all_features(train_state, latents, attrs, pairs, batch_size,
                         save_dir, selected_layers, split):
    """
    split: 'train' or 'val' — features are saved to {save_dir}/{split}/.
    """
    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    noise_rng = jax.random.PRNGKey(42)
    fixed_noise = np.array(jax.random.normal(noise_rng, latents.shape))

    # Save attrs once per split
    attrs_path = os.path.join(split_dir, 'attrs.npy')
    if not os.path.exists(attrs_path):
        np.save(attrs_path, attrs)

    extract_fn = make_extract_fn(train_state)

    print(f"[{split}] Extracting: {len(pairs)} pairs × {len(selected_layers)} layers "
          f"× {len(latents)} images  on {jax.local_device_count()} GPU(s)")
    for dt_base, t in tqdm.tqdm(pairs, desc=f'Pairs [{split}]'):
        path = feat_path(save_dir, dt_base, t, split)
        if os.path.exists(path):
            continue

        all_feats = []
        for i in range(0, len(latents), batch_size):
            x_clean_np = latents[i:i + batch_size]
            x_noise_np = fixed_noise[i:i + batch_size]
            B = x_clean_np.shape[0]

            x_t_np = (1 - (1 - 1e-5) * t) * x_noise_np + t * x_clean_np
            acts, orig_B = extract_fn(x_t_np, t, dt_base, B)

            batch_feats = []
            for lk in selected_layers:
                feat = _unshard(jnp.mean(acts[lk], axis=2), orig_B)  # (B, hidden)
                batch_feats.append(feat.astype(np.float16))
            all_feats.append(np.stack(batch_feats, axis=1))  # (B, L, H)

        np.save(path, np.concatenate(all_feats, axis=0))


# ─── Probe ────────────────────────────────────────────────────────────────────
def _run_pair_layer(dt_base, t, layer_idx, layer_name, save_dir, attr_indices, probe_c):
    """
    Worker: fit probe on train features, score on val features.
    liblinear releases the GIL → safe for thread-based parallelism.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_feats = np.load(feat_path(save_dir, dt_base, t, 'train'))  # (N_tr, L, H)
    val_feats   = np.load(feat_path(save_dir, dt_base, t, 'val'))    # (N_val, L, H)
    train_attrs = np.load(os.path.join(save_dir, 'train', 'attrs.npy'))
    val_attrs   = np.load(os.path.join(save_dir, 'val',   'attrs.npy'))

    X_tr  = train_feats[:, layer_idx, :].astype(np.float32)
    X_val = val_feats[:,   layer_idx, :].astype(np.float32)

    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)

    out = {}
    for ai in attr_indices:
        clf = LogisticRegression(C=probe_c, max_iter=200, solver='liblinear')
        clf.fit(X_tr, train_attrs[:, ai])
        out[(dt_base, t, layer_name, CELEBA_ATTRS[ai])] = clf.score(X_val, val_attrs[:, ai])
    return out


def run_all_probes(pairs, save_dir, selected_layers, probe_c=1.0,
                   attr_indices=None):
    """
    Train on CelebA train-split features, eval on val-split features.
    Parallelises (pair × layer) jobs across all CPU cores.
    Returns dict: {(dt_base, t, layer_name, attr_name): accuracy}
    """
    from joblib import Parallel, delayed

    if attr_indices is None:
        attr_indices = list(range(len(CELEBA_ATTRS)))

    jobs = [
        (dt_base, t, li, ln)
        for dt_base, t in pairs
        if (os.path.exists(feat_path(save_dir, dt_base, t, 'train')) and
            os.path.exists(feat_path(save_dir, dt_base, t, 'val')))
        for li, ln in enumerate(selected_layers)
    ]
    print(f"Training probes: {len(jobs)} jobs "
          f"({len(pairs)} pairs × {len(selected_layers)} layers) "
          f"× {len(attr_indices)} attrs  [parallel, all CPUs]")

    all_out = Parallel(n_jobs=-1, prefer='threads', verbose=1)(
        delayed(_run_pair_layer)(dt_base, t, li, ln, save_dir, attr_indices, probe_c)
        for dt_base, t, li, ln in jobs
    )

    results = {}
    for d in all_out:
        results.update(d)
    return results


# ─── Save & Plot ──────────────────────────────────────────────────────────────
def save_results_csv(results, output_dir):
    import csv
    path = os.path.join(output_dir, 'probe_celeba_results.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dt_base', 't', 'layer', 'attribute', 'accuracy'])
        for (dt_base, t, layer, attr), acc in results.items():
            w.writerow([dt_base, f"{t:.6f}", layer, attr, f"{acc:.4f}"])
    print(f"Results → {path}")


def plot_results(results, pairs, selected_layers, dt_base_max, output_dir,
                 key_attrs=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    last_layer = selected_layers[-1]
    if key_attrs is None:
        key_attrs = KEY_ATTRS

    # ── 1. Heatmap: mean accuracy across key_attrs, per (t, dt_base) ─────────
    n_rows = dt_base_max + 1
    n_cols = 2 ** dt_base_max

    grid = np.full((n_rows, n_cols), np.nan)
    for dt_base, t in pairs:
        accs = [results.get((dt_base, t, last_layer, a), np.nan) for a in key_attrs]
        accs_valid = [a for a in accs if not np.isnan(a)]
        if accs_valid:
            k = round(t * (2 ** dt_base))
            col = k * (n_cols // (2 ** dt_base))
            grid[dt_base, col] = np.mean(accs_valid)

    fig, ax = plt.subplots(figsize=(16, 5))
    vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1.0
    vmin = max(0.5, np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.5
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='RdYlGn',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Mean Accuracy (key attributes)')
    ax.set_xlabel('t  (0 = pure noise → 1 = clean)')
    ax.set_ylabel('dt_base  (step size = 1/2^dt_base)')
    ax.set_title(f'CelebA Linear Probe — Mean Attribute Accuracy\n({last_layer})')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f'dt_base={b}  (d=1/{2**b})' for b in range(n_rows)])
    xt = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks([round(v * (n_cols - 1)) for v in xt])
    ax.set_xticklabels([f'{v:.2f}' for v in xt])
    plt.tight_layout()
    out = os.path.join(output_dir, 'heatmap_mean_accuracy.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── 2. Per-attribute accuracy at t=1.0 (clean) vs dt_base ────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    dt_bases = list(range(dt_base_max + 1))
    width = 0.8 / len(key_attrs)
    for i, attr in enumerate(key_attrs):
        accs = []
        for db in dt_bases:
            # Use highest t for this dt_base (cleanest image)
            t_max = (2 ** db - 1) / (2 ** db)
            accs.append(results.get((db, t_max, last_layer, attr), 0.0))
        x = np.arange(len(dt_bases)) + i * width
        ax.bar(x, accs, width=width, label=attr)
    ax.set_xticks(np.arange(len(dt_bases)) + 0.4)
    ax.set_xticklabels([f'dt={b}\n(d=1/{2**b})' for b in dt_bases])
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Per-Attribute Accuracy at Highest t (Cleanest Image) vs Step Size')
    ax.legend(fontsize=7, ncol=5, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'attr_accuracy_by_dtbase.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── 3. Line plot: mean accuracy vs t for each dt_base ────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for dt_base in range(dt_base_max + 1):
        ts_for_db = sorted([t for db, t in pairs if db == dt_base])
        mean_accs = []
        for t in ts_for_db:
            accs = [results.get((dt_base, t, last_layer, a), np.nan) for a in key_attrs]
            valid = [a for a in accs if not np.isnan(a)]
            mean_accs.append(np.mean(valid) if valid else np.nan)
        ax.plot(ts_for_db, mean_accs, marker='o',
                label=f'dt_base={dt_base} (d=1/{2**dt_base})')
    ax.set_xlabel('t (noise level)')
    ax.set_ylabel('Mean Accuracy (key attributes)')
    ax.set_title(f'CelebA Attribute Probe: Mean Accuracy vs t\n({last_layer})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    out = os.path.join(output_dir, 'lineplot_mean_accuracy.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── 4. Heatmap per key attribute ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, attr in enumerate(key_attrs[:10]):
        grid_attr = np.full((n_rows, n_cols), np.nan)
        for dt_base, t in pairs:
            acc = results.get((dt_base, t, last_layer, attr), np.nan)
            if not np.isnan(acc):
                k = round(t * (2 ** dt_base))
                col = k * (n_cols // (2 ** dt_base))
                grid_attr[dt_base, col] = acc
        ax2 = axes[idx]
        im = ax2.imshow(grid_attr, aspect='auto', origin='lower',
                        cmap='RdYlGn', vmin=0.5, vmax=1.0, interpolation='nearest')
        ax2.set_title(attr, fontsize=10)
        ax2.set_xlabel('t')
        ax2.set_ylabel('dt_base')
        ax2.set_yticks(range(n_rows))
        ax2.set_yticklabels([f'{b}' for b in range(n_rows)], fontsize=7)
    plt.suptitle(f'Per-Attribute Linear Probe Accuracy ({last_layer})', fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, 'heatmap_per_attribute.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved: {out}")

    # ── 5. Summary table: best (t, dt_base) per attribute ────────────────────
    print("\n=== Summary: Best (dt_base, t) per attribute ===")
    print(f"{'Attribute':<22} {'Best acc':>9}  {'dt_base':>7}  {'t':>6}")
    print("-" * 52)
    for attr in key_attrs:
        best_acc, best_key = 0.0, None
        for dt_base, t in pairs:
            acc = results.get((dt_base, t, last_layer, attr), 0.0)
            if acc > best_acc:
                best_acc, best_key = acc, (dt_base, t)
        if best_key:
            print(f"  {attr:<20} {best_acc:>9.4f}  {best_key[0]:>7}  {best_key[1]:>6.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = get_all_pairs(args.dt_base_max)
    print(f"(t, dt_base) pairs: {len(pairs)}  (dt_base_max={args.dt_base_max})")

    cfg = MODEL_CONFIGS[args.model_size]
    selected_layers = get_selected_layers(args.layers, cfg['depth'])
    print(f"Layers: {selected_layers}")

    # Key attribute indices
    key_attr_indices = [CELEBA_ATTRS.index(a) for a in KEY_ATTRS]

    # ── Feature extraction ────────────────────────────────────────────────────
    if not args.skip_extraction:
        # Set up VAE once for both splits
        params_rep, encode_pmap_fn = setup_vae_pmap(CELEBA_ROOT)

        # Encode train split (partition 0)
        train_latents, train_attrs = encode_celeba_split(
            0, args.num_train_samples, args.batch_size,
            params_rep, encode_pmap_fn, CELEBA_ROOT
        )

        # Encode val split (partition 1)
        val_latents, val_attrs = encode_celeba_split(
            1, args.num_val_samples, args.batch_size,
            params_rep, encode_pmap_fn, CELEBA_ROOT
        )

        # Load DiT model
        train_state = load_model(args.checkpoint_dir, args.model_size)

        # Extract features for both splits
        extract_all_features(
            train_state, train_latents, train_attrs, pairs,
            args.batch_size, args.save_dir, selected_layers, split='train'
        )
        extract_all_features(
            train_state, val_latents, val_attrs, pairs,
            args.batch_size, args.save_dir, selected_layers, split='val'
        )
    else:
        print("Skipping extraction (--skip_extraction).")

    # ── Linear probing ────────────────────────────────────────────────────────
    if args.skip_probe:
        print("Skipping probe (--skip_probe).")
        return

    csv_path = os.path.join(args.output_dir, 'probe_celeba_results.csv')
    if args.skip_extraction and os.path.exists(csv_path):
        import csv
        results = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = (int(row['dt_base']), float(row['t']),
                       row['layer'], row['attribute'])
                results[key] = float(row['accuracy'])
        print(f"Loaded {len(results)} probe results from {csv_path}")
    else:
        results = run_all_probes(
            pairs, args.save_dir, selected_layers,
            probe_c=args.probe_c,
            attr_indices=key_attr_indices,
        )
        save_results_csv(results, args.output_dir)

    # ── Plots & summary ───────────────────────────────────────────────────────
    plot_results(results, pairs, selected_layers, args.dt_base_max,
                 args.output_dir, key_attrs=KEY_ATTRS)
    print("\nAll done.")


if __name__ == '__main__':
    main()
