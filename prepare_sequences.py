import os, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    'cpu_percent', 'memory_rss_mb', 'memory_vms_mb',
    'num_threads', 'num_handles',
    'io_read_bytes_delta', 'io_write_bytes_delta',
    'io_read_count_delta', 'io_write_count_delta',
    'net_bytes_sent_delta', 'net_bytes_recv_delta',
    'num_connections', 'num_open_files',
    'cpu_user_time_delta', 'cpu_system_time_delta',
]

def load_all_sessions(input_dir):
    csvs = glob.glob(os.path.join(input_dir, '**/*.csv'), recursive=True)
    dfs = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                dfs.append(df)
                print(f"Loaded: {path} ({len(df)} rows)")
        except Exception as e:
            print(f"Skipped {path}: {e}")
    return pd.concat(dfs, ignore_index=True)

def build_sequences(df, seq_len=10, step=1):
    sequences = []
    labels = []
    for (session, pid), group in df.groupby(['session_id', 'pid']):
        group = group.sort_values('step_number').reset_index(drop=True)
        feat = group[FEATURE_COLS].fillna(0).values
        labs = group['label'].values
        if len(feat) < seq_len:
            continue
        for i in range(0, len(feat) - seq_len + 1, step):
            window = feat[i : i + seq_len]
            label = int(np.round(np.mean(labs[i : i + seq_len])))
            sequences.append(window)
            labels.append(label)
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

def main(input_dir, output_path, seq_len=10):
    print(f"Loading sessions from: {input_dir}")
    df = load_all_sessions(input_dir)
    print(f"Total rows: {len(df):,}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Label dist: {dict(df['label'].value_counts())}")

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].fillna(0))

    print(f"\nBuilding sequences (seq_len={seq_len})...")
    X, y = build_sequences(df, seq_len=seq_len)

    print(f"Sequences shape: {X.shape}")
    print(f"Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    np.savez_compressed(output_path, X=X, y=y)
    joblib.dump(scaler, output_path.replace('.npz', '_scaler.pkl'))
    print(f"\nSaved → {output_path}")
    print(f"Saved → {output_path.replace('.npz', '_scaler.pkl')}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seq_len', type=int, default=10)
    args = parser.parse_args()
    main(args.input_dir, args.output, args.seq_len)