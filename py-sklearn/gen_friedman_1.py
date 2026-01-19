import pandas as pd
from sklearn.datasets import make_friedman1

# 設定
N_SAMPLES = 5000
N_FEATURES = 90
NOISE = 0.1
RANDOM_SEED = 42
N_TARGETS = 3
OUTPUT_FILE = f"../Dataset/friedman_data_n{N_SAMPLES}_d{N_FEATURES}_t{N_TARGETS}.csv"

print(f"Generating Friedman #1 dataset: N={N_SAMPLES}, D={N_FEATURES}, T={N_TARGETS} targets...")

# 特徴量を生成（3つのターゲットで共通）
X, _ = make_friedman1(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=0.0, random_state=RANDOM_SEED)

# 3つの異なるターゲットを生成（異なるrandom_stateでノイズを追加）
targets = []
for i in range(N_TARGETS):
    _, y = make_friedman1(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=NOISE, random_state=RANDOM_SEED + i)
    targets.append(y)

# データフレーム化
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(N_FEATURES)])
for i in range(N_TARGETS):
    df[f'target_{i+1}'] = targets[i]

# CSV出力 (ヘッダーあり、浮動小数点精度を確保)
df.to_csv(OUTPUT_FILE, index=False, header=True, float_format='%.6f')

print(f"Saved to {OUTPUT_FILE}")
print(f"Shape: {df.shape} (rows={df.shape[0]}, columns={df.shape[1]})")
print(f"Columns: {N_FEATURES} features + {N_TARGETS} targets = {N_FEATURES + N_TARGETS} total columns")
