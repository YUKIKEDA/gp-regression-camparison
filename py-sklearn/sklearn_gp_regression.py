import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# データセットのパス
dataset_path = r"..\Dataset\friedman_data_n5000_d90_t3.csv"

# データ読み込み
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

# 特徴量とターゲットを分離
# 特徴量: feature_0 から feature_89 まで（90個）
feature_cols = [col for col in df.columns if col.startswith('feature_')]
# ターゲット: target_1, target_2, target_3（最初のターゲットを使用）
target_col = 'target_1'

X = df[feature_cols].values
y = df[target_col].values

N, D = X.shape
print(f"Dataset loaded: N={N}, D={D}")
print(f"Target: {target_col}")

# カーネル定義 (RBF + ノイズ項)
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)

print(f"\nStart training (sklearn): N={N}, D={D}")
start = time.time()

# 学習 (ハイパーパラメータ探索含む)
model.fit(X, y)

end = time.time()
elapsed_time = end - start

log_marginal_likelihood = model.log_marginal_likelihood()

print(f"Elapsed time: {elapsed_time:.4f} sec")
print("Log Marginal Likelihood:", log_marginal_likelihood)

# 結果をファイルに出力
results = {
    "timestamp": datetime.now().isoformat(),
    "dataset": Path(dataset_path).name,
    "n_samples": int(N),
    "n_features": int(D),
    "target": target_col,
    "kernel": str(kernel),
    "n_restarts_optimizer": model.n_restarts_optimizer,
    "elapsed_time_sec": float(elapsed_time),
    "log_marginal_likelihood": float(log_marginal_likelihood),
    "optimized_kernel": str(model.kernel_)
}

# 出力ファイルパス
output_file = Path(__file__).parent / "sklearn_gp_regression_results.json"

# JSONファイルに出力
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_file}")
