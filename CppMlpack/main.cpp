#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <armadillo>
#include <nlohmann/json.hpp>
#include "csv_loader.hpp"

using namespace mlpack;
using namespace arma;

// ガウス過程回帰クラス
class GaussianProcessRegression
{
private:
	mat X_train;		   // 訓練データ (D x N)
	rowvec y_train;		   // 訓練ターゲット (1 x N)
	GaussianKernel kernel; // RBFカーネル
	double noise_variance; // ノイズ分散
	mat K_inv;			   // カーネル行列の逆行列
	bool trained;

public:
	GaussianProcessRegression(double bandwidth = 1.0, double noise = 0.1)
		: kernel(bandwidth), noise_variance(noise), trained(false) {}

	// カーネル行列を計算
	mat compute_kernel_matrix(const mat &X1, const mat &X2)
	{
		size_t n1 = X1.n_cols;
		size_t n2 = X2.n_cols;
		mat K(n1, n2);

		for (size_t i = 0; i < n1; ++i)
		{
			for (size_t j = 0; j < n2; ++j)
			{
				K(i, j) = kernel.Evaluate(X1.col(i), X2.col(j));
			}
		}
		return K;
	}

	// 訓練
	double train(const mat &X, const rowvec &y)
	{
		X_train = X;
		y_train = y;

		// カーネル行列を計算
		mat K = compute_kernel_matrix(X_train, X_train);

		// ノイズ項を対角に追加
		K += eye<mat>(K.n_rows, K.n_cols) * noise_variance;

		// Cholesky分解を使用して逆行列を計算（数値的安定性のため）
		mat L;
		bool success = chol(L, K, "lower");
		if (!success)
		{
			throw std::runtime_error("カーネル行列のCholesky分解に失敗しました");
		}

		// K_inv = (L^T)^(-1) * L^(-1)
		mat L_inv = inv(trimatl(L));
		K_inv = L_inv.t() * L_inv;

		trained = true;

		// 対数マージナル尤度を計算
		double log_likelihood = compute_log_likelihood(K, L);
		return log_likelihood;
	}

	// 対数マージナル尤度を計算
	double compute_log_likelihood(const mat &K, const mat &L)
	{
		// log p(y|X) = -0.5 * y^T * K^(-1) * y - 0.5 * log|K| - 0.5 * n * log(2π)
		rowvec alpha = y_train * K_inv;
		double data_term = -0.5 * as_scalar(alpha * y_train.t());

		// log|K| = 2 * sum(log(diag(L)))
		double log_det = 2.0 * sum(log(diagvec(L)));

		double const_term = -0.5 * X_train.n_cols * std::log(2.0 * 3.14159265358979323846);

		return data_term - 0.5 * log_det + const_term;
	}

	// 予測
	void predict(const mat &X_test, vec &y_pred, vec &y_var)
	{
		if (!trained)
		{
			throw std::runtime_error("モデルが訓練されていません");
		}

		size_t n_test = X_test.n_cols;
		y_pred.set_size(n_test);
		y_var.set_size(n_test);

		// テスト点と訓練点の間のカーネル行列
		mat K_star = compute_kernel_matrix(X_train, X_test);

		// 予測平均: μ* = K_star^T * K_inv * y_train^T
		rowvec alpha = y_train * K_inv;
		y_pred = (K_star.t() * alpha.t());

		// 予測分散: σ*^2 = k(x*, x*) - K_star^T * K_inv * K_star
		for (size_t i = 0; i < n_test; ++i)
		{
			double k_star_star = kernel.Evaluate(X_test.col(i), X_test.col(i));
			rowvec k_star_i = K_star.col(i).t();
			double var = k_star_star - as_scalar(k_star_i * K_inv * k_star_i.t());
			y_var(i) = std::max(var, 0.0); // 分散は非負
		}
	}

	// カーネルのバンド幅を取得
	double get_bandwidth() const
	{
		return kernel.Bandwidth();
	}

	// ノイズ分散を取得
	double get_noise_variance() const
	{
		return noise_variance;
	}
};

int main()
{
	// データセットパス
	std::string dataset_path = "../../../../Dataset/friedman_data_n5000_d90_t3.csv";

	std::cout << "データセットを読み込み中: " << dataset_path << std::endl;

	// CSVデータの読み込み
	Dataset raw_data;
	try
	{
		raw_data = load_csv(dataset_path, 90);
	}
	catch (const std::exception &e)
	{
		std::cerr << "エラー: " << e.what() << std::endl;
		return -1;
	}

	std::cout << "データセット読み込み完了: N=" << raw_data.N << ", D=" << raw_data.D << std::endl;

	// Armadillo形式へ変換 (mlpack仕様: 列 = 1データ)
	mat X(raw_data.D, raw_data.N);
	rowvec y(raw_data.N);

	for (int i = 0; i < raw_data.N; ++i)
	{
		for (int j = 0; j < raw_data.D; ++j)
		{
			X(j, i) = raw_data.X[i][j];
		}
		y(i) = raw_data.y[i];
	}

	// データの正規化 (平均0, 分散1)
	vec X_mean = arma::mean(X, 1);
	vec X_stddev = arma::stddev(X, 0, 1);
	// ゼロ除算回避
	X_stddev.elem(find(X_stddev < 1e-6)).fill(1.0);

	X.each_col() -= X_mean;
	X.each_col() /= X_stddev;

	double y_mean = arma::mean(y);
	double y_std = arma::stddev(y);
	y -= y_mean;
	y /= (y_std > 1e-6 ? y_std : 1.0);

	std::cout << "\nガウス過程回帰の訓練を開始 (mlpack)..." << std::endl;
	std::cout << "サンプル数: " << raw_data.N << ", 特徴量次元: " << raw_data.D << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	// ガウス過程回帰モデルの作成と訓練
	// ハイパーパラメータ: バンド幅=1.0, ノイズ分散=0.1
	GaussianProcessRegression gp(1.0, 0.1);

	double log_likelihood;
	try
	{
		log_likelihood = gp.train(X, y);
	}
	catch (const std::exception &e)
	{
		std::cerr << "訓練エラー: " << e.what() << std::endl;
		return -1;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << "\n訓練完了!" << std::endl;
	std::cout << "経過時間: " << diff.count() << " 秒" << std::endl;
	std::cout << "対数マージナル尤度: " << log_likelihood << std::endl;
	std::cout << "カーネルバンド幅: " << gp.get_bandwidth() << std::endl;
	std::cout << "ノイズ分散: " << gp.get_noise_variance() << std::endl;

	// 結果をJSONファイルに出力
	using json = nlohmann::json;

	// タイムスタンプを取得
	auto now = std::chrono::system_clock::now();
	auto time_t = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss;
#ifdef _WIN32
	tm timeinfo;
	localtime_s(&timeinfo, &time_t);
	ss << std::put_time(&timeinfo, "%Y-%m-%dT%H:%M:%S");
#else
	ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
#endif
	auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
				  now.time_since_epoch()) %
			  1000000;
	ss << '.' << std::setfill('0') << std::setw(6) << ms.count();

	// データセットファイル名を取得
	std::string dataset_filename = dataset_path;
	size_t last_slash = dataset_path.find_last_of("/\\");
	if (last_slash != std::string::npos)
	{
		dataset_filename = dataset_path.substr(last_slash + 1);
	}

	json results = {
		{"timestamp", ss.str()},
		{"dataset", dataset_filename},
		{"n_samples", raw_data.N},
		{"n_features", raw_data.D},
		{"target", "target_1"},
		{"kernel", "RBF(bandwidth=" + std::to_string(gp.get_bandwidth()) + ") + WhiteKernel(noise=" + std::to_string(gp.get_noise_variance()) + ")"},
		{"n_restarts_optimizer", 0},
		{"elapsed_time_sec", diff.count()},
		{"log_marginal_likelihood", log_likelihood},
		{"optimized_kernel", "RBF(bandwidth=" + std::to_string(gp.get_bandwidth()) + ") + WhiteKernel(noise=" + std::to_string(gp.get_noise_variance()) + ")"}};

	// 出力ファイルパス
	std::string output_file = "mlpack_gp_regression_results.json";

	// JSONファイルに出力
	std::ofstream ofs(output_file);
	if (ofs.is_open())
	{
		ofs << results.dump(2) << std::endl;
		ofs.close();
		std::cout << "\n結果を保存しました: " << output_file << std::endl;
	}
	else
	{
		std::cerr << "警告: 結果ファイルの書き込みに失敗しました: " << output_file << std::endl;
	}

	return 0;
}
