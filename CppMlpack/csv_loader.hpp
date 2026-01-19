#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <csv.hpp>

using namespace csv;

// CSVデータを格納する構造体
struct Dataset
{
    int N;                              // サンプル数
    int D;                              // 特徴量の次元数
    std::vector<std::vector<double>> X; // 特徴量データ [N][D]
    std::vector<double> y;              // ターゲットデータ [N]
};

// CSVファイルを読み込む関数
Dataset load_csv(const std::string &filepath, int n_features)
{
    Dataset data;
    data.D = n_features;
    data.N = 0;

    try
    {
        CSVReader reader(filepath);

        for (CSVRow &row : reader)
        {
            std::vector<double> feature_row;
            feature_row.reserve(n_features);

            // 特徴量を読み込む (feature_0 から feature_89)
            for (int i = 0; i < n_features; ++i)
            {
                std::string col_name = "feature_" + std::to_string(i);
                double value = row[col_name].get<double>();
                feature_row.push_back(value);
            }

            // ターゲットを読み込む
            double target = row["target_1"].get<double>();

            data.X.push_back(feature_row);
            data.y.push_back(target);
            data.N++;
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("CSV読み込みエラー: " + std::string(e.what()));
    }

    if (data.N == 0)
    {
        throw std::runtime_error("データが読み込めませんでした: " + filepath);
    }

    return data;
}
