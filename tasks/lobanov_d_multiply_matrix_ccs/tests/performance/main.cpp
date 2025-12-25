#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"
#include "lobanov_d_multiply_matrix_ccs/mpi/include/ops_mpi.hpp"
#include "lobanov_d_multiply_matrix_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lobanov_d_multiply_matrix_ccs {

CompressedColumnMatrix CreateRandomCompressedColumnMatrix(int row_count, int column_count, double density_factor,
                                                          int seed = 42) {
  CompressedColumnMatrix result_matrix;
  result_matrix.row_count = row_count;
  result_matrix.column_count = column_count;

  // Детерминированный генератор с фиксированным seed
  std::mt19937 rng(seed);

  // Используем хеш от параметров для создания уникального seed для каждой комбинации
  std::hash<std::string> hasher;
  std::string param_hash =
      std::to_string(row_count) + "_" + std::to_string(column_count) + "_" + std::to_string(density_factor);
  rng.seed(static_cast<unsigned int>(seed + hasher(param_hash)));

  std::uniform_real_distribution<double> val_dist(0.1, 10.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  std::vector<std::vector<int>> row_indices_per_column(column_count);
  std::vector<std::vector<double>> values_per_column(column_count);
  int nnz_counter = 0;

  for (int j = 0; j < column_count; ++j) {
    for (int i = 0; i < row_count; ++i) {
      if (prob_dist(rng) < density_factor) {
        row_indices_per_column[j].push_back(i);
        values_per_column[j].push_back(val_dist(rng));
        ++nnz_counter;
      }
    }
  }

  result_matrix.non_zero_count = nnz_counter;
  result_matrix.value_data.reserve(nnz_counter);
  result_matrix.row_index_data.reserve(nnz_counter);
  result_matrix.column_pointer_data.resize(column_count + 1);

  int offset = 0;
  result_matrix.column_pointer_data[0] = 0;

  for (int j = 0; j < column_count; ++j) {
    for (size_t k = 0; k < row_indices_per_column[j].size(); ++k) {
      result_matrix.row_index_data.push_back(row_indices_per_column[j][k]);
      result_matrix.value_data.push_back(values_per_column[j][k]);
    }
    offset += row_indices_per_column[j].size();
    result_matrix.column_pointer_data[j + 1] = offset;
  }

  return result_matrix;
}

// Базовый класс для производительных тестов
class LobanovDMultiplyMatrixPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Используем параметры из GetTestParams()
    auto params = GetTestParams();
    int dimension = std::get<0>(params);
    double density = std::get<1>(params);
    int seed_offset = std::get<2>(params);

    first_matrix_ = CreateRandomCompressedColumnMatrix(dimension, dimension, density, 100 + seed_offset);
    second_matrix_ = CreateRandomCompressedColumnMatrix(dimension, dimension, density, 200 + seed_offset);

    input_data_ = std::make_pair(first_matrix_, second_matrix_);
  }

  // Возвращает параметры теста: размер, плотность, смещение для seed
  virtual std::tuple<int, double, int> GetTestParams() const {
    return {2000, 0.1, 0};  // По умолчанию
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.row_count == 0 && output_data.column_count == 0) {
      return true;
    }

    return output_data.row_count == first_matrix_.row_count &&
           output_data.column_count == second_matrix_.column_count &&
           output_data.column_pointer_data.size() == static_cast<size_t>(output_data.column_count) + 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  CompressedColumnMatrix first_matrix_, second_matrix_;
  InType input_data_;
};

// Тест 1: Малые матрицы (быстрый запуск)
class SmallMatrixPerfTest : public LobanovDMultiplyMatrixPerfTest {
 protected:
  std::tuple<int, double, int> GetTestParams() const override {
    return {500, 0.1, 1};  // 500x500, плотность 10%
  }
};

// Тест 2: Средние матрицы
class MediumMatrixPerfTest : public LobanovDMultiplyMatrixPerfTest {
 protected:
  std::tuple<int, double, int> GetTestParams() const override {
    return {1000, 0.08, 2};  // 1000x1000, плотность 8%
  }
};

// Тест 3: Большие матрицы
class LargeMatrixPerfTest : public LobanovDMultiplyMatrixPerfTest {
 protected:
  std::tuple<int, double, int> GetTestParams() const override {
    return {2000, 0.05, 3};  // 2000x2000, плотность 5%
  }
};

// Тесты для разных размеров матриц
TEST_P(SmallMatrixPerfTest, SmallMatrixPerformance) {
  ExecuteTest(GetParam());
}

TEST_P(MediumMatrixPerfTest, MediumMatrixPerformance) {
  ExecuteTest(GetParam());
}

TEST_P(LargeMatrixPerfTest, LargeMatrixPerformance) {
  ExecuteTest(GetParam());
}

// Создаем задачи производительности для всех реализаций
const auto kAllPerformanceTasks =
    ppc::util::MakeAllPerfTasks<InType, LobanovDMultiplyMatrixMPI, LobanovDMultiplyMatrixSEQ>(
        PPC_SETTINGS_lobanov_d_multiply_matrix_ccs);

const auto kGtestPerformanceValues = ppc::util::TupleToGTestValues(kAllPerformanceTasks);

// Регистрируем три набора тестов
INSTANTIATE_TEST_SUITE_P(SmallMatrixTests, SmallMatrixPerfTest, kGtestPerformanceValues,
                         LobanovDMultiplyMatrixPerfTest::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(MediumMatrixTests, MediumMatrixPerfTest, kGtestPerformanceValues,
                         LobanovDMultiplyMatrixPerfTest::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(LargeMatrixTests, LargeMatrixPerfTest, kGtestPerformanceValues,
                         LobanovDMultiplyMatrixPerfTest::CustomPerfTestName);

}  // namespace lobanov_d_multiply_matrix_ccs
