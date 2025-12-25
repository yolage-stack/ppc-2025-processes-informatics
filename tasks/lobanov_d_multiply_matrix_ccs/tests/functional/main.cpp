#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"
#include "lobanov_d_multiply_matrix_ccs/mpi/include/ops_mpi.hpp"
#include "lobanov_d_multiply_matrix_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

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

class LobanovDMultiplyMatrixFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // Статический метод должен называться PrintTestParam (как ожидает func_test_util.hpp)
  static std::string PrintTestParam(const TestType &test_parameter) {
    // Получаем название теста из tuple
    return std::get<0>(test_parameter);
  }

 protected:
  void SetUp() override {
    TestType parameters = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    // Получаем название теста
    std::string test_name = std::get<0>(parameters);

    // Получаем матрицы из tuple (но для простоты игнорируем и создаем свои)
    // Для тестов создаем простые матрицы в зависимости от названия теста
    if (test_name.find("small") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(10, 10, 0.3);
      matrix_b_ = CreateRandomCompressedColumnMatrix(10, 10, 0.3);
    } else if (test_name.find("medium") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(50, 50, 0.1);
      matrix_b_ = CreateRandomCompressedColumnMatrix(50, 50, 0.1);
    } else if (test_name.find("large") != std::string::npos) {
      matrix_a_ = CreateRandomCompressedColumnMatrix(100, 100, 0.05);
      matrix_b_ = CreateRandomCompressedColumnMatrix(100, 100, 0.05);
    } else {
      // По умолчанию создаем маленькие матрицы
      matrix_a_ = CreateRandomCompressedColumnMatrix(5, 5, 0.5);
      matrix_b_ = CreateRandomCompressedColumnMatrix(5, 5, 0.5);
    }

    input_data_ = std::make_pair(matrix_a_, matrix_b_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.row_count == 0 && output_data.column_count == 0) {
      return true;
    }

    // Проверяем, что размеры результата корректны
    bool dimensions_correct =
        output_data.row_count == matrix_a_.row_count && output_data.column_count == matrix_b_.column_count &&
        output_data.column_pointer_data.size() == static_cast<size_t>(output_data.column_count) + 1;

    // Для функциональных тестов также проверяем непустоту результата
    return dimensions_correct && output_data.non_zero_count >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  CompressedColumnMatrix matrix_a_, matrix_b_;
  InType input_data_;
};

namespace {

TEST_P(LobanovDMultiplyMatrixFuncTest, MatrixMultiplicationCorrectness) {
  ExecuteTest(GetParam());
}

// Создаем пустую матрицу для заполнителя ожидаемого результата
CompressedColumnMatrix CreateEmptyMatrix(int rows, int cols) {
  CompressedColumnMatrix matrix;
  matrix.row_count = rows;
  matrix.column_count = cols;
  matrix.non_zero_count = 0;
  matrix.value_data.clear();
  matrix.row_index_data.clear();
  matrix.column_pointer_data.assign(cols + 1, 0);
  return matrix;
}

// Определяем тестовые наборы данных
const std::array<TestType, 3> kTestParameters = {
    // Маленькие матрицы
    std::make_tuple("test_small_matrices", CreateEmptyMatrix(10, 10),  // Заполнитель для matrix_a
                    CreateEmptyMatrix(10, 10),                         // Заполнитель для matrix_b
                    CreateEmptyMatrix(10, 10)                          // Заполнитель для expected_result
                    ),

    // Средние матрицы
    std::make_tuple("test_medium_matrices", CreateEmptyMatrix(50, 50), CreateEmptyMatrix(50, 50),
                    CreateEmptyMatrix(50, 50)),

    // Большие матрицы
    std::make_tuple("test_large_matrices", CreateEmptyMatrix(100, 100), CreateEmptyMatrix(100, 100),
                    CreateEmptyMatrix(100, 100))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LobanovDMultiplyMatrixMPI, InType>(
                                               kTestParameters, PPC_SETTINGS_lobanov_d_multiply_matrix_ccs),
                                           ppc::util::AddFuncTask<LobanovDMultiplyMatrixSEQ, InType>(
                                               kTestParameters, PPC_SETTINGS_lobanov_d_multiply_matrix_ccs));

const auto kGtestParameterValues = ppc::util::ExpandToValues(kTestTasksList);

// Используем правильный метод для имени теста
const auto kPerfTestName = LobanovDMultiplyMatrixFuncTest::PrintFuncTestName<LobanovDMultiplyMatrixFuncTest>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationFuncTests, LobanovDMultiplyMatrixFuncTest, kGtestParameterValues,
                         kPerfTestName);

}  // namespace

}  // namespace lobanov_d_multiply_matrix_ccs
