#include "lobanov_d_multiply_matrix_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "lobanov_d_multiply_matrix_ccs/common/include/common.hpp"

namespace lobanov_d_multiply_matrix_ccs {

namespace {
constexpr double EPSILON_THRESHOLD = 1e-10;
}  // namespace

LobanovDMultiplyMatrixMPI::LobanovDMultiplyMatrixMPI(const InType &input_matrices) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input_matrices;
  GetOutput() = CompressedColumnMatrix{};
}

bool LobanovDMultiplyMatrixMPI::ValidationImpl() {
  const auto &[matrix_a, matrix_b] = GetInput();
  return (matrix_a.column_count == matrix_b.row_count && matrix_a.row_count > 0 && matrix_a.column_count > 0 && 
          matrix_b.row_count > 0 && matrix_b.column_count > 0);
}

bool LobanovDMultiplyMatrixMPI::PreProcessingImpl() {
  return true;
}

void LobanovDMultiplyMatrixMPI::ComputeTransposedMatrixMPI(const CompressedColumnMatrix &source_matrix, CompressedColumnMatrix &transposed_result) {
  transposed_result.row_count = source_matrix.column_count;
  transposed_result.column_count = source_matrix.row_count;
  transposed_result.non_zero_count = source_matrix.non_zero_count;

  if (source_matrix.non_zero_count == 0) {
    transposed_result.value_data.clear();
    transposed_result.row_index_data.clear();
    transposed_result.column_pointer_data.assign(transposed_result.column_count + 1, 0);
    return;
  }

  std::vector<int> row_element_count(transposed_result.column_count, 0);
  for (int element_index = 0; element_index < source_matrix.non_zero_count; element_index++) {
    row_element_count[source_matrix.row_index_data[element_index]]++;
  }

  transposed_result.column_pointer_data.resize(transposed_result.column_count + 1);
  transposed_result.column_pointer_data[0] = 0;
  for (int row_index = 0; row_index < transposed_result.column_count; row_index++) {
    transposed_result.column_pointer_data[row_index + 1] = transposed_result.column_pointer_data[row_index] + row_element_count[row_index];
  }

  transposed_result.value_data.resize(source_matrix.non_zero_count);
  transposed_result.row_index_data.resize(source_matrix.non_zero_count);

  std::vector<int> current_position_array(transposed_result.column_count, 0);
  int column_counter = 0;

  while (column_counter < source_matrix.column_count) {
      int column_start = source_matrix.column_pointer_data[column_counter];
      int column_end = source_matrix.column_pointer_data[column_counter + 1];
      int element_counter = column_start;
    
      while (element_counter < column_end) {
        int row_position = source_matrix.row_index_data[element_counter];
        double element_value = source_matrix.value_data[element_counter];
        
        int target_slot = transposed_result.column_pointer_data[row_position] + 
                         current_position_array[row_position];
        transposed_result.value_data[target_slot] = element_value;
        transposed_result.row_index_data[target_slot] = column_counter;
        
        current_position_array[row_position]++;
        element_counter++;
    }
    column_counter++;
}
}

std::pair<int, int> LobanovDMultiplyMatrixMPI::DetermineColumnDistribution(int total_columns, int process_rank, int process_count) {
  int base_columns_per_process = total_columns / process_count;
  int remaining_columns = total_columns % process_count;

  int start_column = (process_rank * base_columns_per_process) + std::min(process_rank, remaining_columns);
  int end_column = start_column + base_columns_per_process + (process_rank < remaining_columns ? 1 : 0);

  return {start_column, end_column};
}

void LobanovDMultiplyMatrixMPI::ProcessLocalColumnMPI(const CompressedColumnMatrix &transposed_matrix_a, const std::vector<double> &local_values,
                                                      const std::vector<int> &local_row_indices,
                                                      const std::vector<int> &local_column_pointers, int column_index,
                                                      std::vector<double> &temporary_row_values, std::vector<int> &row_marker_array,
                                                      std::vector<double> &result_values, std::vector<int> &result_row_indices) {
  int column_start_position = local_column_pointers[column_index];
  int column_end_position = local_column_pointers[column_index + 1];

  for (int element_index = column_start_position; element_index < column_end_position; element_index++) {
    int matrix_b_row = local_row_indices[element_index];
    double matrix_b_value = local_values[element_index];

    for (int transposed_index = transposed_matrix_a.column_pointer_data[matrix_b_row]; 
         transposed_index < transposed_matrix_a.column_pointer_data[matrix_b_row + 1]; transposed_index++) {
      int matrix_a_row = transposed_matrix_a.row_index_data[transposed_index];
      double matrix_a_value = transposed_matrix_a.value_data[transposed_index];

      if (row_marker_array[matrix_a_row] != column_index) {
        row_marker_array[matrix_a_row] = column_index;
        temporary_row_values[matrix_a_row] = matrix_a_value * matrix_b_value;
      } else {
        temporary_row_values[matrix_a_row] += matrix_a_value * matrix_b_value;
      }
    }
  }

  for (int row_index = 0; row_index < transposed_matrix_a.column_count; row_index++) {
    if (row_marker_array[row_index] == column_index && std::abs(temporary_row_values[row_index]) > EPSILON_THRESHOLD) {
      result_values.push_back(temporary_row_values[row_index]);
      result_row_indices.push_back(row_index);
    }
  }
}

void LobanovDMultiplyMatrixMPI::ExtractLocalColumnData(const CompressedColumnMatrix &matrix_b, 
                                                       int start_column, int end_column,
                                                       std::vector<double> &local_values, 
                                                       std::vector<int> &local_row_indices,
                                                       std::vector<int> &local_column_pointers) {
    local_values.clear();
    local_row_indices.clear();
    local_column_pointers.clear();

    local_column_pointers.push_back(0);

    auto extract_column_data = [&](int col_index) {
        int col_start = matrix_b.column_pointer_data[col_index];
        int col_end = matrix_b.column_pointer_data[col_index + 1];
        
        for (int idx = col_start; idx < col_end; ++idx) {
            local_values.push_back(matrix_b.value_data[idx]);
            local_row_indices.push_back(matrix_b.row_index_data[idx]);
        }
        
        return static_cast<int>(local_values.size());
    };

    for (int current_col = start_column; current_col < end_column; ++current_col) {
        int new_size = extract_column_data(current_col);
        local_column_pointers.push_back(new_size);
    }
}

void LobanovDMultiplyMatrixMPI::MultiplyLocalMatricesMPI(
    const CompressedColumnMatrix &transposed_matrix_a, 
    const std::vector<double> &local_values,
    const std::vector<int> &local_row_indices,
    const std::vector<int> &local_column_pointers, 
    int local_column_count,
    std::vector<double> &result_values, 
    std::vector<int> &result_row_indices,
    std::vector<int> &result_column_pointers) {
    
    result_values.clear();
    result_row_indices.clear();
    result_column_pointers.clear();
    result_column_pointers.push_back(0);
    
    int matrix_rows = transposed_matrix_a.column_count;
    std::vector<double> temp_row_buffer(matrix_rows, 0.0);
    std::vector<int> row_tracker(matrix_rows, -1);
    
    int column_counter = 0;
    while (column_counter < local_column_count) {
        ProcessLocalColumnMPI(
            transposed_matrix_a, 
            local_values, 
            local_row_indices, 
            local_column_pointers,
            column_counter, 
            temp_row_buffer, 
            row_tracker,
            result_values, 
            result_row_indices
        );
        
        result_column_pointers.push_back(
            static_cast<int>(result_values.size())
        );
        column_counter++;
    }
}

bool LobanovDMultiplyMatrixMPI::ProcessMasterRank(
    const CompressedColumnMatrix &matrix_a, 
    const CompressedColumnMatrix &matrix_b,
    std::vector<double> &local_result_values, 
    std::vector<int> &local_result_row_indices,
    std::vector<int> &local_result_column_pointers, 
    int total_processes) {
  
  // Инициализация результата
  CompressedColumnMatrix result_matrix;
  result_matrix.row_count = matrix_a.row_count;
  result_matrix.column_count = matrix_b.column_count;
  result_matrix.column_pointer_data.push_back(0);
  
  // Структуры для сбора данных
  std::vector<std::vector<double>> value_collections(total_processes);
  std::vector<std::vector<int>> row_index_collections(total_processes);
  std::vector<std::vector<int>> column_pointer_collections(total_processes);
  
  // Лямбда для приема данных от процесса
  auto receive_process_data = [](int process_id, 
                                 std::vector<double> &values,
                                 std::vector<int> &row_indices,
                                 std::vector<int> &col_pointers) {
    int nnz = 0, cols = 0;
    MPI_Recv(&nnz, 1, MPI_INT, process_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&cols, 1, MPI_INT, process_id, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    values.resize(nnz);
    row_indices.resize(nnz);
    col_pointers.resize(cols + 1);
    
    MPI_Recv(values.data(), nnz, MPI_DOUBLE, process_id, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(row_indices.data(), nnz, MPI_INT, process_id, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(col_pointers.data(), cols + 1, MPI_INT, process_id, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  };
  
  // Сохранение локальных данных
  value_collections[0] = std::move(local_result_values);
  row_index_collections[0] = std::move(local_result_row_indices);
  column_pointer_collections[0] = std::move(local_result_column_pointers);
  
  // Получение данных от рабочих процессов
  for (int pid = 1; pid < total_processes; ++pid) {
    receive_process_data(pid, value_collections[pid], 
                        row_index_collections[pid], 
                        column_pointer_collections[pid]);
  }
  
  // Объединение данных
  int value_offset = 0;
  int column_offset = 0;
  
  for (int pid = 0; pid < total_processes; ++pid) {
    // Добавление значений и индексов строк
    result_matrix.value_data.insert(result_matrix.value_data.end(),
                                   value_collections[pid].begin(),
                                   value_collections[pid].end());
    
    result_matrix.row_index_data.insert(result_matrix.row_index_data.end(),
                                       row_index_collections[pid].begin(),
                                       row_index_collections[pid].end());
    
    // Добавление указателей столбцов с учетом смещений
    for (size_t i = 1; i < column_pointer_collections[pid].size(); ++i) {
      result_matrix.column_pointer_data.push_back(
          column_pointer_collections[pid][i] + value_offset
      );
    }
    
    // Обновление смещений
    value_offset += static_cast<int>(value_collections[pid].size());
    column_offset += static_cast<int>(column_pointer_collections[pid].size() - 1);
  }
  
  // Завершение
  result_matrix.non_zero_count = static_cast<int>(result_matrix.value_data.size());
  GetOutput() = result_matrix;
  
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LobanovDMultiplyMatrixMPI::ProcessWorkerRank(const std::vector<double> &local_result_values,
                                                  const std::vector<int> &local_result_row_indices,
                                                  const std::vector<int> &local_result_column_pointers, int local_column_count) {
  int local_non_zero_count = static_cast<int>(local_result_values.size());
  int local_column_total = local_column_count;

  MPI_Send(&local_non_zero_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(&local_column_total, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
  MPI_Send(local_result_values.data(), local_non_zero_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  MPI_Send(local_result_row_indices.data(), local_non_zero_count, MPI_INT, 0, 3, MPI_COMM_WORLD);
  MPI_Send(local_result_column_pointers.data(), local_column_count + 1, MPI_INT, 0, 4, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}


bool LobanovDMultiplyMatrixMPI::RunImpl() {
  int process_rank, total_processes;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
  const auto &[matrix_a, matrix_b] = GetInput();
  CompressedColumnMatrix transposed_matrix_a;
  if (process_rank == 0) {
    ComputeTransposedMatrixMPI(matrix_a, transposed_matrix_a);
    
    int transposed_dims[3] = {transposed_matrix_a.row_count, 
                              transposed_matrix_a.column_count, 
                              transposed_matrix_a.non_zero_count};
    MPI_Bcast(transposed_dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(transposed_matrix_a.value_data.data(), transposed_dims[2], 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(transposed_matrix_a.row_index_data.data(), transposed_dims[2], 
              MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(transposed_matrix_a.column_pointer_data.data(), transposed_dims[1] + 1, 
              MPI_INT, 0, MPI_COMM_WORLD);

    transposed_matrix_a.row_count = transposed_dims[0];
    transposed_matrix_a.column_count = transposed_dims[1];
    transposed_matrix_a.non_zero_count = transposed_dims[2];
  } else {

    int transposed_dims[3];
    MPI_Bcast(transposed_dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    transposed_matrix_a.row_count = transposed_dims[0];
    transposed_matrix_a.column_count = transposed_dims[1];
    transposed_matrix_a.non_zero_count = transposed_dims[2];
    
    transposed_matrix_a.value_data.resize(transposed_matrix_a.non_zero_count);
    transposed_matrix_a.row_index_data.resize(transposed_matrix_a.non_zero_count);
    transposed_matrix_a.column_pointer_data.resize(transposed_matrix_a.column_count + 1);
    
    MPI_Bcast(transposed_matrix_a.value_data.data(), transposed_matrix_a.non_zero_count, 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(transposed_matrix_a.row_index_data.data(), transposed_matrix_a.non_zero_count, 
              MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(transposed_matrix_a.column_pointer_data.data(), transposed_matrix_a.column_count + 1, 
              MPI_INT, 0, MPI_COMM_WORLD);
  }

  int columns_per_process[2] = {0, 0};
  
  if (process_rank == 0) {
    auto [start, end] = DetermineColumnDistribution(matrix_b.column_count, process_rank, total_processes);
    columns_per_process[0] = start;
    columns_per_process[1] = end;
    
    for (int pid = 1; pid < total_processes; ++pid) {
      auto [pid_start, pid_end] = DetermineColumnDistribution(matrix_b.column_count, pid, total_processes);
      int pid_columns[2] = {pid_start, pid_end};
      MPI_Send(pid_columns, 2, MPI_INT, pid, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(columns_per_process, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int start_column = columns_per_process[0];
  int end_column = columns_per_process[1];
  int local_column_count = end_column - start_column;

  std::vector<double> local_values;
  std::vector<int> local_row_indices;
  std::vector<int> local_column_pointers;
  
  ExtractLocalColumnData(matrix_b, start_column, end_column, 
                        local_values, local_row_indices, local_column_pointers);

  std::vector<double> result_values;
  std::vector<int> result_row_indices;
  std::vector<int> result_column_pointers;
  
  MultiplyLocalMatricesMPI(transposed_matrix_a, local_values, local_row_indices, 
                          local_column_pointers, local_column_count,
                          result_values, result_row_indices, result_column_pointers);

  if (process_rank == 0) {
    return ProcessMasterRank(matrix_a, matrix_b, result_values, result_row_indices, 
                            result_column_pointers, total_processes);
  } else {
    return ProcessWorkerRank(result_values, result_row_indices, 
                           result_column_pointers, local_column_count);
  }
}

bool LobanovDMultiplyMatrixMPI::PostProcessingImpl() {

    int current_process_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process_rank);

    const CompressedColumnMatrix& computed_result = GetOutput();

    if (current_process_rank == 0) {
        bool dimensions_valid = computed_result.row_count > 0 && 
                                computed_result.column_count > 0;
        bool structure_valid = computed_result.column_pointer_data.size() == 
                               static_cast<size_t>(computed_result.column_count) + 1;
        return dimensions_valid && structure_valid;
    } else {
        bool is_empty = computed_result.row_count == 0 && 
                        computed_result.column_count == 0;
        return is_empty;
    }
}

}  // namespace lobanov_d_multiply_matrix_ccs