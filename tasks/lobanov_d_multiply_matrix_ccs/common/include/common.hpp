#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lobanov_d_multiply_matrix_ccs {


struct CompressedColumnMatrix {
  int row_count;                      
  int column_count;                   
  int non_zero_count;                 
  std::vector<double> value_data;     
  std::vector<int> row_index_data;    
  std::vector<int> column_pointer_data; 
};

using InType = std::pair<CompressedColumnMatrix, CompressedColumnMatrix>;
using OutType = CompressedColumnMatrix;
using TestType = std::tuple<std::string, CompressedColumnMatrix, CompressedColumnMatrix, CompressedColumnMatrix>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lobanov_d_multiply_matrix_ccs