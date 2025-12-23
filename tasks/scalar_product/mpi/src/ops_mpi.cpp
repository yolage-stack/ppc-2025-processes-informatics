#include "scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "scalar_product/common/include/common.hpp"

namespace scalar_product {

ScalarProductMPI::ScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ScalarProductMPI::ValidationImpl() {
  const auto &vector_a = GetInput().first;
  const auto &vector_b = GetInput().second;
  return vector_a.size() == vector_b.size() && !vector_a.empty();
}

bool ScalarProductMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &vector_a = GetInput().first;
  const auto &vector_b = GetInput().second;

  int global_size = 0;
  if (rank == 0) {
    global_size = static_cast<int>(vector_a.size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(world_size);
  std::vector<int> displase(world_size, 0);

  const int base = global_size / world_size;
  int remain = global_size % world_size;

  for (int i = 0; i < world_size; ++i) {
    if (remain > 0) {
      counts[i] = base + 1;
      --remain;
    } else {
      counts[i] = base;
    }
    if (remain > 0) {
      --remain;
    }
  }

  for (int i = 1; i < world_size; ++i) {
    displase[i] = displase[i - 1] + counts[i - 1];
  }
  const int local_count = counts[rank];

  local_vector_a.assign(static_cast<std::size_t>(local_count), 0);
  local_vector_b.assign(static_cast<std::size_t>(local_count), 0);

  MPI_Scatterv(rank == 0 ? vector_a.data() : nullptr, counts.data(), displase.data(), MPI_INT, local_vector_a.data(),
               local_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(rank == 0 ? vector_b.data() : nullptr, counts.data(), displase.data(), MPI_INT, local_vector_b.data(),
               local_count, MPI_INT, 0, MPI_COMM_WORLD);

  local_sum = 0;
  result = 0;
  return true;
}

bool ScalarProductMPI::RunImpl() {
  local_sum = std::inner_product(local_vector_a.begin(), local_vector_a.end(), local_vector_b.begin(), 0);
  return MPI_Reduce(&local_sum, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
}

bool ScalarProductMPI::PostProcessingImpl() {
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;
  return true;
}

}  // namespace scalar_product
