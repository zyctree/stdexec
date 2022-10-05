#include <range/v3/range/concepts.hpp>
#include <range/v3/view/subrange.hpp>

#include <thrust/device_vector.h>

#include <execution.hpp>
#include <schedulers/stream.cuh>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  const int n = 2 * 1024;
  thrust::device_vector<int> input(n, 1);
  auto r = ranges::subrange(thrust::raw_pointer_cast(input.data()),
                            thrust::raw_pointer_cast(input.data()) + input.size());

  stream::context_t stream_context{};

  auto snd = ex::transfer_just(stream_context.get_scheduler(), r)
           | stream::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
}
