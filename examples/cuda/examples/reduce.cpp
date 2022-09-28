#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <thrust/device_vector.h>

#include <cstdio>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  const int n = 2 * 1024;
  thrust::device_vector<int> input(n);
  int *d_in = thrust::raw_pointer_cast(input.data());

  stream::context_t stream_context{};

  auto snd = ex::schedule(stream_context.get_scheduler()) 
           | ex::bulk(n, [d_in](int idx) { d_in[idx] = idx; })
           | stream::reduce(d_in, n);

  auto  expect = n * (n - 1) / 2;
  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  std::cout << (expect == result ? "OK" : "FAIL") 
            << " result: " << result  
            << " expect: " << expect << std::endl;
}
