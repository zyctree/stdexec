#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <cstdio>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  stream::scheduler_t scheduler{};

  auto snd = ex::schedule(scheduler) | 
             ex::bulk(4, [](int idx) {
               std::printf("hello from %d\n", idx);
             });

  std::this_thread::sync_wait(snd);
}
