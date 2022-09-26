#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <cstdio>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  using example::cuda::is_on_gpu;

  stream::context_t stream_context{};

  auto snd = ex::schedule(stream_context.get_scheduler()) 
           | ex::bulk(4, [](int idx) { std::printf("hello from %d\n", idx); });

  std::this_thread::sync_wait(snd);
}
