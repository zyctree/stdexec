#include <execution.hpp>
#include <schedulers/stream.cuh>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  std::vector<int> input(2048, 1);

  stream::context_t stream_context{};

  auto snd = ex::transfer_just(stream_context.get_scheduler(), input)
           | stream::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
}
