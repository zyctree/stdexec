#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <cstdio>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  using example::cuda::is_on_gpu;

  stream::context_t stream_context{};
  ex::scheduler auto sch = stream_context.get_scheduler();

  auto bf = [](int lbl) {
    return [=](int i) { 
      std::printf("B%d: i = %d\n", lbl, i); 
    };
  };

  auto tf = [](int lbl) {
    return [=] {
      std::printf("T%d\n", lbl);
    };
  };

  auto snd = ex::transfer_when_all(
               sch,
               ex::schedule(sch) | ex::bulk(4, bf(1)),
               ex::schedule(sch) | ex::then(tf(1)),
               ex::schedule(sch) | ex::bulk(4, bf(2)))
           | ex::then(tf(2));

  std::this_thread::sync_wait(std::move(snd));
}

