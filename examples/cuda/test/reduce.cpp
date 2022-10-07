#include <catch2/catch.hpp>

#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("reduce returns a sender", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  auto snd = ex::transfer_just(
               ctx.get_scheduler(),
               ranges::views::repeat_n(1, 2048))
           | stream::reduce();

  STATIC_REQUIRE(ex::sender_of<decltype(snd), ex::set_value_t(int&)>);

  (void)snd;
}

TEST_CASE("reduce binds the range and the function", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  constexpr int N = 2048;

  auto snd = ex::schedule(ctx.get_scheduler())
           | stream::reduce(
               ranges::views::iota(0, N),
               [] (int l, int r) {
                 return std::max(l, r);
               });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 2047);
}

TEST_CASE("reduce binds the range and uses the default function", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  constexpr int N = 2048;

  auto snd = ex::schedule(ctx.get_scheduler())
           | stream::reduce(ranges::views::iota(0, N));

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == (N * (N - 1)) / 2);
}

TEST_CASE("reduce binds the range and takes the function from the predecessor", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  constexpr int N = 2048;

  auto snd = ex::schedule(ctx.get_scheduler())
           | ex::then([] {
               return [] (int l, int r) {
                 return std::max(l, r);
               };
             })
           | stream::reduce(ranges::views::iota(0, N));

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 2047);
}

TEST_CASE("reduce takes the range from the predecessor and binds the function", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  constexpr int N = 2048;

  auto snd = ex::transfer_just(
               ctx.get_scheduler(),
               ranges::views::iota(0, N))
           | stream::reduce(
               [] (int l, int r) {
                 return std::max(l, r);
               });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 2047);
}

TEST_CASE("reduce takes the range from the predecessor and uses the default function", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  constexpr int N = 2048;

  auto snd = ex::transfer_just(
               ctx.get_scheduler(),
               ranges::views::iota(0, N))
           | stream::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == (N * (N - 1)) / 2);
}

TEST_CASE("reduce takes the range and function from the predecessor", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  constexpr int N = 2048;

  auto snd = ex::transfer_just(
               ctx.get_scheduler(),
               ranges::views::iota(0, N),
               [] (int l, int r) {
                 return std::max(l, r);
               })
           | stream::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == (N * (N - 1)) / 2);
}

TEST_CASE("reduce accepts std::vector", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  std::vector<int> input(1, 2048);

  auto snd = ex::transfer_just(
               ctx.get_scheduler(),
               input)
           | stream::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 2047);
}

TEST_CASE("reduce executes on GPU", "[cuda][stream][adaptors][reduce]") {
  stream::context_t ctx{};

  auto snd = ex::transfer_just(
               ctx.get_scheduler(),
               ranges::views::repeat_n(1, 2048))
           | stream::reduce(
               [] (int l, int r) {
                 return is_on_gpu() ? l + r : 0;
               });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 2048);
}

