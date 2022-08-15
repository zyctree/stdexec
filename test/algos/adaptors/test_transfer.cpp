/*
 * Copyright (c) Lucian Radu Teodorescu
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#if defined(__GNUC__) && !defined(__clang__)
#else

#include <catch2/catch.hpp>
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <examples/schedulers/static_thread_pool.hpp>

#include <chrono>

namespace ex = std::execution;

using namespace std::chrono_literals;

namespace {
  namespace custom {
    struct domain {};
  } // namespace custom
} // anonymous namespace
using custom_inline_scheduler = basic_inline_scheduler<custom::domain>;
using custom_impulse_scheduler = basic_impulse_scheduler<custom::domain>;

TEST_CASE("unscoped_transfer returns a sender", "[adaptors][unscoped_transfer]") {
  auto snd = ex::unscoped_transfer(ex::just(13), inline_scheduler{});
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("unscoped_transfer with environment returns a sender", "[adaptors][unscoped_transfer]") {
  auto snd = ex::unscoped_transfer(ex::just(13), inline_scheduler{});
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("unscoped_transfer simple example", "[adaptors][unscoped_transfer]") {
  auto snd = ex::unscoped_transfer(ex::just(13), inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("unscoped_transfer can be piped", "[adaptors][unscoped_transfer]") {
  // Just unscoped_transfer a value to the impulse scheduler
  ex::scheduler auto sched = impulse_scheduler{};
  ex::sender auto snd = ex::unscoped_transfer(ex::just(13), sched);
  // Start the operation
  int res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex<int>(&res));
  ex::start(op);

  // The value will be available when the scheduler will execute the next operation
  REQUIRE(res == 0);
  sched.start_next();
  REQUIRE(res == 13);
}

TEST_CASE("unscoped_transfer calls the receiver when the scheduler dictates", "[adaptors][unscoped_transfer]") {
  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::unscoped_transfer(ex::just(13), sched);
  auto op = ex::connect(snd, expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CHECK(recv_value == 13);
}

TEST_CASE("unscoped_transfer calls the given sender when the scheduler dictates", "[adaptors][unscoped_transfer]") {
  bool called{false};
  auto snd_base = ex::just() | ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::unscoped_transfer(std::move(snd_base), sched);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // The sender is started, even if the scheduler hasn't yet triggered
  CHECK(called);
  // ... but didn't send the value to the receiver yet
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CHECK(called);
  CHECK(recv_value == 19);
}

TEST_CASE("unscoped_transfer works when changing threads", "[adaptors][unscoped_transfer]") {
  example::static_thread_pool pool{2};
  bool called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = ex::unscoped_transfer(ex::just(), pool.get_scheduler()) //
                          | ex::then([&] { called = true; });
    ex::start_detached(std::move(snd));
  }
  // wait for the work to be executed, with timeout
  // perform a poor-man's sync
  // NOTE: it's a shame that the `join` method in static_thread_pool is not public
  for (int i = 0; i < 1000 && !called; i++)
    std::this_thread::sleep_for(1ms);
  // the work should be executed
  REQUIRE(called);
}

TEST_CASE("unscoped_transfer can be called with rvalue ref scheduler", "[adaptors][unscoped_transfer]") {
  auto snd = ex::unscoped_transfer(ex::just(13), inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}
TEST_CASE("unscoped_transfer can be called with const ref scheduler", "[adaptors][unscoped_transfer]") {
  const inline_scheduler sched;
  auto snd = ex::unscoped_transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}
TEST_CASE("unscoped_transfer can be called with ref scheduler", "[adaptors][unscoped_transfer]") {
  inline_scheduler sched;
  auto snd = ex::unscoped_transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("unscoped_transfer forwards set_error calls", "[adaptors][unscoped_transfer]") {
  error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
  auto snd = ex::unscoped_transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The receiver checks if we receive an error
}
TEST_CASE("unscoped_transfer forwards set_error calls of other types", "[adaptors][unscoped_transfer]") {
  error_scheduler<std::string> sched{std::string{"error"}};
  auto snd = ex::unscoped_transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The receiver checks if we receive an error
}
TEST_CASE("unscoped_transfer forwards set_stopped calls", "[adaptors][unscoped_transfer]") {
  stopped_scheduler sched{};
  auto snd = ex::unscoped_transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

TEST_CASE(
    "unscoped_transfer has the values_type corresponding to the given values", "[adaptors][unscoped_transfer]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(ex::unscoped_transfer(ex::just(1), sched));
  check_val_types<type_array<type_array<int, double>>>(ex::unscoped_transfer(ex::just(3, 0.14), sched));
  check_val_types<type_array<type_array<int, double, std::string>>>(
      ex::unscoped_transfer(ex::just(3, 0.14, std::string{"pi"}), sched));
}
TEST_CASE("unscoped_transfer keeps error_types from scheduler's sender", "[adaptors][unscoped_transfer]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<std::exception_ptr>>(ex::unscoped_transfer(ex::just(1), sched1));
  check_err_types<type_array<std::exception_ptr>>(ex::unscoped_transfer(ex::just(2), sched2));
  check_err_types<type_array<std::exception_ptr, int>>(ex::unscoped_transfer(ex::just(3), sched3));
}
TEST_CASE("unscoped_transfer keeps sends_stopped from scheduler's sender", "[adaptors][unscoped_transfer]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(ex::unscoped_transfer(ex::just(1), sched1));
  check_sends_stopped<true>(ex::unscoped_transfer(ex::just(2), sched2));
  check_sends_stopped<true>(ex::unscoped_transfer(ex::just(3), sched3));
}

struct val_type1 {
  int val_;
};
struct val_type2 {
  int val_;
};
struct val_type3 {
  int val_;
};

namespace {
  namespace custom {
    // Customization of unscoped_transfer
    // Return a different sender when we invoke
    // unscoped_transfer() in the custom domain
    ex::sender_of<ex::no_env, val_type1> auto tag_invoke(
        ex::connect_transform_t,
        custom::domain,
        ex::unscoped_transfer_t,
        ex::sender_of<ex::no_env, val_type1> auto&& unscoped_transfer,
        auto&& env) {
      auto &&[snd, sched] = unscoped_transfer;
      return ex::unscoped_schedule_from(sched, ex::just(val_type1{53}));
    }

    // Customization of unscoped_schedule_from
    // Return a different sender when we invoke unscoped_schedule_from()
    // in the custom domain.
    ex::sender_of<ex::no_env, val_type2> auto tag_invoke(
        ex::connect_transform_t,
        custom::domain,
        ex::unscoped_schedule_from_t,
        ex::sender_of<ex::no_env, val_type2> auto&& sched_from,
        auto&& env) {
      auto &&[sched, snd] = sched_from;
      return ex::unscoped_schedule_from(sched, ex::just(val_type2{59}));
    }

    // Customization of unscoped_transfer with scheduler
    // Return a different sender when we invoke unscoped_transfer()
    // in the custom domain when transfering out of a
    // particular kind of execution context.
    template <class _Env>
      requires std::invocable<ex::get_scheduler_t, _Env> &&
        std::same_as<std::invoke_result_t<ex::get_scheduler_t, _Env>, custom_impulse_scheduler>
    ex::sender_of<ex::no_env, val_type3> auto tag_invoke(
        ex::connect_transform_t,
        custom::domain,
        ex::unscoped_transfer_t,
        ex::sender_of<ex::no_env, val_type3> auto&& unscoped_transfer,
        _Env&& env) {
      auto &&[snd, sched] = unscoped_transfer;      
      return ex::unscoped_transfer(ex::just(val_type3{61}), sched);
    }
  } // namespace custom
} // anonymous namespace

TEST_CASE("unscoped_transfer can be customized", "[adaptors][unscoped_transfer]") {
  // The customization will return a different value
  auto snd =
      ex::unscoped_transfer(ex::just(val_type1{1}), inline_scheduler{})
    | ex::complete_on(custom_inline_scheduler{});
  val_type1 res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex<val_type1>(&res));
  ex::start(op);
  REQUIRE(res.val_ == 53);
}

TEST_CASE("unscoped_transfer follows unscoped_schedule_from customization", "[adaptors][unscoped_transfer]") {
  // The unscoped_schedule_from customization will return a different value
  auto snd =
      ex::unscoped_transfer(ex::just(val_type2{2}), inline_scheduler{})
    | ex::complete_on(custom_inline_scheduler{});
  val_type2 res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex<val_type2>(&res));
  ex::start(op);
  REQUIRE(res.val_ == 59);
}

TEST_CASE("unscoped_transfer can be customized with two schedulers", "[adaptors][unscoped_transfer]") {
  // The customization will return a different value
  auto snd =
      ex::unscoped_transfer(
        ex::unscoped_transfer(
          ex::just(val_type3{1}),
          custom_impulse_scheduler{}),
        inline_scheduler{})
    | ex::complete_on(custom_impulse_scheduler{});
  val_type3 res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex<val_type3>(&res));
  ex::start(op);
  // we are not using custom_impulse_scheduler anymore, so the value should be available
  REQUIRE(res.val_ == 61);
}

#endif
