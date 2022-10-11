/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2022 NVIDIA Corporation
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <exec/on.hpp>
#include <exec/static_thread_pool.hpp>

#include <chrono>

namespace ex = std::execution;

using namespace std::chrono_literals;

TEST_CASE("exec::on returns a sender", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("exec::on with environment returns a sender", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("exec::on simple example", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("exec::on calls the receiver when the scheduler dictates", "[adaptors][exec::on]") {
  int recv_value{0};
  impulse_scheduler sched;
  auto snd = exec::on(sched, ex::just(13)) | exec::complete_on(inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CHECK(recv_value == 13);
}

TEST_CASE("exec::on calls the given sender when the scheduler dictates", "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() | ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched;
  auto snd = exec::on(sched, std::move(snd_base)) | exec::complete_on(inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CHECK(called);
  CHECK(recv_value == 19);
}

TEST_CASE("exec::on works when changing threads", "[adaptors][exec::on]") {
  exec::static_thread_pool pool{2};
  bool called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = exec::on(pool.get_scheduler(), ex::just()) //
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

TEST_CASE("exec::on can be called with rvalue ref scheduler", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}
TEST_CASE("exec::on can be called with const ref scheduler", "[adaptors][exec::on]") {
  const inline_scheduler sched;
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}
TEST_CASE("exec::on can be called with ref scheduler", "[adaptors][exec::on]") {
  inline_scheduler sched;
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("exec::on forwards set_error calls", "[adaptors][exec::on]") {
  error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The receiver checks if we receive an error
}
TEST_CASE("exec::on forwards set_error calls of other types", "[adaptors][exec::on]") {
  error_scheduler<std::string> sched{std::string{"error"}};
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
  ex::start(op);
  // The receiver checks if we receive an error
}
TEST_CASE("exec::on forwards set_stopped calls", "[adaptors][exec::on]") {
  stopped_scheduler sched{};
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

TEST_CASE("exec::on has the values_type corresponding to the given values", "[adaptors][exec::on]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(exec::on(sched, ex::just(1)));
  check_val_types<type_array<type_array<int, double>>>(exec::on(sched, ex::just(3, 0.14)));
  check_val_types<type_array<type_array<int, double, std::string>>>(
      exec::on(sched, ex::just(3, 0.14, std::string{"pi"})));
}
TEST_CASE("exec::on keeps error_types from scheduler's sender", "[adaptors][exec::on]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<std::exception_ptr>>(exec::on(sched1, ex::just(1)));
  check_err_types<type_array<std::exception_ptr>>(exec::on(sched2, ex::just(2)));
  check_err_types<type_array<std::exception_ptr, int>>(exec::on(sched3, ex::just(3)));
}
TEST_CASE("exec::on keeps sends_stopped from scheduler's sender", "[adaptors][exec::on]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(exec::on(sched1, ex::just(1)));
  check_sends_stopped<true>(exec::on(sched2, ex::just(2)));
  check_sends_stopped<true>(exec::on(sched3, ex::just(3)));
}

TEST_CASE("exec::on transitions back to the receiver's scheduler when completing with a value", "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() | ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd = exec::on(sched1, std::move(snd_base)) | exec::complete_on(sched2);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the base sender is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE("exec::on transitions back to the receiver's scheduler when completing with an error", "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() | ex::let_value([&]() {
    called = true;
    return ex::just_error(19);
  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd = exec::on(sched1, std::move(snd_base)) | exec::complete_on(sched2);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the base sender is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the base sender is called, and an error is sent to the receiver
  CHECK(recv_error == 19);
}

TEST_CASE("inner on transitions back to outer on's scheduler when completing with a value", "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() | ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd =
      exec::on(sched1, exec::on(sched2, std::move(snd_base)))
    | exec::complete_on(sched3);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched2
  sched1.start_next();

  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the base sender and post work back to sched1
  sched2.start_next();

  // Now the base sender is called, and execution is transfered back
  // to sched1
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched1 to start executing one task. This will post work to
  // sched3
  sched1.start_next();

  // The final receiver still hasn't been called
  CHECK(recv_value == 0);

  // Tell sched3 to start executing one task. It should call the
  // final receiver
  sched3.start_next();

  // Now the value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE("inner on transitions back to outer on's scheduler when completing with an error", "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() | ex::let_value([&]() {
    called = true;
    return ex::just_error(19);
  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd =
      exec::on(sched1, exec::on(sched2, std::move(snd_base)))
    | exec::complete_on(sched3);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched2
  sched1.start_next();

  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the base sender and post work back to sched1
  sched2.start_next();

  // Now the base sender is called, and execution is transfered back
  // to sched1
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched1 to start executing one task. This will post work to
  // sched3
  sched1.start_next();

  // The final receiver still hasn't been called
  CHECK(recv_error == 0);

  // Tell sched3 to start executing one task. It should call the
  // final receiver
  sched3.start_next();

  // Now the error is sent to the receiver
  CHECK(recv_error == 19);
}

TEST_CASE("exec::on(closure) transitions onto and back off of the scheduler when completing with a value", "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd =
      ex::just()
    | exec::on(sched1, std::move(closure))
    | exec::complete_on(sched2);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the closure is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the closure is called, and a value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE("exec::on(closure) transitions onto and back off of the scheduler when completing with an error", "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::let_value([&]() {
    called = true;
    return ex::just_error(19);
  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd =
      ex::just()
    | exec::on(sched1, std::move(closure))
    | exec::complete_on(sched2);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the closure is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the closure is called, and a error is sent to the receiver
  CHECK(recv_error == 19);
}

TEST_CASE("inner on(closure) transitions back to outer on's scheduler when completing with a value", "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::then([&](int i) -> int {
    called = true;
    return i;
  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd =
      exec::on(sched1, ex::just(19))
    | exec::on(sched2, std::move(closure))
    | exec::complete_on(sched3);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{&recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched3
  sched1.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched3 to start executing one task. This post work to
  // sched2.
  sched3.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the closure and post work back to sched3
  sched2.start_next();

  // Now the closure is called, and execution is transfered back
  // to sched3
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched3 to start executing one task. This will call the
  // receiver
  sched3.start_next();

  // Now the value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE("inner on(closure) transitions back to outer on's scheduler when completing with an error", "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::let_value([&](int i) {
    called = true;
    return ex::just_error(i);
  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd =
      exec::on(sched1, ex::just(19))
    | exec::on(sched2, std::move(closure))
    | exec::complete_on(sched3);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched3
  sched1.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched3 to start executing one task. This post work to
  // sched2.
  sched3.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the closure and post work back to sched3
  sched2.start_next();

  // Now the closure is called, and execution is transfered back
  // to sched3
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched3 to start executing one task. This will call the
  // receiver
  sched3.start_next();

  // Now the error is sent to the receiver
  CHECK(recv_error == 19);
}
