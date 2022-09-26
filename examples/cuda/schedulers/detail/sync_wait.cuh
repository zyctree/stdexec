/*
 * Copyright (c) NVIDIA
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
#pragma once

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"
#include "schedulers/detail/queue.cuh"

namespace _P2300::this_thread {
  namespace stream_sync_wait {
    namespace __impl {
      template <class _Sender>
        using __into_variant_result_t =
          decltype(execution::into_variant(__declval<_Sender>()));

      struct __env {
        execution::run_loop::__scheduler __sched_;

        friend auto tag_invoke(execution::get_scheduler_t, const __env& __self) noexcept
          -> execution::run_loop::__scheduler {
          return __self.__sched_;
        }

        friend auto tag_invoke(execution::get_delegatee_scheduler_t, const __env& __self) noexcept
          -> execution::run_loop::__scheduler {
          return __self.__sched_;
        }
      };

      // What should sync_wait(just_stopped()) return?
      template <class _Sender>
          requires execution::sender<_Sender, __env>
        using __sync_wait_result_t =
          execution::value_types_of_t<
            _Sender,
            __env,
            execution::__decayed_tuple,
            __single_t>;

      template <class _Sender>
        using __sync_wait_with_variant_result_t =
          __sync_wait_result_t<__into_variant_result_t<_Sender>>;

      template <class _SenderId>
        struct __state;

        struct __sink_receiver : example::cuda::stream::receiver_base_t {
          template <class... _As>
          friend void tag_invoke(execution::set_value_t, __sink_receiver&& __rcvr, _As&&... __as) noexcept {
          }
          template <class _Error>
          friend void tag_invoke(execution::set_error_t, __sink_receiver&& __rcvr, _Error __err) noexcept {
          }
          friend void tag_invoke(execution::set_stopped_t __d, __sink_receiver&& __rcvr) noexcept {
          }
          friend execution::__empty_env
          tag_invoke(execution::get_env_t, const __sink_receiver& __rcvr) noexcept {
            return {};
          }
        };

      template <class _SenderId>
        struct __receiver : example::cuda::stream::receiver_base_t {
          using _Sender = __t<_SenderId>;

          __state<_SenderId>* __state_;
          execution::run_loop* __loop_;
          example::cuda::stream::operation_state_base_t<std::__x<__sink_receiver>>& op_state_;

          template <class _Error>
          void __set_error(_Error __err) noexcept {
            if constexpr (__decays_to<_Error, std::exception_ptr>)
              __state_->__data_.template emplace<2>((_Error&&) __err);
            else if constexpr (__decays_to<_Error, std::error_code>)
              __state_->__data_.template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
            else
              __state_->__data_.template emplace<2>(std::make_exception_ptr((_Error&&) __err));
            __loop_->finish();
          }
          template <class _Sender2 = _Sender, class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires constructible_from<__sync_wait_result_t<_Sender2>, _As...>
          friend void tag_invoke(execution::set_value_t, __receiver&& __rcvr, _As&&... __as) noexcept try {
            cudaStreamSynchronize(__rcvr.op_state_.stream_);
            _NVCXX_EXPAND_PACK(_As, __as,
              __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
            )
            __rcvr.__loop_->finish();
          } catch(...) {
            __rcvr.__set_error(std::current_exception());
          }
          template <class _Error>
          friend void tag_invoke(execution::set_error_t, __receiver&& __rcvr, _Error __err) noexcept {
            cudaStreamSynchronize(__rcvr.op_state_.stream_);
            __rcvr.__set_error((_Error &&) __err);
          }
          friend void tag_invoke(execution::set_stopped_t __d, __receiver&& __rcvr) noexcept {
            cudaStreamSynchronize(__rcvr.op_state_.stream_);
            __rcvr.__state_->__data_.template emplace<3>(__d);
            __rcvr.__loop_->finish();
          }
          friend std::execution::__empty_env
          tag_invoke(execution::get_env_t, const __receiver& __rcvr) noexcept {
            return {};
          }
        };

      template <class _SenderId>
        struct __state {
          using _Tuple = __sync_wait_result_t<__t<_SenderId>>;
          std::variant<std::monostate, _Tuple, std::exception_ptr, execution::set_stopped_t> __data_{};
        };

      template <class _Sender>
        using __into_variant_result_t =
          decltype(execution::into_variant(__declval<_Sender>()));
    } // namespace __impl

    struct sync_wait_t {
      template <execution::__single_value_variant_sender<__impl::__env> _Sender>
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>) &&
          (!tag_invocable<sync_wait_t, _Sender>) &&
          execution::sender<_Sender, __impl::__env> &&
          execution::sender_to<_Sender, __impl::__receiver<__x<_Sender>>>
      auto operator()(example::cuda::stream::detail::queue::task_hub_t* hub, _Sender&& __sndr) const
        -> std::optional<__impl::__sync_wait_result_t<_Sender>> {
        using state_t = __impl::__state<__x<_Sender>>;
        state_t __state {};
        execution::run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state =
          example::cuda::stream::stream_op_state(
            hub,
            (_Sender&&) __sndr,
            __impl::__sink_receiver{},
            [&](example::cuda::stream::operation_state_base_t<std::__x<__impl::__sink_receiver>>& stream_provider) -> __impl::__receiver<__x<_Sender>> {
              return __impl::__receiver<__x<_Sender>>{{}, &__state, &__loop, stream_provider};
            });
        execution::start(__op_state);

        // Wait for the variant to be filled in.
        __loop.run();

        if (__state.__data_.index() == 2)
          rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return std::nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    };
  } // namespace stream_sync_wait
}

