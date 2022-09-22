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

namespace _P2300::execution {
  namespace stream_split {
    template <class _SharedState>
      class __receiver : example::cuda::stream::receiver_base_t {
        _SharedState &__sh_state_;

      public:
        template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
        friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept {
          _SharedState &__state = __self.__sh_state_;

          if constexpr (std::is_base_of_v<example::cuda::stream::detail::op_state_base_t, typename _SharedState::inner_op_state_t>) {
            cudaStreamSynchronize(__state.__op_state2_.stream_);
          }

          _NVCXX_EXPAND_PACK(_As, __as,
            try {
              using __tuple_t = __decayed_tuple<_Tag, _As...>;
              __state.__data_.template emplace<__tuple_t>(__tag, (_As &&) __as...);
            } catch (...) {
              using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
              __state.__data_.template emplace<__tuple_t>(set_error, std::current_exception());
            }
          )
          __state.__notify();
        }

        friend auto tag_invoke(get_env_t, const __receiver& __self)
          -> make_env_t<with_t<get_stop_token_t, in_place_stop_token>> {
          return make_env(with(get_stop_token, __self.__sh_state_.__stop_source_.get_token()));
        }

        explicit __receiver(_SharedState &__sh_state) noexcept
          : __sh_state_(__sh_state) {
        }
    };

    struct __operation_base {
      using __notify_fn = void(__operation_base*) noexcept;

      __operation_base * __next_{};
      __notify_fn* __notify_{};
    };

    template <class _SenderId>
      struct __sh_state {
        using _Sender = __t<_SenderId>;

        template <class... _Ts>
          using __bind_tuples =
            __mbind_front_q<
              __variant,
              std::tuple<set_stopped_t>, // Initial state of the variant is set_stopped
              std::tuple<set_error_t, std::exception_ptr>,
              _Ts...>;

        using __bound_values_t =
          __value_types_of_t<
            _Sender,
            make_env_t<with_t<get_stop_token_t, in_place_stop_token>>,
            __mbind_front_q<__decayed_tuple, set_value_t>,
            __q<__bind_tuples>>;

        using __variant_t =
          __error_types_of_t<
            _Sender,
            make_env_t<with_t<get_stop_token_t, in_place_stop_token>>,
            __transform<
              __mbind_front_q<__decayed_tuple, set_error_t>,
              __bound_values_t>>;

        using __receiver_ = __receiver<__sh_state>;
        using inner_op_state_t = connect_result_t<_Sender, __receiver_>;

        in_place_stop_source __stop_source_{};
        inner_op_state_t __op_state2_;
        __variant_t __data_;
        std::atomic<void*> __head_;

        explicit __sh_state(_Sender& __sndr)
          : __op_state2_(connect((_Sender&&) __sndr, __receiver_{*this}))
          , __head_{nullptr}
        {}

        void __notify() noexcept {
          void* const __completion_state = static_cast<void*>(this);
          void *__old = __head_.exchange(__completion_state, std::memory_order_acq_rel);
          __operation_base *__op_state = static_cast<__operation_base*>(__old);

          while(__op_state != nullptr) {
            __operation_base *__next = __op_state->__next_;
            __op_state->__notify_(__op_state);
            __op_state = __next;
          }
        }
      };

    // TODO Stream operation
    template <class _SenderId, class _ReceiverId>
      class __operation : public __operation_base {
        using _Sender = __t<_SenderId>;
        using _Receiver = __t<_ReceiverId>;

        struct __on_stop_requested {
          in_place_stop_source& __stop_source_;
          void operator()() noexcept {
            __stop_source_.request_stop();
          }
        };
        using __on_stop = std::optional<typename stop_token_of_t<
            env_of_t<_Receiver> &>::template callback_type<__on_stop_requested>>;

        _Receiver __recvr_;
        __on_stop __on_stop_{};
        std::shared_ptr<__sh_state<_SenderId>> __shared_state_;

      public:
        __operation(_Receiver&& __rcvr,
                    std::shared_ptr<__sh_state<_SenderId>> __shared_state)
            noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          : __operation_base{nullptr, __notify}
          , __recvr_((_Receiver&&)__rcvr)
          , __shared_state_(move(__shared_state)) {
        }
        _P2300_IMMOVABLE(__operation);

        static void __notify(__operation_base* __self) noexcept {
          __operation *__op = static_cast<__operation*>(__self);
          __op->__on_stop_.reset();

          std::visit([&](const auto& __tupl) noexcept -> void {
            std::apply([&](auto __tag, const auto&... __args) noexcept -> void {
              __tag((_Receiver&&) __op->__recvr_, __args...);
            }, __tupl);
          }, __op->__shared_state_->__data_);
        }

        friend void tag_invoke(start_t, __operation& __self) noexcept {
          __sh_state<_SenderId>* __shared_state = __self.__shared_state_.get();
          std::atomic<void*>& __head = __shared_state->__head_;
          void* const __completion_state = static_cast<void*>(__shared_state);
          void* __old = __head.load(std::memory_order_acquire);

          if (__old != __completion_state) {
            __self.__on_stop_.emplace(
                get_stop_token(get_env(__self.__recvr_)),
                __on_stop_requested{__shared_state->__stop_source_});
          }

          do {
            if (__old == __completion_state) {
              __self.__notify(&__self);
              return;
            }
            __self.__next_ = static_cast<__operation_base*>(__old);
          } while (!__head.compare_exchange_weak(
              __old, static_cast<void *>(&__self),
              std::memory_order_release,
              std::memory_order_acquire));

          if (__old == nullptr) {
            // the inner sender isn't running
            if (__shared_state->__stop_source_.stop_requested()) {
              // 1. resets __head to completion state
              // 2. notifies waiting threads
              // 3. propagates "stopped" signal to `out_r'`
              __shared_state->__notify();
            } else {
              start(__shared_state->__op_state2_);
            }
          }
        }
      };

    template <class _SenderId>
      class __sender : example::cuda::stream::sender_base_t {
        using _Sender = __t<_SenderId>;
        using __sh_state_ = __sh_state<_SenderId>;
        template <class _Receiver>
          using __operation = __operation<_SenderId, __x<remove_cvref_t<_Receiver>>>;

        _Sender __sndr_;
        std::shared_ptr<__sh_state_> __shared_state_;

      public:
        template <__decays_to<__sender> _Self, receiver _Receiver>
            requires receiver_of<_Receiver, completion_signatures_of_t<_Self, __empty_env>>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __recvr)
            noexcept(std::is_nothrow_constructible_v<decay_t<_Receiver>, _Receiver>)
            -> __operation<_Receiver> {
            return __operation<_Receiver>{(_Receiver &&) __recvr,
                                          __self.__shared_state_};
          }

        template <tag_category<forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires (!__is_instance_of<_Tag, get_completion_scheduler_t>) &&
              __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
            _NVCXX_EXPAND_PACK_RETURN(_As, __as,
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            )
          }

        template <class... _Tys>
        using __set_value_t = completion_signatures<set_value_t(const decay_t<_Tys>&...)>;

        template <class _Ty>
        using __set_error_t = completion_signatures<set_error_t(const decay_t<_Ty>&)>;

        template <__decays_to<__sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
            make_completion_signatures<
              _Sender,
              make_env_t<with_t<get_stop_token_t, in_place_stop_token>>,
              completion_signatures<set_error_t(const std::exception_ptr&)>,
              __set_value_t,
              __set_error_t>;

        explicit __sender(_Sender __sndr)
            : __sndr_((_Sender&&) __sndr)
            , __shared_state_{std::make_shared<__sh_state_>(__sndr_)}
        {}
      };

    struct split_t {
      template <class _Sender>
        using __sender = __sender<__x<remove_cvref_t<_Sender>>>;

      template <sender _Sender>
        requires __tag_invocable_with_completion_scheduler<split_t, set_value_t, _Sender>
      sender auto operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<split_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(split_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!__tag_invocable_with_completion_scheduler<split_t, set_value_t, _Sender>) &&
          tag_invocable<split_t, _Sender>
      sender auto operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<split_t, _Sender>) {
        return tag_invoke(split_t{}, (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!__tag_invocable_with_completion_scheduler<split_t, set_value_t, _Sender>) &&
          (!tag_invocable<split_t, _Sender>)
      __sender<_Sender> operator()(_Sender&& __sndr) const {
        return __sender<_Sender>{(_Sender&&) __sndr};
      }
      __binder_back<split_t> operator()() const {
        return {{}, {}, {}};
      }
    };
  } // namespace stream_split
}

