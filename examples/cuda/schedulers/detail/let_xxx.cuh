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
  namespace stream_let {
    namespace __impl {

      template <class Fun, class ResultSenderT, class... As>
        __launch_bounds__(1) 
        __global__ void kernel_with_result(Fun fn, ResultSenderT* result, As... as) {
          new (result) ResultSenderT(fn(as...));
        }

      template <class... _Ts>
        struct __as_tuple {
          __decayed_tuple<_Ts...> operator()(_Ts...) const;
        };

      template <class... Sizes>
        struct max_in_pack {
          static constexpr std::size_t value = std::max({std::size_t{}, std::__v<Sizes>...});
        };

      template <class _SenderId, class _ReceiverId, class _FunId, class _Let>
        struct __receiver;

      template <class... _Ts>
        struct __which_tuple_ : _Ts... {
          using _Ts::operator()...;
        };

      struct __which_tuple_base {
        template <class... _Ts>
          __decayed_tuple<_Ts...> operator()(_Ts&&...) const;
      };

      template <sender, class, class>
        struct __which_tuple : __which_tuple_base {};

      template <class _Sender, class _Env>
          requires sender<_Sender, _Env>
        struct __which_tuple<_Sender, _Env, set_value_t>
          : value_types_of_t<_Sender, _Env, __as_tuple, __which_tuple_> {};

      template <class _Sender, class _Env>
          requires sender<_Sender, _Env>
        struct __which_tuple<_Sender, _Env, set_error_t>
          : __error_types_of_t<
              _Sender,
              _Env,
              __transform<__q<__as_tuple>, __q<__which_tuple_>>> {};

      template <class _Fun>
        struct __applyable_fn {
          template <class... _As>
            __ operator()(_As&&...) const;

          template <class... _As>
              requires invocable<_Fun, _As...>
            std::invoke_result_t<_Fun, _As...> operator()(_As&&...) const {
              std::terminate(); // this is never called; but we need a body
            }
        };

      template <class _Fun, class _Tuple>
        concept __applyable =
          requires (__applyable_fn<_Fun> __fun, _Tuple&& __tupl) {
            {std::apply(__fun, (_Tuple&&) __tupl)} -> __none_of<__>;
          };
      template <class _Fun, class _Tuple>
          requires __applyable<_Fun, _Tuple>
        using __apply_result_t =
          decltype(std::apply(__applyable_fn<_Fun>{}, __declval<_Tuple>()));

      template <class _T>
        using __decay_ref = decay_t<_T>&;

      template <class _Fun, class... _As>
        using __result_sender_t = __call_result_t<_Fun, __decay_ref<_As>...>;

      template <class _Sender, class _Receiver, class _Fun, class _SetTag>
          requires sender<_Sender, env_of_t<_Receiver>>
        struct __storage {
          template <class... _As>
            struct __op_state_for_ {
              using __t = 
                connect_result_t<
                  __result_sender_t<_Fun, _As...>, 
                  example::cuda::stream::propagate_receiver_t<std::__x<_Receiver>>>;
            };
          template <class... _As>
            using __op_state_for_t = __t<__op_state_for_<_As...>>;

          // Compute a variant of tuples to hold all the values of the input
          // sender:
          using __args_t =
            __gather_sigs_t<_SetTag, _Sender, env_of_t<_Receiver>, __q<__decayed_tuple>, __nullable_variant_t>;
          __args_t __args_;

          // Compute a variant of operation states:
          using __op_state3_t =
            __gather_sigs_t<_SetTag, _Sender, env_of_t<_Receiver>, __q<__op_state_for_t>, __nullable_variant_t>;
          __op_state3_t __op_state3_;
        };

      template <class _Sender, class _Receiver, class _Fun, class _SetTag>
          requires sender<_Sender, env_of_t<_Receiver>>
        struct __max_sender_size {
          template <class... _As>
            struct __sender_size_for_ {
              using __t = __index<sizeof(__result_sender_t<_Fun, _As...>)>;
            };
          template <class... _As>
            using __sender_size_for_t = __t<__sender_size_for_<_As...>>;

          static constexpr std::size_t value =
            __v<__gather_sigs_t<_SetTag, _Sender, env_of_t<_Receiver>, __q<__sender_size_for_t>, __q<max_in_pack>>>;
        };

      template <class _Env, class _Fun, class _Set, class _Sig>
        struct __tfx_signal_impl {};

      template <class _Env, class _Fun, class _Set, class _Ret, class... _Args>
          requires (!same_as<_Set, _Ret>)
        struct __tfx_signal_impl<_Env, _Fun, _Set, _Ret(_Args...)> {
          using __t = completion_signatures<_Ret(_Args...)>;
        };

      template <class _Env, class _Fun, class _Set, class... _Args>
          requires invocable<_Fun, __decay_ref<_Args>...> &&
            sender<std::invoke_result_t<_Fun, __decay_ref<_Args>...>, _Env>
        struct __tfx_signal_impl<_Env, _Fun, _Set, _Set(_Args...)> {
          using __t =
            make_completion_signatures<
              __result_sender_t<_Fun, _Args...>,
              _Env,
              // because we don't know if connect-ing the result sender will throw:
              completion_signatures<set_error_t(std::exception_ptr)>>;
        };

      template <class _SenderId, class _ReceiverId, class _FunId, class _Let>
        struct __operation;

      template <class _SenderId, class _ReceiverId, class _FunId, class _Let>
        struct __receiver : example::cuda::stream::receiver_base_t {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using _Fun = __t<_FunId>;
          using _Env = env_of_t<_Receiver>;

          template <class... _As>
            using __which_tuple_t =
              __call_result_t<__which_tuple<_Sender, _Env, _Let>, _As...>;

          template <class... _As>
            using __op_state_for_t =
              __minvoke2<__q2<connect_result_t>, __result_sender_t<_Fun, _As...>, example::cuda::stream::propagate_receiver_t<_ReceiverId>>;

          // handle the case when let_error is used with an input sender that
          // never completes with set_error(exception_ptr)
          template <__decays_to<std::exception_ptr> _Error>
              requires same_as<_Let, set_error_t> &&
                (!__v<__error_types_of_t<_Sender, _Env, __transform<__q1<decay_t>, __contains<std::exception_ptr>>>>)
            friend void tag_invoke(set_error_t, __receiver&& __self, _Error&& __err) noexcept {
              __self.__op_state_->propagate_completion_signal(set_error, (_Error&&) __err);
            }

          template <__one_of<_Let> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
              requires __applyable<_Fun, __which_tuple_t<_As...>&> &&
                sender_to<__apply_result_t<_Fun, __which_tuple_t<_As...>&>, _Receiver>
            friend void tag_invoke(_Tag, __receiver&& __self, _As&&... __as) noexcept try {
              _NVCXX_EXPAND_PACK(_As, __as,
                using __tuple_t = __which_tuple_t<_As...>;
                using __op_state_t = __mapply<__q<__op_state_for_t>, __tuple_t>;

                using result_sender_t = __result_sender_t<_Fun, std::decay_t<_As>...>;

                cudaStream_t stream = __self.__op_state_->stream_;

                auto result_sender = reinterpret_cast<result_sender_t *>(__self.__op_state_->__sender_memory_.get());
                kernel_with_result<<<1, 1, 0, stream>>>(__self.__op_state_->__fun_, result_sender, __as...);
                cudaStreamSynchronize(stream);

                auto& __args =
                  __self.__op_state_->__storage_.__args_.template emplace<__tuple_t>((_As&&) __as...);
                auto& __op = __self.__op_state_->__storage_.__op_state3_.template emplace<__op_state_t>(
                  __conv{[&] {
                    return connect(
                        *result_sender, 
                        example::cuda::stream::propagate_receiver_t<_ReceiverId>{
                          {}, 
                          static_cast<example::cuda::stream::operation_state_base_t<_ReceiverId>&>(
                              *__self.__op_state_)});
                  }}
                );
                start(__op);
              )
            } catch(...) {
              __self.__op_state_->propagate_completion_signal(set_error, std::current_exception());
            }

          template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
              requires __none_of<_Tag, _Let> && __callable<_Tag, _Receiver, _As...>
            friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept {
              // _NVCXX_EXPAND_PACK(_As, __as,
                static_assert(__nothrow_callable<_Tag, _Receiver, _As...>);
                __self.__op_state_->propagate_completion_signal(__tag, (_As&&)__as...);
              // )
            }

          friend auto tag_invoke(get_env_t, const __receiver& __self)
            -> env_of_t<_Receiver> {
            return get_env(__self.__op_state_->receiver_);
          }

          __operation<_SenderId, _ReceiverId, _FunId, _Let>* __op_state_;
        };

      inline void cuda_deleter(std::uint8_t *ptr) {
        if (ptr) {
          cudaFree(ptr);
        }
      }

      template <class _SenderId, class _ReceiverId, class _FunId, class _Let>
        using __operation_base = 
          example::cuda::stream::detail::operation_state_t<
            _SenderId, 
            std::__x<__receiver<_SenderId, _ReceiverId, _FunId, _Let>>,
            _ReceiverId>;

      template <class _SenderId, class _ReceiverId, class _FunId, class _Let>
        struct __operation : __operation_base<_SenderId, _ReceiverId, _FunId, _Let> {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using _Fun = __t<_FunId>;
          using __receiver_t = __receiver<_SenderId, _ReceiverId, _FunId, _Let>;

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            // TODO Check allocation errors
            std::uint8_t *sender_memory{};
            cudaMallocManaged(&sender_memory, std::__v<__max_sender_size<_Sender, _Receiver, _Fun, _Let>>);
            __self.__sender_memory_.reset(sender_memory);

            start(__self.inner_op_);
          }

          template <class _Receiver2>
            __operation(_Sender&& __sndr, _Receiver2&& __rcvr, _Fun __fun)
              : __operation_base<_SenderId, _ReceiverId, _FunId, _Let>(
                  (_Sender&&) __sndr, 
                  std::execution::get_completion_scheduler<std::execution::set_value_t>(__sndr).hub_,
                  (_Receiver2&&)__rcvr,
                  [this] (example::cuda::stream::operation_state_base_t<std::__x<_Receiver2>> &) -> __receiver_t {
                    return __receiver_t{{}, this};
                  })
              , __fun_((_Fun&&) __fun)
              , __sender_memory_(nullptr, cuda_deleter)
            {}
          _P2300_IMMOVABLE(__operation);

          _Fun __fun_;
          __storage<_Sender, _Receiver, _Fun, _Let> __storage_;
          std::unique_ptr<std::uint8_t[], void(*)(std::uint8_t*)> __sender_memory_;
        };

      template <class _SenderId, class _FunId, class _SetId>
        struct __sender : example::cuda::stream::sender_base_t {
          using _Sender = __t<_SenderId>;
          using _Fun = __t<_FunId>;
          using _Set = __t<_SetId>;
          template <class _Self, class _Receiver>
            using __operation_t =
              __operation<
                __x<__member_t<_Self, _Sender>>,
                __x<remove_cvref_t<_Receiver>>,
                _FunId,
                _Set>;
          template <class _Self, class _Receiver>
            using __receiver_t =
              __receiver<
                __x<__member_t<_Self, _Sender>>,
                __x<remove_cvref_t<_Receiver>>,
                _FunId,
                _Set>;

          template <class _Env, class _Sig>
            using __tfx_signal_t = __t<__tfx_signal_impl<_Env, _Fun, _Set, _Sig>>;

          template <class _Env>
            using __tfx_signal = __mbind_front_q1<__tfx_signal_t, _Env>;

          template <class _Sender, class _Env>
            using __with_error =
              __if_c<
                __sends<_Set, _Sender, _Env>,
                __with_exception_ptr,
                completion_signatures<>>;

          template <class _Sender, class _Env>
            using __completions =
              __mapply<
                __transform<
                  __tfx_signal<_Env>,
                  __mbind_front_q<__concat_completion_signatures_t, __with_error<_Sender, _Env>>>,
                completion_signatures_of_t<_Sender, _Env>>;

          template <__decays_to<__sender> _Self, receiver _Receiver>
              requires
                sender_to<__member_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation_t<_Self, _Receiver> {
              return __operation_t<_Self, _Receiver>{
                  ((_Self&&) __self).__sndr_,
                  (_Receiver&&) __rcvr,
                  ((_Self&&) __self).__fun_
              };
            }

          template <tag_category<forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
              requires __callable<_Tag, const _Sender&, _As...>
            friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
              noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
              -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
              _NVCXX_EXPAND_PACK_RETURN(_As, __as,
                return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
              )
            }

          template <__decays_to<__sender> _Self, class _Env>
            friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
              -> dependent_completion_signatures<_Env>;
          template <__decays_to<__sender> _Self, class _Env>
            friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
              -> __completions<__member_t<_Self, _Sender>, _Env> requires true;

          _Sender __sndr_;
          _Fun __fun_;
        };
    } // namespace __impl
  } // namespace stream_let
}

