/*
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
#pragma once

#include <stdexec/execution.hpp>
#include <exec/env.hpp>

namespace exec {

  /////////////////////////////////////////////////////////////////////////////
  // A scoped version of [execution.senders.adaptors.on]
  namespace __on {
    using namespace stdexec;

    enum class on_kind { start_on, continue_on };

    template <on_kind>
      struct on_t;

    template <class _SchedulerId, class _SenderId>
      struct __start_fn {
        using _Scheduler = __t<_SchedulerId>;
        using _Sender = __t<_SenderId>;
        _Scheduler __sched_;
        _Sender __sndr_;

        template <class _Self, class _OldScheduler>
          static auto __call(_Self&& __self, _OldScheduler __old_sched) {
            return std::move(((_Self&&) __self).__sndr_)
              | transfer(__old_sched)
              | exec::write(exec::with(get_scheduler, __self.__sched_));
          }

        auto operator()(auto __old_sched) && {
          return __call(std::move(*this), __old_sched);
        }
        auto operator()(auto __old_sched) const & {
          return __call(*this, __old_sched);
        }
      };
    template <class _Scheduler, class _Sender>
      __start_fn(_Scheduler, _Sender)
        -> __start_fn<__x<_Scheduler>, __x<_Sender>>;

    template <class _SchedulerId, class _SenderId>
      struct __start_on_sender {
        using _Scheduler = __t<_SchedulerId>;
        using _Sender = __t<_SenderId>;

        _Scheduler __sched_;
        _Sender __sndr_;

        template <class _Self>
          static auto __call(_Self&& __self) {
            return let_value(
              exec::read_with_default(get_scheduler, __self.__sched_)
                | transfer(__self.__sched_),
              __start_fn{__self.__sched_, ((_Self&&) __self).__sndr_});
          }

        template <class _Self>
          using __inner_t = decltype(__call(__declval<_Self>()));

        template <__decays_to<__start_on_sender> _Self, receiver _Receiver>
            requires constructible_from<_Sender, __member_t<_Self, _Sender>> &&
              sender_to<__inner_t<_Self>, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __receiver)
            -> connect_result_t<__inner_t<_Self>, _Receiver> {
            return connect(__call((_Self&&) __self), (_Receiver&&) __receiver);
          }

        template <__decays_to<__start_on_sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
            -> completion_signatures_of_t<__inner_t<_Self>, _Env>;

        // forward sender queries:
        template <tag_category<forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __start_on_sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
            _NVCXX_EXPAND_PACK_RETURN(_As, __as,
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            )
          }
      };
    template <class _Scheduler, class _Sender>
      __start_on_sender(_Scheduler, _Sender)
        -> __start_on_sender<__x<_Scheduler>, __x<_Sender>>;

    template <>
      struct on_t<on_kind::start_on> {
        template <scheduler _Scheduler, sender _Sender>
            requires constructible_from<decay_t<_Sender>, _Sender>
          auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const {
            return __start_on_sender{(_Scheduler&&) __sched, (_Sender&&) __sndr};
          }
      };

    template <class _SenderId, class _SchedulerId, class _ClosureId>
      struct __continue_fn {
        using _Sender = __t<_SenderId>;
        using _Scheduler = __t<_SchedulerId>;
        using _Closure = __t<_ClosureId>;
        _Sender __sndr_;
        _Scheduler __sched_;
        _Closure __closure_;

        template <class _Self, class _OldScheduler>
          static auto __call(_Self&& __self, _OldScheduler __old_sched) {
            return ((_Self&&) __self).__sndr_
              | transfer(__self.__sched_)
              | exec::write(exec::with(get_scheduler, __old_sched))
              | ((_Self&&) __self).__closure_
              | transfer(__old_sched)
              | exec::write(exec::with(get_scheduler, __self.__sched_));
          }

        auto operator()(auto __old_sched) && {
          return __call(std::move(*this), __old_sched);
        }
        auto operator()(auto __old_sched) const & {
          return __call(*this, __old_sched);
        }
      };
    template <class _Sender, class _Scheduler, class _Closure>
      __continue_fn(_Sender, _Scheduler, _Closure)
        -> __continue_fn<__x<_Sender>, __x<_Scheduler>, __x<_Closure>>;

    template <class _SenderId, class _SchedulerId, class _ClosureId>
      struct __continue_on_sender {
        using _Sender = __t<_SenderId>;
        using _Scheduler = __t<_SchedulerId>;
        using _Closure = __t<_ClosureId>;

        _Sender __sndr_;
        _Scheduler __sched_;
        _Closure __closure_;

        template <class _Self>
          static auto __call(_Self&& __self) {
            return let_value(
              exec::read_with_default(get_scheduler, __self.__sched_),
              __continue_fn{
                ((_Self&&) __self).__sndr_,
                __self.__sched_,
                ((_Self&&) __self).__closure_});
          }

        template <class _Self>
          using __inner_t = decltype(__call(__declval<_Self>()));

        template <__decays_to<__continue_on_sender> _Self, receiver _Receiver>
            requires constructible_from<_Sender, __member_t<_Self, _Sender>> &&
              sender_to<__inner_t<_Self>, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __receiver)
            -> connect_result_t<__inner_t<_Self>, _Receiver> {
            return connect(__call((_Self&&) __self), (_Receiver&&) __receiver);
          }

        template <__decays_to<__continue_on_sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
            -> completion_signatures_of_t<__inner_t<_Self>, _Env>;

        // forward sender queries:
        template <tag_category<forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __continue_on_sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
            _NVCXX_EXPAND_PACK_RETURN(_As, __as,
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            )
          }
      };
    template <class _Sender, class _Scheduler, class _Closure>
      __continue_on_sender(_Sender, _Scheduler, _Closure)
        -> __continue_on_sender<__x<_Sender>, __x<_Scheduler>, __x<_Closure>>;

    template <>
      struct on_t<on_kind::continue_on> {
        template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
            requires constructible_from<decay_t<_Sender>, _Sender>
          auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure __closure) const {
            return __continue_on_sender{
              (_Sender&&) __sndr,
              (_Scheduler&&) __sched,
              (_Closure&&) __closure};
          }

        template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
          auto operator()(_Scheduler&& __sched, _Closure __closure) const
            -> __binder_back<on_t, decay_t<_Scheduler>, _Closure> {
            return {{}, {}, {(_Scheduler&&) __sched, (_Closure&&) __closure}};
          }
      };

    struct __on_t
      : on_t<on_kind::start_on>
      , on_t<on_kind::continue_on> {
      using on_t<on_kind::start_on>::operator();
      using on_t<on_kind::continue_on>::operator();
    };
  } // namespace __on

  using __on::on_kind;
  using __on::on_t;
  inline constexpr __on::__on_t on{};

  // namespace __complete_on
  namespace __complete_on {
    using namespace stdexec;

    struct __complete_on_t;

    template <class _SchedulerId, class _ReceiverId>
      struct __operation_base {
        using _Scheduler = __t<_SchedulerId>;
        using _Receiver = __t<_ReceiverId>;
        _Scheduler __sched_;
        _Receiver __rcvr_;
      };

    template <class _SchedulerId, class _ReceiverId>
      struct __receiver
        : receiver_adaptor<__receiver<_SchedulerId, _ReceiverId>> {
        using _Scheduler = stdexec::__t<_SchedulerId>;
        using _Receiver = stdexec::__t<_ReceiverId>;

        _Receiver&& base() && noexcept {
          return (_Receiver&&) __op_->__rcvr_;
        }
        const _Receiver& base() const & noexcept {
          return __op_->__rcvr_;
        }

        auto get_env() const {
          return exec::make_env(
            stdexec::get_env(base()),
            exec::with(get_scheduler, __op_->__sched_));
        }

        __operation_base<_SchedulerId, _ReceiverId>* __op_;
      };

    template <class _SenderId, class _SchedulerId, class _ReceiverId>
      struct __operation : __operation_base<_SchedulerId, _ReceiverId> {
        using _Sender = __t<_SenderId>;
        using __base_t = __operation_base<_SchedulerId, _ReceiverId>;
        using __receiver_t = __receiver<_SchedulerId, _ReceiverId>;
        connect_result_t<_Sender, __receiver_t> __state_;

        __operation(_Sender&& __sndr, auto&& __sched, auto&& __rcvr)
          : __base_t{(decltype(__sched)) __sched, (decltype(__rcvr)) __rcvr}
          , __state_{connect((_Sender&&) __sndr, __receiver_t{{}, this})}
        {}

        friend void tag_invoke(start_t, __operation& __self) noexcept {
          start(__self.__state_);
        }
      };

    template <class _SenderId, class _SchedulerId>
      struct __sender {
        using _Sender = __t<_SenderId>;
        using _Scheduler = __t<_SchedulerId>;
        template <class _ReceiverId>
          using __receiver_t =
            __receiver<_SchedulerId, _ReceiverId>;
        template <class _Self, class _ReceiverId>
          using __operation_t =
            __operation<__x<__member_t<_Self, _Sender>>, _SchedulerId, _ReceiverId>;

        _Sender __sndr_;
        _Scheduler __sched_;

        template <__decays_to<__sender> _Self, receiver _Receiver>
          requires
            sender_to<__member_t<_Self, _Sender>, __receiver_t<__x<decay_t<_Receiver>>>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Self, __x<decay_t<_Receiver>>> {
          return {((_Self&&) __self).__sndr_,
                  ((_Self&&) __self).__sched_,
                  (_Receiver&&) __rcvr};
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
            requires (!__callable<get_scheduler_t, _Env>)
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> make_completion_signatures<
                _Sender,
                exec::make_env_t<_Env, with_t<get_scheduler_t, _Scheduler>>>;
      };

    struct __complete_on_t {
      template <sender _Sender, scheduler _Scheduler>
        auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const
          -> __sender<__x<decay_t<_Sender>>, __x<decay_t<_Scheduler>>> {
          return {(_Sender&&) __sndr, (_Scheduler&&) __sched};
        }

      template <scheduler _Scheduler>
        auto operator()(_Scheduler&& __sched) const
          -> __binder_back<__complete_on_t, decay_t<_Scheduler>> {
          return {{}, {}, {(_Scheduler&&) __sched}};
        }
    };
  } // namespace __complete_on
  inline constexpr __complete_on::__complete_on_t complete_on {};

} // namespace exec
