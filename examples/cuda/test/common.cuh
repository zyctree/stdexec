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

#include <algorithm>
#include <execution.hpp>
#include <cstdio>

template <int N = 1>
  requires (N > 0)
class flags_storage_t {
  int *flags_{};

public:
  class flags_t {
    int *flags_{};

    flags_t(int *flags)
      : flags_(flags) {
    }

  public:
    __device__ void set(int idx = 0) const { 
      if (idx < N) {
        flags_[idx] += 1; 
      }
    }

    friend flags_storage_t;
  };

  flags_storage_t(const flags_storage_t &) = delete;
  flags_storage_t(flags_storage_t &&) = delete;

  void operator()(const flags_storage_t &) = delete;
  void operator()(flags_storage_t &&) = delete;

  flags_t get() {
    return {flags_}; 
  }

  flags_storage_t() {
    cudaMallocHost(&flags_, sizeof(int) * N);
    memset(flags_, 0, sizeof(int) * N);
  }

  ~flags_storage_t() {
    cudaFreeHost(flags_);
    flags_ = nullptr;
  }

  bool is_set_n_times(int n) {
    return std::count(flags_, flags_ + N, n) == N;
  }

  bool all_set_once() {
    return is_set_n_times(1);
  }

  bool all_unset() { 
    return !all_set_once(); 
  }
};

namespace detail {
  template <class SenderId, class ReceiverId>
    struct operation_state_t {
      using Sender = std::__t<SenderId>;
      using Receiver = std::__t<ReceiverId>;
      using inner_op_state_t = std::execution::connect_result_t<Sender, Receiver>;

      inner_op_state_t inner_op_;

      friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
        std::execution::start(op.inner_op_);
      }

      operation_state_t(Sender&& sender, Receiver&& receiver)
        : inner_op_{std::execution::connect((Sender&&)sender, (Receiver&&)receiver)}
      {}
    };

  template <class ReceiverId, class Fun>
    class receiver_t : std::execution::receiver_adaptor<receiver_t<ReceiverId, Fun>, std::__t<ReceiverId>> {
      using Receiver = std::__t<ReceiverId>;
      friend std::execution::receiver_adaptor<receiver_t, Receiver>;

      Fun f_;

      template <class... As>
      void set_value(As&&... as) && noexcept 
        requires std::__callable<Fun, As&&...> {
        using result_t = std::invoke_result_t<Fun, As&&...>;

        if constexpr (std::is_same_v<void, result_t>) {
          f_((As&&)as...);
          std::execution::set_value(std::move(this->base()));
        } else {
          std::execution::set_value(std::move(this->base()), f_((As&&)as...));
        }
      }

     public:
      explicit receiver_t(Receiver rcvr, Fun fun)
        : std::execution::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
        , f_((Fun&&) fun)
      {}
    };

  template <class SenderId, class FunId>
    struct sender_t {
      using Sender = std::__t<SenderId>;
      using Fun = std::__t<FunId>;

      Sender sndr_;
      Fun fun_;

      template <class Receiver>
        using receiver_th = receiver_t<std::__x<Receiver>, Fun>;

      template <class Self, class Receiver>
        using op_t = operation_state_t<
          std::__x<std::__member_t<Self, Sender>>, 
          std::__x<receiver_th<Receiver>>>;

      template <class Self, class Env>
        using completion_signatures =
          std::execution::__make_completion_signatures<
            std::__member_t<Self, Sender>,
            Env,
            std::execution::__with_error_invoke_t<
              std::execution::set_value_t, 
              Fun, 
              std::__member_t<Self, Sender>, 
              Env>,
            std::__mbind_front_q<std::execution::__set_value_invoke_t, Fun>>;

      template <std::__decays_to<sender_t> Self, std::execution::receiver Receiver>
        requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
      friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
        -> op_t<Self, Receiver> {
        return op_t<Self, Receiver>(((Self&&)self).sndr_, receiver_th<Receiver>((Receiver&&)rcvr, self.fun_));
      }

      template <std::__decays_to<sender_t> Self, class Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
        -> std::execution::dependent_completion_signatures<Env>;

      template <std::__decays_to<sender_t> Self, class Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env> requires true;

      template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
        requires std::__callable<Tag, const Sender&, As...>
      friend auto tag_invoke(Tag tag, const sender_t& self, As&&... as)
        noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
        -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
        return ((Tag&&) tag)(self.sndr_, (As&&) as...);
      }
    };
}

struct a_sender_t {
  template <class _Sender, class _Fun>
    using sender_th = detail::sender_t<
      std::__x<std::remove_cvref_t<_Sender>>, 
      std::__x<std::remove_cvref_t<_Fun>>>;

  template <std::execution::sender _Sender, class _Fun>
    requires std::execution::sender<sender_th<_Sender, _Fun>>
  sender_th<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
    return sender_th<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
  }

  template <class _Fun>
  std::execution::__binder_back<a_sender_t, _Fun> operator()(_Fun __fun) const {
    return {{}, {}, {(_Fun&&) __fun}};
  }
};

constexpr a_sender_t a_sender;
