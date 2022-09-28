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

namespace example::cuda::stream::submit {

template <class SenderId, class ReceiverId>
struct op_state_t {
  using Sender = std::__t<SenderId>;
  using Receiver = std::__t<ReceiverId>;
  struct receiver_t : receiver_base_t {
    op_state_t* op_state_;

    template <std::__one_of<std::execution::set_value_t, std::execution::set_error_t, std::execution::set_stopped_t> Tag, class... As>
      requires std::__callable<Tag, Receiver, As...>
    friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as)
        noexcept(std::__nothrow_callable<Tag, Receiver, As...>) {
      // Delete the state as cleanup:
      std::unique_ptr<op_state_t> g{self.op_state_};
      return tag((Receiver&&) self.op_state_->rcvr_, (As&&) as...);
    }
    // Forward all receiever queries.
    friend auto tag_invoke(std::execution::get_env_t, const receiver_t& self)
      -> std::execution::env_of_t<Receiver> {
      return std::execution::get_env((const Receiver&) self.op_state_->rcvr_);
    }
  };
  Receiver rcvr_;
  std::execution::connect_result_t<Sender, receiver_t> op_state_;

  op_state_t(Sender&& sndr, std::__decays_to<Receiver> auto&& rcvr)
    : rcvr_((decltype(rcvr)&&) rcvr)
    , op_state_(std::execution::connect((Sender&&) sndr, receiver_t{{}, this}))
  {}
};

struct submit_t {
  template <std::execution::receiver Receiver, std::execution::sender_to<Receiver> Sender>
  void operator()(Sender&& sndr, Receiver&& rcvr) const noexcept(false) {
    std::execution::start((new op_state_t<std::__x<Sender>, std::__x<std::decay_t<Receiver>>>{
        (Sender&&) sndr, (Receiver&&) rcvr})->op_state_);
  }
};

}