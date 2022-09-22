#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <cstdio>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

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

      using set_error_t = 
        std::execution::completion_signatures<
          std::execution::set_error_t(std::exception_ptr)>;

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

int main() {
  using example::cuda::is_on_gpu;

  stream::context_t stream_context{};

  auto snd = ex::schedule(stream_context.get_scheduler()) 
           | ex::bulk(4, [](int idx) { std::printf("hello from %d\n", idx); }) 
           | a_sender([]{ std::printf("a sender on %s\n", is_on_gpu() ? "GPU" : "CPU"); });

  std::this_thread::sync_wait(snd);
}
