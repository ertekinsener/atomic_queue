#pragma once

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "defs.h"

namespace atomic_queue {
namespace details {

template <typename T>
constexpr T nil() noexcept {
  static_assert(std::atomic<T>::is_always_lock_free,
                "Queue element type T is not atomic. Use "
                "AtomicQueue2/AtomicQueueB2 for such element types.");
  return {};
}

template <typename T>
ATOMIC_QUEUE_INLINE void destroy_n(T* ATOMIC_QUEUE_RESTRICT p,
                                   unsigned n) noexcept {
  for (auto q = p + n; p != q;) (p++)->~T();
}

template <typename T>
ATOMIC_QUEUE_INLINE void swap_relaxed(std::atomic<T>& a,
                                      std::atomic<T>& b) noexcept {
  auto a2 = a.load(X);
  a.store(b.load(X), X);
  b.store(a2, X);
}

template <typename T>
ATOMIC_QUEUE_INLINE void copy_relaxed(std::atomic<T>& a,
                                      std::atomic<T> const& b) noexcept {
  a.store(b.load(X), X);
}

}  // namespace details

enum class State : uint8_t { EMPTY = 0, STORED = 1, STORING = 2, LOADING = 4 };
using AtomicState = std::atomic<State>;

template <typename Derived>
class AtomicQueueCommon {
 protected:
  alignas(CACHE_LINE_SIZE) std::atomic<unsigned> head_ = {};
  alignas(CACHE_LINE_SIZE) std::atomic<unsigned> tail_ = {};

  AtomicQueueCommon() noexcept {
    assert(is_suitably_aligned(static_cast<Derived*>(this)));
  }

  AtomicQueueCommon(AtomicQueueCommon const& b) noexcept
      : head_(b.head_.load(X)), tail_(b.tail_.load(X)) {
    assert(is_suitably_aligned(static_cast<Derived const*>(this)));
  }

  AtomicQueueCommon& operator=(AtomicQueueCommon const& b) noexcept {
    details::copy_relaxed(head_, b.head_);
    details::copy_relaxed(tail_, b.tail_);
    return *this;
  }

  void swap(AtomicQueueCommon& b) noexcept {
    details::swap_relaxed(head_, b.head_);
    details::swap_relaxed(tail_, b.tail_);
  }

  template <typename T>
  ATOMIC_QUEUE_INLINE T do_pop(std::atomic<T>* ATOMIC_QUEUE_RESTRICT elements,
                               unsigned index) noexcept {
    constexpr T NIL = Derived::nil_;
    T element;
    auto& q_element = elements[index];

    if constexpr (Derived::spsc_) {
      for (;;) {
        element = q_element.load(A);
        if (ATOMIC_QUEUE_LIKELY(element != NIL)) break;
        if constexpr (Derived::maximize_throughput_) spin_loop_pause();
      }
      q_element.store(NIL, R);
    } else {
      for (;;) {
        element = q_element.exchange(NIL, AR);
        if (ATOMIC_QUEUE_LIKELY(element != NIL)) break;
        do {
          spin_loop_pause();
        } while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ &&
                                       q_element.load(X) == NIL));
      }
    }
    return element;
  }

  template <typename T>
  ATOMIC_QUEUE_INLINE void do_push(
      T element, std::atomic<T>* ATOMIC_QUEUE_RESTRICT elements,
      unsigned index) noexcept {
    constexpr T NIL = Derived::nil_;
    assert(element != NIL);
    auto& q_element = elements[index];

    if constexpr (Derived::spsc_) {
      while (ATOMIC_QUEUE_UNLIKELY(q_element.load(A) != NIL)) {
        if constexpr (Derived::maximize_throughput_) spin_loop_pause();
      }
      q_element.store(element, R);
    } else {
      T expected;
      while (ATOMIC_QUEUE_UNLIKELY(
          !q_element.compare_exchange_weak((expected = NIL), element, AR, X))) {
        do {
          spin_loop_pause();
        } while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ &&
                                       q_element.load(X) != NIL));
      }
    }
  }

  template <typename T>
  ATOMIC_QUEUE_INLINE T do_pop(std::atomic<State>* ATOMIC_QUEUE_RESTRICT states,
                               T* ATOMIC_QUEUE_RESTRICT elements,
                               unsigned index) noexcept {
    auto& state = states[index];

    if constexpr (Derived::spsc_) {
      while (ATOMIC_QUEUE_UNLIKELY(state.load(A) != State::STORED)) {
        if constexpr (Derived::maximize_throughput_) spin_loop_pause();
      }
    } else {
      State expected, desired = State::LOADING;
      ATOMIC_QUEUE_LEAN_REG(desired);
      while (ATOMIC_QUEUE_UNLIKELY(!state.compare_exchange_weak(
          (expected = State::STORED), desired, A, X))) {
        do {
          spin_loop_pause();
        } while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ &&
                                       state.load(X) != State::STORED));
        ATOMIC_QUEUE_LEAN_REG(desired);
      }
    }

    T element{std::move(elements[index])};
    state.store(State::EMPTY, R);
    return element;
  }

  template <typename U, typename T>
  ATOMIC_QUEUE_INLINE void do_push(
      U&& element, std::atomic<State>* ATOMIC_QUEUE_RESTRICT states,
      T* ATOMIC_QUEUE_RESTRICT elements, unsigned index) noexcept {
    auto& state = states[index];

    if constexpr (Derived::spsc_) {
      while (ATOMIC_QUEUE_UNLIKELY(state.load(A) != State::EMPTY)) {
        if constexpr (Derived::maximize_throughput_) spin_loop_pause();
      }
    } else {
      State expected, desired = State::STORING;
      ATOMIC_QUEUE_LEAN_REG(desired);
      while (ATOMIC_QUEUE_UNLIKELY(!state.compare_exchange_weak(
          (expected = State::EMPTY), desired, A, X))) {
        do {
          spin_loop_pause();
        } while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ &&
                                       state.load(X) != State::EMPTY));
        ATOMIC_QUEUE_LEAN_REG(desired);
      }
    }

    elements[index] = std::forward<U>(element);
    state.store(State::STORED, R);
  }

 public:
  // C++23 Explicit Object Parameter (Deducing This)
  template <typename T>
  ATOMIC_QUEUE_INLINE bool try_push(this auto&& self, T&& element) noexcept {
    auto head = self.head_.load(X);
    if constexpr (Derived::spsc_) {
      if (ATOMIC_QUEUE_UNLIKELY(as_signed(head - self.tail_.load(X)) >=
                                as_signed(self.size_)))
        return false;
      self.head_.store(head + 1, X);
    } else {
      do {
        if (ATOMIC_QUEUE_UNLIKELY(as_signed(head - self.tail_.load(X)) >=
                                  as_signed(self.size_)))
          return false;
      } while (ATOMIC_QUEUE_UNLIKELY(
          !self.head_.compare_exchange_weak(head, head + 1, X, X)));
    }

    self.do_push(std::forward<T>(element), head);
    return true;
  }

  template <typename T>
  ATOMIC_QUEUE_INLINE bool try_pop(this auto&& self, T& element) noexcept {
    auto tail = self.tail_.load(X);
    if constexpr (Derived::spsc_) {
      if (ATOMIC_QUEUE_UNLIKELY(as_signed(self.head_.load(X) - tail) <= 0))
        return false;
      self.tail_.store(tail + 1, X);
    } else {
      do {
        if (ATOMIC_QUEUE_UNLIKELY(as_signed(self.head_.load(X) - tail) <= 0))
          return false;
      } while (ATOMIC_QUEUE_UNLIKELY(
          !self.tail_.compare_exchange_weak(tail, tail + 1, X, X)));
    }

    element = self.do_pop(tail);
    return true;
  }

  template <typename T>
  ATOMIC_QUEUE_INLINE void push(this auto&& self, T&& element) noexcept {
    unsigned head;
    if constexpr (Derived::spsc_) {
      head = self.head_.load(X);
      self.head_.store(head + 1, X);
    } else {
      constexpr auto memory_order = Derived::total_order_
                                        ? std::memory_order_seq_cst
                                        : std::memory_order_relaxed;
      head = self.head_.fetch_add(1, memory_order);
    }
    self.do_push(std::forward<T>(element), head);
  }

  ATOMIC_QUEUE_INLINE auto pop(this auto&& self) noexcept {
    unsigned tail;
    if constexpr (Derived::spsc_) {
      tail = self.tail_.load(X);
      self.tail_.store(tail + 1, X);
    } else {
      constexpr auto memory_order = Derived::total_order_
                                        ? std::memory_order_seq_cst
                                        : std::memory_order_relaxed;
      tail = self.tail_.fetch_add(1, memory_order);
    }
    return self.do_pop(tail);
  }

  ATOMIC_QUEUE_INLINE bool was_empty(this auto const& self) noexcept {
    return !self.was_size();
  }

  ATOMIC_QUEUE_INLINE bool was_full(this auto const& self) noexcept {
    return self.was_size() >= self.capacity();
  }

  ATOMIC_QUEUE_INLINE unsigned was_size(this auto const& self) noexcept {
    unsigned n{self.head_.load(X) - self.tail_.load(X)};
    return max_value(as_signed(n), 0);
  }

  ATOMIC_QUEUE_INLINE unsigned capacity(this auto const& self) noexcept {
    return self.size_;
  }

  static constexpr bool is_spsc() noexcept { return Derived::spsc_; }
};

}  // namespace atomic_queue