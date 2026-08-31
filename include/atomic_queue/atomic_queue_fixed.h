#ifndef INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_FIXED_
#define INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_FIXED_
#include "atomic_queue_bits.h"
#include "atomic_queue_common.h"
namespace atomic_queue {
template <typename T, unsigned SIZE, T NIL = details::nil<T>(),
          bool MINIMIZE_CONTENTION = true, bool MAXIMIZE_THROUGHPUT = true,
          bool TOTAL_ORDER = false, bool SPSC = false>
class AtomicQueue : public AtomicQueueCommon<
                        AtomicQueue<T, SIZE, NIL, MINIMIZE_CONTENTION,
                                    MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>> {
  using Base =
      AtomicQueueCommon<AtomicQueue<T, SIZE, NIL, MINIMIZE_CONTENTION,
                                    MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>>;
  friend Base;

  static constexpr unsigned size_ =
      MINIMIZE_CONTENTION ? details::round_up_to_power_of_2(SIZE) : SIZE;
  static constexpr int SHUFFLE_BITS =
      details::GetIndexShuffleBits<MINIMIZE_CONTENTION, size_,
                                   CACHE_LINE_SIZE /
                                       sizeof(std::atomic<T>)>::value;
  using B = details::IndexBits<SHUFFLE_BITS>;
  static constexpr bool total_order_ = TOTAL_ORDER;
  static constexpr bool spsc_ = SPSC;
  static constexpr bool maximize_throughput_ = MAXIMIZE_THROUGHPUT;
  static constexpr T nil_ = NIL;

  alignas(CACHE_LINE_SIZE) std::atomic<T> elements_[size_];

  ATOMIC_QUEUE_INLINE T do_pop(unsigned tail) noexcept {
    auto index = remap(tail, size_, B{});
    return Base::do_pop(elements_, index);
  }

  ATOMIC_QUEUE_INLINE void do_push(T element, unsigned head) noexcept {
    auto index = remap(head, size_, B{});
    Base::do_push(element, elements_, index);
  }

 public:
  using value_type = T;

  AtomicQueue() noexcept {
    assert(std::atomic<T>{NIL}
               .is_lock_free());  // Queue element type T is not atomic. Use
                                  // AtomicQueue2/AtomicQueueB2 for such element
                                  // types.
    for (auto p = elements_, q = elements_ + size_; p != q; ++p)
      p->store(NIL, X);
  }

  AtomicQueue(AtomicQueue const&) = delete;
  AtomicQueue& operator=(AtomicQueue const&) = delete;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, unsigned SIZE, bool MINIMIZE_CONTENTION = true,
          bool MAXIMIZE_THROUGHPUT = true, bool TOTAL_ORDER = false,
          bool SPSC = false>
class AtomicQueue2 : public AtomicQueueCommon<
                         AtomicQueue2<T, SIZE, MINIMIZE_CONTENTION,
                                      MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>> {
  using Base =
      AtomicQueueCommon<AtomicQueue2<T, SIZE, MINIMIZE_CONTENTION,
                                     MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>>;
  friend Base;

  static constexpr unsigned size_ =
      MINIMIZE_CONTENTION ? details::round_up_to_power_of_2(SIZE) : SIZE;
  static constexpr int SHUFFLE_BITS = details::GetIndexShuffleBits<
      MINIMIZE_CONTENTION, size_, CACHE_LINE_SIZE / sizeof(AtomicState)>::value;
  using B = details::IndexBits<SHUFFLE_BITS>;
  static constexpr bool total_order_ = TOTAL_ORDER;
  static constexpr bool spsc_ = SPSC;
  static constexpr bool maximize_throughput_ = MAXIMIZE_THROUGHPUT;

  alignas(CACHE_LINE_SIZE) AtomicState states_[size_] = {};
  alignas(CACHE_LINE_SIZE) T elements_[size_] = {};

  ATOMIC_QUEUE_INLINE T do_pop(unsigned tail) noexcept {
    auto index = remap(tail, size_, B{});
    return Base::do_pop(states_, elements_, index);
  }

  template <typename U>
  ATOMIC_QUEUE_INLINE void do_push(U&& element, unsigned head) noexcept {
    auto index = remap(head, size_, B{});
    Base::do_push(std::forward<U>(element), states_, elements_, index);
  }

 public:
  using value_type = T;

  AtomicQueue2() noexcept = default;
  AtomicQueue2(AtomicQueue2 const&) = delete;
  AtomicQueue2& operator=(AtomicQueue2 const&) = delete;
};
}  // namespace atomic_queue
#endif /* INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_FIXED_ */
