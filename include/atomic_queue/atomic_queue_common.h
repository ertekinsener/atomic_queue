#ifndef INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_COMMON_
#define INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_COMMON_
#include "defs.h"

namespace atomic_queue
{
    namespace details
    {

        template <typename T>
        constexpr T nil() noexcept
        {
#if __cpp_lib_atomic_is_always_lock_free // Better compile-time error message requires C++17.
            static_assert(std::atomic<T>::is_always_lock_free, "Queue element type T is not atomic. Use AtomicQueue2/AtomicQueueB2 for such element types.");
#endif
            return {};
        }

        template <typename T>
        ATOMIC_QUEUE_SINLINE void destroy_n(T *ATOMIC_QUEUE_RESTRICT p, unsigned n) noexcept
        {
            for (auto q = p + n; p != q;)
                (p++)->~T();
        }

        template <typename T>
        ATOMIC_QUEUE_SINLINE void swap_relaxed(std::atomic<T> &a, std::atomic<T> &b) noexcept
        {
            auto a2 = a.load(X);
            a.store(b.load(X), X);
            b.store(a2, X);
        }

        template <typename T>
        ATOMIC_QUEUE_SINLINE void copy_relaxed(std::atomic<T> &a, std::atomic<T> const &b) noexcept
        {
            a.store(b.load(X), X);
        }
    }

    using State = unsigned char;
    using AtomicState = std::atomic<State>;

    enum StateE : State
    {
        EMPTY,
        STORED = 1,
        STORING = 2,
        LOADING = 4
    };

    template <typename Derived>
    class AtomicQueueCommon
    {
        ATOMIC_QUEUE_INLINE constexpr auto &downcast() noexcept { return static_cast<Derived &>(*this); }
        ATOMIC_QUEUE_INLINE constexpr auto &downcast() const noexcept { return static_cast<Derived const &>(*this); }

    protected:
        // Put these on different cache lines to avoid false sharing between readers and writers.
        alignas(CACHE_LINE_SIZE) std::atomic<unsigned> head_ = {};
        alignas(CACHE_LINE_SIZE) std::atomic<unsigned> tail_ = {};

        // The special member functions are not thread-safe.

        AtomicQueueCommon() noexcept
        {
            assert(is_suitably_aligned(&downcast()));
        }

        AtomicQueueCommon(AtomicQueueCommon const &b) noexcept
            : head_(b.head_.load(X)), tail_(b.tail_.load(X))
        {
            assert(is_suitably_aligned(&downcast()));
        }

        AtomicQueueCommon &operator=(AtomicQueueCommon const &b) noexcept
        {
            details::copy_relaxed(head_, b.head_);
            details::copy_relaxed(tail_, b.tail_);
            return *this;
        }

        // Relatively semi-special swap is not thread-safe either.
        void swap(AtomicQueueCommon &b) noexcept
        {
            details::swap_relaxed(head_, b.head_);
            details::swap_relaxed(tail_, b.tail_);
        }

        template <typename T>
        ATOMIC_QUEUE_SINLINE T do_pop(std::atomic<T> *ATOMIC_QUEUE_RESTRICT elements, unsigned index) noexcept
        {
            constexpr T NIL = Derived::nil_;
            T element;
            auto &q_element = elements[index];

            if (Derived::spsc_)
            {
                for (;;)
                {
                    element = q_element.load(A);
                    if (ATOMIC_QUEUE_LIKELY(element != NIL))
                        break;
                    if (Derived::maximize_throughput_)
                        spin_loop_pause();
                }
                q_element.store(NIL, R);
            }
            else
            {
                for (;;)
                {
                    element = q_element.exchange(NIL, AR); // (2) The store to wait for.
                    if (ATOMIC_QUEUE_LIKELY(element != NIL))
                        break;
                    // Do speculative loads while busy-waiting to avoid broadcasting RFO messages.
                    do
                        spin_loop_pause();
                    while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ && q_element.load(X) == NIL));
                }
            }
            return element;
        }

        template <typename T>
        ATOMIC_QUEUE_SINLINE void do_push(T element, std::atomic<T> *ATOMIC_QUEUE_RESTRICT elements, unsigned index) noexcept
        {
            constexpr T NIL = Derived::nil_;
            assert(element != NIL);
            auto &q_element = elements[index];

            if (Derived::spsc_)
            {
                while (ATOMIC_QUEUE_UNLIKELY(q_element.load(A) != NIL)) // Hint the branch as not taken when the queue is not full.
                    if (Derived::maximize_throughput_)
                        spin_loop_pause();
                q_element.store(element, R);
            }
            else
            {
                T expected;
                while (ATOMIC_QUEUE_UNLIKELY(!q_element.compare_exchange_weak((expected = NIL), element, AR, X))) // Hint the branch as not taken when the queue is not full.
                    do                                                                                            // Do speculative loads while busy-waiting to avoid broadcasting RFO messages.
                        spin_loop_pause();                                                                        // (1) Wait for store (2) to complete.
                    while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ && q_element.load(X) != NIL));
            }
        }

        template <typename T>
        ATOMIC_QUEUE_SINLINE T do_pop(std::atomic<State> *ATOMIC_QUEUE_RESTRICT states, T *ATOMIC_QUEUE_RESTRICT elements, unsigned index) noexcept
        {
            auto &state = states[index];

            if (Derived::spsc_)
            {
                while (ATOMIC_QUEUE_UNLIKELY(state.load(A) != STORED)) // Hint the branch as not taken when the queue is not empty.
                    if (Derived::maximize_throughput_)
                        spin_loop_pause();
            }
            else
            {
                State expected, desired = LOADING;
                ATOMIC_QUEUE_LEAN_REG(desired);
                while (ATOMIC_QUEUE_UNLIKELY(!state.compare_exchange_weak((expected = STORED), desired, A, X)))
                {      // Hint the branch as not taken when the queue is not empty.
                    do // Do speculative loads while busy-waiting to avoid broadcasting RFO messages.
                        spin_loop_pause();
                    while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ && state.load(X) != STORED));
                    ATOMIC_QUEUE_LEAN_REG(desired);
                }
            }

            T element{std::move(elements[index])};
            state.store(EMPTY, R);
            return element;
        }

        template <typename U, typename T>
        ATOMIC_QUEUE_SINLINE void do_push(U &&element, std::atomic<State> *ATOMIC_QUEUE_RESTRICT states, T *ATOMIC_QUEUE_RESTRICT elements, unsigned index) noexcept
        {
            auto &state = states[index];

            if (Derived::spsc_)
            {
                while (ATOMIC_QUEUE_UNLIKELY(state.load(A) != EMPTY)) // Hint the branch as not taken when the queue is not full.
                    if (Derived::maximize_throughput_)
                        spin_loop_pause();
            }
            else
            {
                State expected, desired = STORING;
                ATOMIC_QUEUE_LEAN_REG(desired);
                while (ATOMIC_QUEUE_UNLIKELY(!state.compare_exchange_weak((expected = EMPTY), desired, A, X)))
                {      // Hint the branch as not taken when the queue is not full.
                    do // Do speculative loads while busy-waiting to avoid broadcasting RFO messages.
                        spin_loop_pause();
                    while (ATOMIC_QUEUE_UNLIKELY(Derived::maximize_throughput_ && state.load(X) != EMPTY));
                    ATOMIC_QUEUE_LEAN_REG(desired);
                }
            }

            elements[index] = std::forward<U>(element);
            state.store(STORED, R);
        }

    public:
        template <typename T>
        ATOMIC_QUEUE_INLINE bool try_push(T &&element) noexcept
        {
            auto head = head_.load(X);
            if (Derived::spsc_)
            {
                if (ATOMIC_QUEUE_UNLIKELY(as_signed(head - tail_.load(X)) >= as_signed(downcast().size_)))
                    return false;
                head_.store(head + 1, X);
            }
            else
            {
                do
                {
                    if (ATOMIC_QUEUE_UNLIKELY(as_signed(head - tail_.load(X)) >= as_signed(downcast().size_)))
                        return false;
                } while (ATOMIC_QUEUE_UNLIKELY(!head_.compare_exchange_weak(head, head + 1, X, X))); // This loop is not FIFO.
            }

            downcast().do_push(std::forward<T>(element), head);
            return true;
        }

        template <typename T>
        ATOMIC_QUEUE_INLINE bool try_pop(T &element) noexcept
        {
            auto tail = tail_.load(X);
            if (Derived::spsc_)
            {
                if (ATOMIC_QUEUE_UNLIKELY(as_signed(head_.load(X) - tail) <= 0))
                    return false;
                tail_.store(tail + 1, X);
            }
            else
            {
                do
                {
                    if (ATOMIC_QUEUE_UNLIKELY(as_signed(head_.load(X) - tail) <= 0))
                        return false;
                } while (ATOMIC_QUEUE_UNLIKELY(!tail_.compare_exchange_weak(tail, tail + 1, X, X))); // This loop is not FIFO.
            }

            element = downcast().do_pop(tail);
            return true;
        }

        template <typename T>
        ATOMIC_QUEUE_INLINE void push(T &&element) noexcept
        {
            unsigned head;
            if (Derived::spsc_)
            {
                head = head_.load(X);
                head_.store(head + 1, X);
            }
            else
            {
                constexpr auto memory_order = Derived::total_order_ ? std::memory_order_seq_cst : std::memory_order_relaxed;
                head = head_.fetch_add(1, memory_order); // FIFO and total order on Intel regardless, as of 2019.
            }
            downcast().do_push(std::forward<T>(element), head);
        }

        ATOMIC_QUEUE_INLINE auto pop() noexcept
        {
            unsigned tail;
            if (Derived::spsc_)
            {
                tail = tail_.load(X);
                tail_.store(tail + 1, X);
            }
            else
            {
                constexpr auto memory_order = Derived::total_order_ ? std::memory_order_seq_cst : std::memory_order_relaxed;
                tail = tail_.fetch_add(1, memory_order); // FIFO and total order on Intel regardless, as of 2019.
            }
            return downcast().do_pop(tail);
        }

        ATOMIC_QUEUE_INLINE bool was_empty() const noexcept
        {
            return !was_size();
        }

        ATOMIC_QUEUE_INLINE bool was_full() const noexcept
        {
            return was_size() >= capacity();
        }

        ATOMIC_QUEUE_INLINE unsigned was_size() const noexcept
        {
            // tail_ can be greater than head_ because of consumers doing pop, rather that try_pop, when the queue is empty.
            unsigned n{head_.load(X) - tail_.load(X)};
            return max_value(as_signed(n), 0);
        }

        ATOMIC_QUEUE_INLINE unsigned capacity() const noexcept
        {
            return downcast().size_;
        }

        ATOMIC_QUEUE_SINLINE constexpr bool is_spsc() noexcept
        {
            return Derived::spsc_;
        }
    };

} // namespace atomic_queue
#endif /* INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_COMMON_ */
