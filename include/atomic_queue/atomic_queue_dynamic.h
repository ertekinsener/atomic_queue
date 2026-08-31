#ifndef INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_DYNAMIC_
#define INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_DYNAMIC_
#include "atomic_queue_common.h"
#include "atomic_queue_bits.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <cassert>

namespace atomic_queue
{
    template <typename T, typename A = std::allocator<T>, T NIL = details::nil<T>(), bool MAXIMIZE_THROUGHPUT = true, bool TOTAL_ORDER = false, bool SPSC = false>
    class AtomicQueueB : private std::allocator_traits<A>::template rebind_alloc<std::atomic<T>>,
                         public AtomicQueueCommon<AtomicQueueB<T, A, NIL, MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>>
    {
        using AllocatorElements = typename std::allocator_traits<A>::template rebind_alloc<std::atomic<T>>;
        using Base = AtomicQueueCommon<AtomicQueueB<T, A, NIL, MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>>;
        friend Base;

        static constexpr bool total_order_ = TOTAL_ORDER;
        static constexpr bool spsc_ = SPSC;
        static constexpr bool maximize_throughput_ = MAXIMIZE_THROUGHPUT;
        static constexpr T nil_ = NIL;

        static constexpr auto ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(std::atomic<T>);
        static_assert(ELEMENTS_PER_CACHE_LINE, "Unexpected ELEMENTS_PER_CACHE_LINE.");

        static constexpr auto SHUFFLE_BITS = details::GetCacheLineIndexBits<ELEMENTS_PER_CACHE_LINE>::value;
        static_assert(SHUFFLE_BITS, "Unexpected SHUFFLE_BITS.");
        using B = details::IndexBits<SHUFFLE_BITS>;

        // AtomicQueueCommon members are stored into by readers and writers.
        // Allocate these immutable members on another cache line which never gets invalidated by stores.
        alignas(CACHE_LINE_SIZE) unsigned size_;

        // The C++ strict aliasing rules assume that pointers to the same decayed type may alias.
        // The C++ strict aliasing rules assume that pointers to any char type may alias anything and everything.
        // A dynamically allocated array may not alias anything else by construction.
        // Explicitly annotate the circular buffer array pointer as not aliasing anything else with restrict keyword.
        std::atomic<T> *ATOMIC_QUEUE_RESTRICT elements_;

        ATOMIC_QUEUE_INLINE T do_pop(unsigned tail) noexcept
        {
            auto index = remap(tail, size_, B{});
            return Base::do_pop(elements_, index);
        }

        ATOMIC_QUEUE_INLINE void do_push(T element, unsigned head) noexcept
        {
            auto index = remap(head, size_, B{});
            Base::do_push(element, elements_, index);
        }

    public:
        using value_type = T;
        using allocator_type = A;

        // The special member functions are not thread-safe.

        AtomicQueueB(unsigned size, A const &allocator = A{})
            : AllocatorElements(allocator), size_(max_value(details::round_up_to_power_of_2(size), 1u << (SHUFFLE_BITS * 2))), elements_(AllocatorElements::allocate(size_))
        {
            assert(std::atomic<T>{NIL}.is_lock_free()); // Queue element type T is not atomic. Use AtomicQueue2/AtomicQueueB2 for such element types.
            std::uninitialized_fill_n(elements_, size_, NIL);
            assert(get_allocator() == allocator); // The standard requires the original and rebound allocators to manage the same state.
        }

        AtomicQueueB(AtomicQueueB &&b) noexcept
            : AllocatorElements(static_cast<AllocatorElements &&>(b)) // TODO: This must be noexcept, static_assert that.
              ,
              Base(static_cast<Base &&>(b)), size_(std::exchange(b.size_, 0)), elements_(std::exchange(b.elements_, nullptr))
        {
        }

        AtomicQueueB &operator=(AtomicQueueB &&b) noexcept
        {
            b.swap(*this);
            return *this;
        }

        ~AtomicQueueB() noexcept
        {
            if (elements_)
            {
                details::destroy_n(elements_, size_);
                AllocatorElements::deallocate(elements_, size_); // TODO: This must be noexcept, static_assert that.
            }
        }

        A get_allocator() const noexcept
        {
            return *this; // The standard requires implicit conversion between rebound allocators.
        }

        void swap(AtomicQueueB &b) noexcept
        {
            using std::swap;
            swap(static_cast<AllocatorElements &>(*this), static_cast<AllocatorElements &>(b));
            Base::swap(b);
            swap(size_, b.size_);
            swap(elements_, b.elements_);
        }

        ATOMIC_QUEUE_INLINE friend void swap(AtomicQueueB &a, AtomicQueueB &b) noexcept
        {
            a.swap(b);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, typename A = std::allocator<T>, bool MAXIMIZE_THROUGHPUT = true, bool TOTAL_ORDER = false, bool SPSC = false>
    class AtomicQueueB2 : private std::allocator_traits<A>::template rebind_alloc<unsigned char>,
                          public AtomicQueueCommon<AtomicQueueB2<T, A, MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>>
    {
        using StorageAllocator = typename std::allocator_traits<A>::template rebind_alloc<unsigned char>;
        using Base = AtomicQueueCommon<AtomicQueueB2<T, A, MAXIMIZE_THROUGHPUT, TOTAL_ORDER, SPSC>>;
        friend Base;

        static constexpr bool total_order_ = TOTAL_ORDER;
        static constexpr bool spsc_ = SPSC;
        static constexpr bool maximize_throughput_ = MAXIMIZE_THROUGHPUT;

        // AtomicQueueCommon members are stored into by readers and writers.
        // Allocate these immutable members on another cache line which never gets invalidated by stores.
        alignas(CACHE_LINE_SIZE) unsigned size_;

        // The C++ strict aliasing rules assume that pointers to the same decayed type may alias.
        // The C++ strict aliasing rules assume that pointers to any char type may alias anything and everything.
        // A dynamically allocated array may not alias anything else by construction.
        // Explicitly annotate the circular buffer array pointers as not aliasing anything else with restrict keyword.
        AtomicState *ATOMIC_QUEUE_RESTRICT states_;
        T *ATOMIC_QUEUE_RESTRICT elements_;

        static constexpr auto STATES_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(AtomicState);
        static_assert(STATES_PER_CACHE_LINE, "Unexpected STATES_PER_CACHE_LINE.");

        static constexpr auto SHUFFLE_BITS = details::GetCacheLineIndexBits<STATES_PER_CACHE_LINE>::value;
        static_assert(SHUFFLE_BITS, "Unexpected SHUFFLE_BITS.");
        using B = details::IndexBits<SHUFFLE_BITS>;

        ATOMIC_QUEUE_INLINE T do_pop(unsigned tail) noexcept
        {
            auto index = remap(tail, size_, B{});
            return Base::do_pop(states_, elements_, index);
        }

        template <typename U>
        ATOMIC_QUEUE_INLINE void do_push(U &&element, unsigned head) noexcept
        {
            auto index = remap(head, size_, B{});
            Base::do_push(std::forward<U>(element), states_, elements_, index);
        }

        template <typename U>
        U *allocate_()
        {
            U *p = reinterpret_cast<U *>(StorageAllocator::allocate(size_ * sizeof(U)));
            assert(is_suitably_aligned(p)); // Allocated storage must be suitably aligned for U.
            return p;
        }

        template <typename U>
        void deallocate_(U *p) noexcept
        {
            StorageAllocator::deallocate(reinterpret_cast<unsigned char *>(p), size_ * sizeof(U)); // TODO: This must be noexcept, static_assert that.
        }

    public:
        using value_type = T;
        using allocator_type = A;

        // The special member functions are not thread-safe.

        AtomicQueueB2(unsigned size, A const &allocator = A{})
            : StorageAllocator(allocator), size_(max_value(details::round_up_to_power_of_2(size), 1u << (SHUFFLE_BITS * 2))), states_(allocate_<AtomicState>()), elements_(allocate_<T>())
        {
            std::uninitialized_fill_n(states_, size_, State::EMPTY);
            A a = get_allocator();
            assert(a == allocator); // The standard requires the original and rebound allocators to manage the same state.
            for (auto p = elements_, q = elements_ + size_; p < q; ++p)
                std::allocator_traits<A>::construct(a, p);
        }

        AtomicQueueB2(AtomicQueueB2 &&b) noexcept
            : StorageAllocator(static_cast<StorageAllocator &&>(b)) // TODO: This must be noexcept, static_assert that.
              ,
              Base(static_cast<Base &&>(b)), size_(std::exchange(b.size_, 0)), states_(std::exchange(b.states_, nullptr)), elements_(std::exchange(b.elements_, nullptr))
        {
        }

        AtomicQueueB2 &operator=(AtomicQueueB2 &&b) noexcept
        {
            b.swap(*this);
            return *this;
        }

        ~AtomicQueueB2() noexcept
        {
            if (elements_)
            {
                A a = get_allocator();
                for (auto p = elements_, q = elements_ + size_; p < q; ++p)
                    std::allocator_traits<A>::destroy(a, p);
                deallocate_(elements_);
                details::destroy_n(states_, size_);
                deallocate_(states_);
            }
        }

        A get_allocator() const noexcept
        {
            return *this; // The standard requires implicit conversion between rebound allocators.
        }

        void swap(AtomicQueueB2 &b) noexcept
        {
            using std::swap;
            swap(static_cast<StorageAllocator &>(*this), static_cast<StorageAllocator &>(b));
            Base::swap(b);
            swap(size_, b.size_);
            swap(states_, b.states_);
            swap(elements_, b.elements_);
        }

        ATOMIC_QUEUE_INLINE friend void swap(AtomicQueueB2 &a, AtomicQueueB2 &b) noexcept
        {
            a.swap(b);
        }
    };

} // namespace atomic_queue

#endif /* INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_DYNAMIC_ */
