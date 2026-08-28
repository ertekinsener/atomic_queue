/* -*- mode: c++; c-basic-offset: 4; indent-tabs-mode: nil; tab-width: 4 -*- */
#ifndef ATOMIC_QUEUE_BENCHMARKS_H_INCLUDED
#define ATOMIC_QUEUE_BENCHMARKS_H_INCLUDED

#include <utility>

#include "atomic_queue/defs.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace atomic_queue {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Context {
    int producers;
    int consumers;
};

template<typename T> typename T::ContextType context_of_(int);
template<typename T> NoContext context_of_(long);
template<typename T> using ContextOf = decltype(context_of_<T>(0));

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct NoToken {
    template<typename... Args>
    ATOMIC_QUEUE_INLINE constexpr NoToken(Args&&...) noexcept {}

    template<typename Queue, typename T>
    ATOMIC_QUEUE_INLINE static void push(Queue& q, T&& element) noexcept {
        q.push(std::forward<T>(element));
    }

    template<typename Queue>
    ATOMIC_QUEUE_INLINE static auto pop(Queue& q) noexcept {
        return q.pop();
    }
};

template<typename T> typename T::Producer producer_of_(int);
template<typename T> NoToken producer_of_(long);
template<typename T> using ProducerOf = decltype(producer_of_<T>(1));

template<typename T> typename T::Consumer consumer_of_(int);
template<typename T> NoToken consumer_of_(long);
template<typename T> using ConsumerOf = decltype(consumer_of_<T>(1));

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Queue, size_t Capacity>
struct CapacityArgAdaptor : Queue {
    ATOMIC_QUEUE_INLINE CapacityArgAdaptor()
        : Queue(Capacity)
    {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Queue>
struct RetryDecorator : Queue {
    using T = typename Queue::value_type;

    using Queue::Queue;

    ATOMIC_QUEUE_INLINE void push(T element) noexcept {
        while(ATOMIC_QUEUE_UNLIKELY(!this->try_push(element)))
            spin_loop_pause();
    }

    ATOMIC_QUEUE_INLINE T pop() noexcept {
        T element;
        while(ATOMIC_QUEUE_UNLIKELY(!this->try_pop(element)))
            spin_loop_pause();
        return element;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // atomic_queue

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // ATOMIC_QUEUE_BENCHMARKS_H_INCLUDED
