#ifndef INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_BITS_
#define INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_BITS_

#include <bit>
#include <cstddef>

#include "defs.h"
namespace atomic_queue::details {
using std::uint32_t;
using std::uint64_t;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <std::size_t ElementsPerCacheLine>
struct GetCacheLineIndexBits {
  static constexpr int value =
      std::has_single_bit(ElementsPerCacheLine)
          ? static_cast<int>(std::bit_width(ElementsPerCacheLine) - 1)
          : 0;
};

template <bool MinimizeContention, unsigned ArraySize,
          std::size_t ElementsPerCacheLine>
struct GetIndexShuffleBits {
  static constexpr int value = []() constexpr -> int {
    if constexpr (!MinimizeContention) {
      return 0;
    } else {
      constexpr int bits = GetCacheLineIndexBits<ElementsPerCacheLine>::value;

      // max_shift dışarı taşmaz, lambda içinde lokal kalır
      constexpr std::size_t max_shift = sizeof(std::size_t) * 8;
      constexpr std::size_t min_size =
          (bits * 2 < max_shift) ? (std::size_t{1} << (bits * 2)) : 0;

      return (min_size != 0 && ArraySize >= min_size) ? bits : 0;
    }
  }();
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Multiple writers/readers contend on the same cache line when storing/loading
// elements at subsequent indexes, aka false sharing. For power of 2 ring buffer
// size it is possible to re-map the index in such a way that each subsequent
// element resides on another cache line, which minimizes contention. This is
// done by swapping the lowest order N bits (which are the index of the element
// within the cache line) with the next N bits (which are the index of the cache
// line) of the element index.

template <unsigned N_BITS>
struct IndexBits {
  enum : unsigned {
    mask_elem_idx = ~(~0u << N_BITS),
    mask_line_idx = mask_elem_idx << N_BITS,
    mask_hi = ~0u << (2 * N_BITS),
    count = N_BITS,
    count2 = N_BITS << 8 | N_BITS,
  };
};

struct RemapXor {
  // Each step depends on the previous, a serial chain of ~6 instructions, ~6
  // cycles.
  template <typename B>
  ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index, B) noexcept {
    unsigned const mix{(index ^ (index >> B::count)) & B::mask_elem_idx};
    return index ^ mix ^ (mix << B::count);
  }

  template <typename B>
  ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index, unsigned size,
                                                B b) noexcept {
    return remap(index & (size - 1), b);
  }
};

struct RemapAnd {
  // Faster index remapping with independent parallel computations of index
  // components. The shifts and ands dispatch in parallel, ~8 instructions, ~4
  // cycles. At least +1% faster throughput benchmark relative to RemapXor.
  template <typename B>
  ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index, unsigned size,
                                                B) noexcept {
    return ((index >> B::count) & B::mask_elem_idx) |
           ((index & B::mask_elem_idx) << B::count) |
           (index & (B::mask_hi & (size - 1)));
  }

  template <typename B>
  ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index, B b) noexcept {
    return remap(index, 0, b);
  }
};

#ifdef __BMI__
struct RemapBmi {
  // Shorter and faster machine code for swapping bits with BMI instructions, if
  // available. BMI1 (and, bextr, mov + and) dispatch in parallel, 7
  // instructions, ~3 cycles. BMI2 (and, bextr, bzhi) dispatch in parallel, 6
  // instructions, ~3 cycles. At least +1.5% faster throughput benchmark
  // relative to RemapXor.
  template <typename B>
  ATOMIC_QUEUE_SINLINE unsigned remap(unsigned index, unsigned size,
                                      B) noexcept {
    static_assert(ATOMIC_QUEUE_FULL_THROTTLE == 1,
                  "Unexpected ATOMIC_QUEUE_FULL_THROTTLE value.");
    unsigned nn = B::count2;
    ATOMIC_QUEUE_REG(nn);  // Disable constant propagation for nn to prevent the
                           // compiler from transforming the following code.

#ifdef __BMI2__
    unsigned new_line_idx = _bzhi_u32(index, nn)
                            << B::count;  // BMI2 bzhi supersedes mov + and.
    // unsigned new_line_idx = (index << nn) & B::mask_line_idx; // BMI2 shlx
    // supersedes mov + shl.
    unsigned new_elem_idx =
        __bextr_u32(index, nn);  // BMI1 bextr supersedes mov + shr + and.
#else
    unsigned new_elem_idx =
        __bextr_u32(index, nn);  // BMI1 bextr supersedes mov + shr + and.
    unsigned new_line_idx = (index & B::mask_elem_idx) << B::count;
#endif

    new_elem_idx |= index & (B::mask_hi & (size - 1));
    ATOMIC_QUEUE_ORDER(new_elem_idx,
                       new_line_idx);    // Do not commute the arguments of the
                                         // adjacent two or instructions.
    return new_elem_idx | new_line_idx;  // Or with new_line_idx last.
  }

  template <typename B>
  ATOMIC_QUEUE_SINLINE unsigned remap(unsigned index, B b) noexcept {
    return remap(index, 0, b);
  }
};
#endif  // __BMI__

template <typename Remap>
struct Remap0 : Remap {
  using Remap::remap;

  ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index, unsigned size,
                                                IndexBits<0>) noexcept {
    return index % size;
  }

  ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index,
                                                IndexBits<0>) noexcept {
    return index;
  }

  template <typename B, typename... A>
  ATOMIC_QUEUE_INLINE auto operator()(B bits, A... a) const noexcept {
    return this->remap(a..., bits);
  }
};

#ifdef ATOMIC_QUEUE_REMAP
// Defining ATOMIC_QUEUE_REMAP overrides the default remapper.
using Remap = Remap0<ATOMIC_QUEUE_REMAP>;
#elif defined(__BMI__)
using Remap = Remap0<RemapBmi>;
#else
using Remap = Remap0<RemapAnd>;
#endif

template <unsigned N_BITS>
ATOMIC_QUEUE_SINLINE constexpr unsigned remap(unsigned index, unsigned size,
                                              IndexBits<N_BITS> b) noexcept {
  return Remap::remap(index, size, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Implement a "bit-twiddling hack" for finding the next power of 2 in either 32
// bits or 64 bits in C++11 compatible constexpr functions. The library no
// longer maintains C++11 compatibility.

// "Runtime" version for 32 bits
// --a;
// a |= a >> 1;
// a |= a >> 2;
// a |= a >> 4;
// a |= a >> 8;
// a |= a >> 16;
// ++a;

template <typename T>
ATOMIC_QUEUE_SINLINE constexpr T decrement(T x) noexcept {
  return x - 1;
}

template <typename T>
ATOMIC_QUEUE_SINLINE constexpr T increment(T x) noexcept {
  return x + 1;
}

template <typename T>
ATOMIC_QUEUE_SINLINE constexpr T or_equal(T x, unsigned u) noexcept {
  return x | x >> u;
}

template <typename T, typename... Args>
ATOMIC_QUEUE_SINLINE constexpr T or_equal(T x, unsigned u,
                                          Args... rest) noexcept {
  return or_equal(or_equal(x, u), rest...);
}

ATOMIC_QUEUE_SINLINE constexpr uint32_t round_up_to_power_of_2(
    uint32_t a) noexcept {
  return increment(or_equal(decrement(a), 1, 2, 4, 8, 16));
}

ATOMIC_QUEUE_SINLINE constexpr uint64_t round_up_to_power_of_2(
    uint64_t a) noexcept {
  return increment(or_equal(decrement(a), 1, 2, 4, 8, 16, 32));
}

}  // namespace atomic_queue::details

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* INCLUDE_ATOMIC_QUEUE_ATOMIC_QUEUE_BITS_ */
