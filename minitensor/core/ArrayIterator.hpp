// minitensor/core/ArrayIterator.hpp
//
// Internal kernel-launching primitive for elementwise ops. Owns dtype
// dispatch, operand validation, and the traversal loop so individual op
// sites stay focused on the math.
//
// NOTE: Despite the name, this is NOT an STL-style iterator. It is a
// PyTorch-style TensorIterator analogue. Future SIMD, parallel-for,
// broadcasting, strided traversal, and device dispatch will all land
// inside `for_each` rather than in each op site.
//
// v0 scope:
//   - contiguous storage only
//   - same dtype, same shape, same device
//   - exactly 1 output, 1 or 2 inputs
//   - serial CPU loop
//
// Lifetime: this iterator BORROWS its operands. The natural fluent
// pattern keeps them alive:
//     Config().add_output(r).add_input(a).build().for_each(...);
// Stashing the iterator across statements is a use-after-free risk.

#pragma once

#include "Macros.hpp"

#include <cstddef>
#include <minitensor/minitensor.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mt
{

class ArrayIterator;

// ---------------------------------------------------------
// Builder
// ---------------------------------------------------------
class ArrayIteratorConfig {
  public:
    ArrayIteratorConfig& add_output(Array& out);
    ArrayIteratorConfig& add_input(const Array& in);

    // Invariants checked at build() time. Defaults match v0 expectations;
    // future broadcasting / promotion can flip these without changing the
    // public API surface.
    ArrayIteratorConfig& check_same_dtype(bool v = true);
    ArrayIteratorConfig& check_same_shape(bool v = true);
    ArrayIteratorConfig& check_same_device(bool v = true);

    [[nodiscard]] ArrayIterator build() const;

  private:
    Array*                    output_ = nullptr;
    std::vector<const Array*> inputs_{};
    bool                      check_dtype_  = true;
    bool                      check_shape_  = true;
    bool                      check_device_ = true;
};

// ---------------------------------------------------------
// Iterator
// ---------------------------------------------------------
class ArrayIterator {
  public:
    // Run an elementwise kernel over every position. Arity is determined
    // by the number of inputs configured.
    //
    // Kernel convention (template lambda over the common dtype T):
    //   1 input  : []<typename T>(T x)      -> T { ... }
    //   2 inputs : []<typename T>(T x, T y) -> T { ... }
    //
    // Generic lambdas (`[](auto x, auto y) { ... }`) also work; template
    // lambdas are preferred when the body needs `if constexpr` on T.
    //
    // The kernel must be invocable for every DataType in {i32,i64,f32,f64}.

    ArrayIterator()                     = delete;
    ArrayIterator(const ArrayIterator&) = delete;
    ArrayIterator(ArrayIterator&&)      = delete;

    template <class Kernel>
    void for_each(Kernel&& k);

    [[nodiscard]] std::size_t numel() const noexcept {
        return numel_;
    }
    [[nodiscard]] DataType common_dtype() const noexcept {
        return dtype_;
    }
    [[nodiscard]] DeviceType device() const noexcept {
        return device_;
    }
    [[nodiscard]] std::size_t n_inputs() const noexcept {
        return inputs_.size();
    }

  private:
    friend class ArrayIteratorConfig;

    ArrayIterator(std::size_t numel, DataType dtype, DeviceType device, Array* output, std::vector<const Array*> inputs)
        : numel_(numel), dtype_(dtype), device_(device), output_(output), inputs_(std::move(inputs)) {
    }

    std::size_t               numel_ = 0;
    DataType                  dtype_{};
    DeviceType                device_{};
    Array*                    output_ = nullptr;
    std::vector<const Array*> inputs_{};
};

// ---------------------------------------------------------
// Inline template definitions
// ---------------------------------------------------------
template <class Kernel>
void ArrayIterator::for_each(Kernel&& k) {
    if (numel_ == 0)
        return;

    // MT_DISPATCH binds both a device tag (Dev) and a dtype alias (T)
    // for the kernel scope. Today the CPU and CUDA branches generate
    // identical code; when a CUDA backend lands, the loop body below
    // can branch on `if constexpr (std::is_same_v<Dev, mt::CudaDevice>)`
    // without touching any op site.
    // `Dev` is intentionally unused in v0 — both CPU and CUDA dispatches
    // fall through to the same serial loop. The seam is declared so a
    // future CUDA backend slots in here, not at every op site. The macro
    // marks DEVICE_TAG `[[maybe_unused]]` so this is warning-free.
    MT_DISPATCH(device_, Dev, dtype_, T, [&]() {
        T* out = output_->data<T>();
        // Kernel arity is detected at compile time via std::is_invocable_v.
        // `if constexpr` ensures only the matching branch is instantiated,
        // so unary kernels never have to compile against a binary call site
        // (and vice versa).
        if constexpr (std::is_invocable_v<Kernel&, T, T>) {
            if (inputs_.size() != 2)
                throw std::logic_error("ArrayIterator: binary kernel but inputs.size() != 2");
            const T* a = inputs_[0]->data<T>();
            const T* b = inputs_[1]->data<T>();
            for (std::size_t i = 0; i < numel_; ++i)
                out[i] = k(a[i], b[i]);
        } else if constexpr (std::is_invocable_v<Kernel&, T>) {
            if (inputs_.size() != 1)
                throw std::logic_error("ArrayIterator: unary kernel but inputs.size() != 1");
            const T* a = inputs_[0]->data<T>();
            for (std::size_t i = 0; i < numel_; ++i)
                out[i] = k(a[i]);
        } else {
            static_assert(sizeof(Kernel*) == 0, "ArrayIterator::for_each: kernel must be invocable as (T) or (T, T) "
                                                "for every supported dtype");
        }
    });
}

} // namespace mt
