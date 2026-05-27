// minitensor/core/ArrayIterator.cpp

#include "ArrayIterator.hpp"
#include "ErrorMacros.hpp"

#include <minitensor/minitensor.hpp>

namespace mt
{

// ---------------------------------------------------------
// ArrayIteratorConfig
// ---------------------------------------------------------
ArrayIteratorConfig& ArrayIteratorConfig::add_output(Array& out) {
    output_ = &out;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::add_input(const Array& in) {
    inputs_.push_back(&in);
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::check_same_dtype(bool v) {
    check_dtype_ = v;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::check_same_shape(bool v) {
    check_shape_ = v;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::check_same_device(bool v) {
    check_device_ = v;
    return *this;
}

ArrayIterator ArrayIteratorConfig::build() const {
    // --- structural checks --------------------------------------------------
    MT_CHECK(output_ != nullptr, mt::ShapeError, "ArrayIterator: no output configured");
    MT_CHECK(!inputs_.empty(), mt::ShapeError, "ArrayIterator: no inputs configured");
    MT_CHECK(inputs_.size() <= 2, mt::ShapeError,
             "ArrayIterator v0 supports at most 2 inputs, got " << inputs_.size());

    // --- defined checks -----------------------------------------------------
    MT_CHECK_DEFINED(*output_);
    for (const Array* in : inputs_) {
        MT_CHECK_DEFINED(*in);
    }

    // --- pairwise invariants (output vs each input) -------------------------
    const Array& ref = *output_;
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
        const Array& in = *inputs_[i];
        if (check_device_) {
            MT_CHECK(in.device() == ref.device(), mt::DeviceError,
                     "Device mismatch! Input " << i << " is on " << in.device() << " but output is on "
                                               << ref.device());
        }
        if (check_dtype_) {
            MT_CHECK(in.dtype() == ref.dtype(), mt::DTypeError,
                     "DataType mismatch! Input " << i << " has dtype " << in.dtype() << " but output has dtype "
                                                 << ref.dtype());
        }
        if (check_shape_) {
            MT_CHECK(in.shape() == ref.shape(), mt::ShapeError,
                     "Shape mismatch! Input " << i << " has shape " << in.shape() << " but output has shape "
                                              << ref.shape());
        }
    }

    return ArrayIterator(ref.numel(), ref.dtype(), ref.device(), output_, inputs_);
}

} // namespace mt
