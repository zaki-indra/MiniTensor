// minitensor/core/tensorimpl.hpp

#pragma once

#include "minitensor.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace mt
{

// Forward declaration of the compute graph node
struct AutogradNode;

struct TensorImpl {
  public:
    // ---------------------------------------------------------
    // 1. Metadata
    // ---------------------------------------------------------
    Shape      shape_;
    Shape      strides_;
    DataType   dtype_  = DataType::f32;
    DeviceType device_ = DeviceType::cpu;

    // ---------------------------------------------------------
    // 2. Data Storage
    // ---------------------------------------------------------
    // For a minimal library, a flat vector is sufficient for CPU.
    // To support CUDA later, you would replace this with a custom `Storage`
    // struct that wraps a void* and a custom device allocator.
    std::vector<float> data_;

    // ---------------------------------------------------------
    // 3. Autograd State
    // ---------------------------------------------------------
    bool requires_grad_ = false;

    // A tensor is a "leaf" if it was created directly by the user (e.g., Weights, Biases)
    // rather than being the result of a math operation (e.g., the output of a Matmul).
    bool is_leaf_ = true;

    // The accumulated gradients. We use a unique pointer to a Tensor (which itself
    // wraps a TensorImpl) to lazily allocate the gradient only when needed.
    std::unique_ptr<Tensor> grad_;

    // Pointer to the node in the compute graph that created this tensor.
    // Null if this is a leaf tensor.
    std::shared_ptr<AutogradNode> grad_fn_;

    // ---------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------
    TensorImpl() = default;

    TensorImpl(Shape shape, bool requires_grad) : shape_(std::move(shape)), requires_grad_(requires_grad) {
        // Default to CPU f32 for now
        std::size_t size = 1;
        for (auto s : shape_)
            size *= s;
        data_.resize(size, 0.0f);
        compute_strides();
    }

    TensorImpl(std::vector<float> data, Shape shape, bool requires_grad)
        : shape_(std::move(shape)), data_(std::move(data)), requires_grad_(requires_grad) {
        compute_strides();
    }

    // Helper to calculate contiguous strides
    void compute_strides() {
        strides_.resize(shape_.size());
        std::size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    template <typename... Indices> float& operator[](Indices... indices);
};

// ---------------------------------------------------------
// Autograd Engine Design (The "Tape")
// ---------------------------------------------------------

// A node in the computational graph. Every math operation (Add, Matmul, ReLU)
// will inherit from this class and implement the `apply` method.
struct AutogradNode {
    virtual ~AutogradNode() = default;

    // The backward pass function. It takes the gradients flowing *backward* // from the output of this operation,
    // calculates the local gradients, and passes them back to the parent tensors.
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) = 0;

    // Defines the edges of our Directed Acyclic Graph (DAG).
    // Points to the `grad_fn_` of the input tensors that fed into this operation.
    struct Edge {
        std::shared_ptr<AutogradNode> function;
        std::size_t                   input_nr; // Which input of the next function this corresponds to
    };

    std::vector<Edge> next_edges;
};

// Example of how an operation will be structured:
/*
struct AddBackward : public AutogradNode {
  // Add backward just passes the gradient equally to both inputs
  std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
      return {grad_outputs[0], grad_outputs[0]};
  }
};
*/

} // namespace mt