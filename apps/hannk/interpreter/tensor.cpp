#include "interpreter/tensor.h"
#include "interpreter/model.h"

namespace hannk {

namespace {

HalideBuffer<void> make_unallocated_buffer(halide_type_t type, const Box &bounds) {
    TensorDimensions dims(bounds.size());
    int stride = 1;
    for (int i = 0; i < (int)bounds.size(); i++) {
        dims[i].min = bounds[i].min;
        dims[i].extent = bounds[i].extent();
        dims[i].stride = stride;
        stride *= dims[i].extent;
    }
    return HalideBuffer<void>(type, nullptr, (int)dims.size(), dims.data());
}

}  // namespace

struct TensorStorage final {
    HalideBuffer<void> buffer;

    TensorStorage(halide_type_t type, int rank, const halide_dimension_t *dimensions)
        : buffer(type, nullptr, rank, dimensions) {
    }

    TensorStorage() = delete;
    TensorStorage(const TensorStorage &) = delete;
    TensorStorage &operator=(const TensorStorage &) = delete;
    TensorStorage(TensorStorage &&) = delete;
    TensorStorage &operator=(TensorStorage &&) = delete;
};

Tensor::Tensor(std::string name, HalideBuffer<void> buffer, QuantizationInfo quantization)
    : name_(std::move(name)),
      buffer_(std::move(buffer)),
      quantization_(std::move(quantization)) {
}

Tensor::Tensor(std::string name, halide_type_t type, const Box &bounds, QuantizationInfo quantization)
    : Tensor(name, make_unallocated_buffer(type, bounds), quantization) {
}

void Tensor::add_consumer(Op *op) {
    consumers_.push_back(op);
}

void Tensor::add_producer(Op *op) {
    producers_.push_back(op);
}

void Tensor::remove_consumer(Op *op) {
    consumers_.remove(op);
}

void Tensor::remove_producer(Op *op) {
    producers_.remove(op);
}

std::shared_ptr<TensorStorage> Tensor::storage() {
    if (!storage_) {
        halide_buffer_t *raw_buf = buffer_.raw_buffer();
        // TensorStorage always allocates as uint.
        halide_type_t storage_type(halide_type_uint, raw_buf->type.bytes() * 8);
        storage_ = std::make_shared<TensorStorage>(storage_type, raw_buf->dimensions, raw_buf->dim);
    }
    return storage_;
}

bool Tensor::is_allocated() const {
    return buffer_.data() != nullptr;
}

void Tensor::set_external_buffer(HalideBuffer<void> external_buffer) {
    assert(!is_dynamic());
    assert(is_external());

    // No: it's ok to set this to different values over time,
    // so don't assert that host is currently null (or already equal to the new value)
    // assert(!is_allocated());

    // TODO: we don't allow aliasing of external tensors right now.
    // If we do, we need to maintain and update storage_ appropriately.
    assert(storage_ == nullptr);

    for (int i = 0; i < buffer_.dimensions(); i++) {
        assert(external_buffer.dim(i).min() == buffer_.dim(i).min());
        assert(external_buffer.dim(i).extent() == buffer_.dim(i).extent());
    }
    buffer_ = std::move(external_buffer);
}

namespace {

// Copy a Halide buffer without the internal reference counting.
// This reduces overhead of buffer copies, and is unnecessary because
// we do our own reference counting.
template<typename T>
HalideBuffer<T> drop_reference(const HalideBuffer<T> &buf) {
    halide_buffer_t *raw_buf = buf.raw_buffer();
    return HalideBuffer<T>(raw_buf->type, raw_buf->host, raw_buf->dimensions, raw_buf->dim);
}

}  // namespace

void Tensor::allocate() {
    if (buffer_.data()) {
        return;
    }

    if (is_dynamic() || is_external()) {
        return;
    }

    auto &storage_buffer = storage()->buffer;
    if (storage_buffer.data()) {
        // If our storage buffer already has data allocated, then
        // we must be an alias (ie we are sharing the storage with another Tensor)...
        assert(is_alias());
    } else {
        // ...but keep in mind that we *still* could be an alias in this branch,
        // if we are the first in a group of aliases to get allocated.
        storage_buffer.allocate();
    }

    halide_buffer_t *raw_storage_buffer = storage_buffer.raw_buffer();
    // Note that this may have a different type than storage_buffer,
    // though the *size* of the types must match!
    assert(raw_storage_buffer->type.bytes() == buffer_.type().bytes());
    HalideBuffer<void> allocated_buffer(buffer_.type(), raw_storage_buffer->host,
                                        raw_storage_buffer->dimensions, raw_storage_buffer->dim);

    if (is_alias()) {
        for (int i = 0; i < allocated_buffer.dimensions(); i++) {
            Interval dim_i(buffer_.dim(i).min(), buffer_.dim(i).max());
            if (i < (int)storage_offset_.size()) {
                dim_i += storage_offset_[i];
            }
            assert(allocated_buffer.dim(i).min() <= dim_i.min);
            assert(allocated_buffer.dim(i).max() >= dim_i.max);

            allocated_buffer.crop(i, dim_i.min, dim_i.extent());
            allocated_buffer.translate(i, -dim_i.min);
            assert(allocated_buffer.dim(i).min() == buffer_.dim(i).min());
            assert(allocated_buffer.dim(i).max() == buffer_.dim(i).max());
        }
    } else {
        // Note that storage_offset_ is sometimes empty for the is_alias=true case,
        // but should *always* be true here.
        assert(storage_offset_.empty());
    }

    buffer_ = std::move(allocated_buffer);
}

size_t Tensor::storage_size() const {
    assert(storage_ != nullptr);
    return storage_->buffer.size_in_bytes();
}

void Tensor::resize(const Box &new_shape) {
    assert(is_dynamic());
    assert(!is_external());

    TensorDimensions new_dims;

    const halide_dimension_t *old_dims = buffer_.raw_buffer()->dim;

    bool all_same = (buffer_.dimensions() == (int)new_shape.size());
    // Resizing a dynamic tensor shouldn't (AFAICT) ever change the
    // number of dimensions -- just the extents -- but let's guard
    // against that just in case, because it's easy to do.
    assert(all_same);

    int stride = 1;
    for (const auto &d : new_shape) {
        const int d_min = d.min;
        const int d_extent = d.extent();
        if (all_same && (d_min != old_dims->min || d_extent != old_dims->extent)) {
            all_same = false;
        }
        new_dims.emplace_back(d_min, d_extent, stride);
        stride *= d_extent;
    }
    if (all_same) {
        return;
    }

    HalideBuffer<void> new_buffer(buffer_.type(), nullptr, (int)new_dims.size(), new_dims.data());
    new_buffer.allocate();
    if (buffer_.data()) {
        new_buffer.copy_from(buffer_);
    }
    buffer_ = std::move(new_buffer);
    storage_ = nullptr;
}

void Tensor::set_alias_of(const TensorPtr &t, const SmallVector<int, max_rank> &storage_offset) {
    assert(!is_dynamic());
    assert(!is_external());
    assert(!is_alias());
    // No: 't' may (or may not) already have is_alias_ = true,
    // but both will be considered an alias after this call.
    // assert(!t->is_alias_);

    storage_ = t->storage();
    storage_offset_ = storage_offset;

#ifndef NDEBUG
    // Reality-check.
    Box offset_bounds = bounds();
    for (int i = 0; i < (int)storage_offset_.size(); i++) {
        offset_bounds[i] += storage_offset_[i];
    }
    auto &shared_buffer = storage_->buffer;
    assert(shared_buffer.type().bytes() == type().bytes());
    assert(shared_buffer.dimensions() == (int)offset_bounds.size());
    assert(!shared_buffer.data());

    // Check that the storage is big enough for this buffer.
    for (int i = 0; i < shared_buffer.dimensions(); i++) {
        assert(offset_bounds[i].min >= shared_buffer.dim(i).min());
        assert(offset_bounds[i].max <= shared_buffer.dim(i).max());
    }
#endif

    is_alias_ = true;
    t->is_alias_ = true;
}

void Tensor::replace_all_consumers_with(const TensorPtr &other) {
    // We need to make a copy of the list of consumers so it doesn't get invalidated
    // by set_input below.
    auto consumers = consumers_;
    for (Op *i : consumers) {
        for (int j = 0; j < i->input_count(); j++) {
            if (i->input(j).get() == this) {
                i->set_input(j, other);
            }
        }
    }
}

void Tensor::dump(std::ostream &os) const {
    os << "  " << buffer_.type() << " x ";

    const auto *b = buffer_.raw_buffer();
    os << '{';
    for (int i = 0; i < b->dimensions; i++) {
        if (i > 0) {
            os << ", ";
        }
        os << b->dim[i];
    }
    os << '}';

    if (is_allocated()) {
        os << " allocated";
    }
    if (is_constant()) {
        os << " constant";
    }
    if (is_external()) {
        os << " external";
    }
    if (is_dynamic()) {
        os << " dynamic";
    }

    os << " " << name() << std::endl;
}

}  // namespace hannk
