#pragma once
#include <cstddef>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <initializer_list>

namespace vix::ai::core
{

    class Tensor
    {
    public:
        Tensor() = default;

        explicit Tensor(std::vector<std::size_t> shape)
            : shape_(std::move(shape)), data_(numel(), 0.0f) {}

        Tensor(std::initializer_list<std::size_t> shape)
            : Tensor(std::vector<std::size_t>(shape)) {}

        const std::vector<std::size_t> &shape() const noexcept { return shape_; }
        std::size_t rank() const noexcept { return shape_.size(); }

        std::size_t numel() const noexcept
        {
            if (shape_.empty())
                return 0;
            return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                                   [](std::size_t a, std::size_t b)
                                   { return a * b; });
        }

        std::vector<float> &data() noexcept { return data_; }
        const std::vector<float> &data() const noexcept { return data_; }

        float *raw() noexcept { return data_.data(); }
        const float *raw() const noexcept { return data_.data(); }

        void reshape(const std::vector<std::size_t> &new_shape)
        {
            std::size_t old = numel();
            std::size_t now = std::accumulate(new_shape.begin(), new_shape.end(), std::size_t{1},
                                              [](std::size_t a, std::size_t b)
                                              { return a * b; });
            if (old != now)
                throw std::invalid_argument("reshape: size mismatch");
            shape_ = new_shape;
        }

        void fill(float v)
        {
            std::fill(data_.begin(), data_.end(), v);
        }

        static Tensor zeros(std::initializer_list<std::size_t> shape)
        {
            Tensor t{shape};
            t.fill(0.0f);
            return t;
        }
        static Tensor ones(std::initializer_list<std::size_t> shape)
        {
            Tensor t{shape};
            t.fill(1.0f);
            return t;
        }

    private:
        std::vector<std::size_t> shape_{};
        std::vector<float> data_{};
    };

} // namespace vix::ai::core
