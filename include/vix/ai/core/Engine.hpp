#pragma once
#include <string>
#include <memory>
#include <numeric>
#include "Tensor.hpp"
#include "Device.hpp"

namespace vix::ai::core
{

    class Engine
    {
    public:
        Engine() = default;
        explicit Engine(Device dev) : device_(std::move(dev)) {}

        const Device &device() const noexcept { return device_; }

        std::string compute(const Tensor &t) const
        {
            // mini “kernel”: somme des éléments pour vérifier le chemin critique
            float sum = std::accumulate(t.data().begin(), t.data().end(), 0.0f);
            return std::string{"Engine["} + device().name() + "] rank=" +
                   std::to_string(t.rank()) + " numel=" + std::to_string(t.numel()) +
                   " sum=" + std::to_string(sum);
        }

        std::string info() const
        {
            return std::string{"vix-ai-core Engine on "} + device().name();
        }

    private:
        Device device_{};
    };

} // namespace vix::ai::core
