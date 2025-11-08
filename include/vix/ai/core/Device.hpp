#pragma once
#include <string>
#include <stdexcept>
#include <functional>

namespace vix::ai::core
{

    enum class DeviceType
    {
        CPU,
        CUDA
    };

    class Device
    {
    public:
        explicit Device(DeviceType type = DeviceType::CPU, int index = 0) noexcept
            : type_(type), index_(index) {}

        DeviceType type() const noexcept { return type_; }
        int index() const noexcept { return index_; }

        std::string name() const
        {
            return (type_ == DeviceType::CPU ? "cpu" : "cuda") +
                   (index_ > 0 ? std::string(":") + std::to_string(index_) : std::string());
        }

        static Device from_string(const std::string &s)
        {
            if (s.rfind("cpu", 0) == 0)
                return Device(DeviceType::CPU, 0);
            if (s.rfind("cuda", 0) == 0)
            {
                int idx = 0;
                if (s.size() > 4 && s[4] == ':')
                    idx = std::stoi(s.substr(5));
                return Device(DeviceType::CUDA, idx);
            }
            throw std::invalid_argument("Unknown device string: " + s);
        }

        friend bool operator==(const Device &a, const Device &b) noexcept
        {
            return a.type_ == b.type_ && a.index_ == b.index_;
        }
        friend bool operator!=(const Device &a, const Device &b) noexcept { return !(a == b); }

    private:
        DeviceType type_{DeviceType::CPU};
        int index_{0};
    };

} // namespace vix::ai::core

// hash support
namespace std
{
    template <>
    struct hash<vix::ai::core::Device>
    {
        size_t operator()(const vix::ai::core::Device &d) const noexcept
        {
            return (static_cast<size_t>(d.type()) << 32) ^ static_cast<size_t>(d.index());
        }
    };
}
