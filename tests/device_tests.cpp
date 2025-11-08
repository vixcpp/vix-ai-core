#include <cassert>
#include <string>
#include <vix/ai/core/Device.hpp>

using namespace vix::ai::core;

int main()
{
    Device d; // default cpu:0
    assert(d.type() == DeviceType::CPU);
    assert(d.index() == 0);
    assert(d.name() == "cpu");

    {
        Device c0 = Device::from_string("cuda");
        assert(c0.type() == DeviceType::CUDA);
        assert(c0.index() == 0);
        assert(c0.name() == "cuda");
        (void)c0;
    }

    {
        Device c1 = Device::from_string("cuda:1");
        assert(c1.type() == DeviceType::CUDA);
        assert(c1.index() == 1);
        assert(c1.name() == "cuda:1");
        (void)c1;
    }

    {
        Device cpu = Device::from_string("cpu");
        assert(cpu == Device(DeviceType::CPU, 0));
        (void)cpu;
    }

    return 0;
}
