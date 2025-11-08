#include <cassert>
#include <string>
#include <vix/ai/core/Engine.hpp>
#include <vix/ai/core/Tensor.hpp>

using namespace vix::ai::core;

int main()
{
    Engine e{Device::from_string("cpu")};

    Tensor t({2, 3});
    t.fill(2.0f);
    auto msg = e.compute(t);
    // Just check the message contains expected info
    assert(msg.find("Engine[cpu]") != std::string::npos);
    assert(msg.find("rank=") != std::string::npos);
    assert(msg.find("numel=6") != std::string::npos);
    assert(msg.find("sum=12") != std::string::npos);
    return 0;
}
