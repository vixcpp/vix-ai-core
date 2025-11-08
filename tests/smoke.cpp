#include <cassert>
#include <iostream>
#include <vix/ai/core/Version.hpp>
#include <vix/ai/core/Engine.hpp>
#include <vix/ai/core/Tensor.hpp>

using namespace vix::ai::core;

int main()
{
    std::cout << "vix-ai-core version: " << version() << "\n";
    Tensor t({2, 2});
    Engine e{Device::from_string("cpu")};
    auto msg = e.compute(t);
    std::cout << msg << "\n";
    assert(t.rank() == 2);
    return 0;
}
