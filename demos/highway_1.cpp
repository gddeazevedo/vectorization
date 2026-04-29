#include <cstdio>
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

void AddArrays(
    const float *HWY_RESTRICT a,
    const float *HWY_RESTRICT b,
    float *HWY_RESTRICT out,
    const size_t n)
{
    const hn::ScalableTag<float> d;
    const size_t N_LANES = hn::Lanes(d);

    size_t i = 0;
    for (; i + N_LANES <= n; i += N_LANES)
    {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        auto vsum = hn::Add(va, vb);
        hn::Store(vsum, d, out + i);
    }
    for (; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
    constexpr size_t n = 10;
    float a[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[n] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    float out[n];

    AddArrays(a, b, out, n);

    for (size_t i = 0; i < n; ++i)
    {
        std::printf("%.1f ", out[i]);
    }
    std::printf("\n");
    return 0;
}