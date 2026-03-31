#include <iostream>
#include <vector>
#include <hwy/highway.h>

// ============================
// Implementação vetorizada
// ============================
HWY_BEFORE_NAMESPACE();
namespace example {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void VecAdd(const float* HWY_RESTRICT a,
            const float* HWY_RESTRICT b,
            float* HWY_RESTRICT out,
            size_t n) {
    
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    size_t i = 0;

    // Loop vetorizado
    for (; i + lanes <= n; i += lanes) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        auto vc = hn::Add(va, vb);
        hn::Store(vc, d, out + i);
    }

    // Tail (resto)
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace example
HWY_AFTER_NAMESPACE();

// ============================
// Wrapper (sem dynamic dispatch)
// ============================
namespace example {
HWY_EXPORT(VecAdd); 
inline void VecAddStatic(const float* a,
                         const float* b,
                         float* out,
                         size_t n) {
    HWY_STATIC_DISPATCH(VecAdd)(a, b, out, n);
}

}  // namespace example

int main() {
    constexpr size_t N = 16;

    std::vector<float> a(N), b(N), out(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
        out[i] = 0;
    }

    // example::VecAddStatic(a.data(), b.data(), out.data(), N);

    // // Print resultado
    for (size_t i = 0; i < N; ++i) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}