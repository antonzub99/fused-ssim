#include <torch/extension.h>
#include "ssim.h"
#include "ssim_bfloat.h"
#include "ssim_half.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
  m.def("fusedssim_bfloat16", &fusedssim_bfloat16);
  m.def("fusedssim_bfloat16_backward", &fusedssim_bfloat16_backward);
  m.def("fusedssim_half", &fusedssim_half);
  m.def("fusedssim_half_backward", &fusedssim_half_backward);
}
