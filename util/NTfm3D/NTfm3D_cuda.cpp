#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

int NTfm3D_cuda_forward(const torch::Tensor *points, 
                        const torch::Tensor *masks,
                        const torch::Tensor *tfms,
                        torch::Tensor *tfmpoints);

int NTfm3D_cuda_backward(const torch::Tensor *points,
                         const torch::Tensor *masks,
                         const torch::Tensor *tfms,
                         const torch::Tensor *tfmpoints,
                         torch::Tensor *gradPoints,
                         torch::Tensor *gradMasks,
                         torch::Tensor *gradTfms,
                         const torch::Tensor *gradTfmpoints,
                         int useMaskGradMag);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x->type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x->is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int NTfm3D_forward(torch::Tensor *points, 
                   torch::Tensor *masks,
                   torch::Tensor *tfms,
                   torch::Tensor *tfmpoints) {
  CHECK_INPUT(points);
  CHECK_INPUT(masks);
  CHECK_INPUT(tfms);


  NTfm3D_cuda_forward(points, masks, tfms, tfmpoints);

  return 1;
}

int NTfm3D_backward(torch::Tensor *points, 
                    torch::Tensor *masks,
                    torch::Tensor *tfms,
                    torch::Tensor *tfmpoints,
                    torch::Tensor *gradPoints,
                    torch::Tensor *gradMasks,
                    torch::Tensor *gradTfms,
                    torch::Tensor *gradTfmpoints,
                    int useMaskGradMag) {
  CHECK_INPUT(points);
  CHECK_INPUT(masks);
  CHECK_INPUT(tfms);
  CHECK_INPUT(tfmpoints);
  CHECK_INPUT(gradTfmpoints);

  NTfm3D_cuda_backward(points, masks, tfms, tfmpoints, gradPoints, gradMasks, gradTfms, gradTfmpoints, useMaskGradMag);
  


  return 1; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &NTfm3D_forward, "NTfm3D forward (CUDA)");
  m.def("backward", &NTfm3D_backward, "NTfm3D backward (CUDA)");
}


  // auto points_accessor = points->accessor<float,4>();
  // auto masks_accessor = masks->accessor<float,4>();
  // auto tfms_accessor = tfms->accessor<float,4>();

  // // Initialize vars
  // int batchSize = points_accessor.size(0);
  // int ndim      = points_accessor.size(1);
  // int nrows     = points_accessor.size(2);
  // int ncols     = points_accessor.size(3);
  // int nSE3      = masks_accessor.size(1);
  // assert(ndim == 3); // 3D points

  // // int nTfmParams = points->numel();
  // int nTfmParams = tfms->numel();

  // // New memory in case the inputs are not contiguous
  // CHECK_INPUT(points);
  // CHECK_INPUT(masks);
  // CHECK_INPUT(tfms);

  // // Resize output and set defaults
  // tfmpoints->resize_as_(*points);

  // // Get strides
  // long ps_d[4] = {points_accessor.stride(0), points_accessor.stride(1), points_accessor.stride(2), points_accessor.stride(3)};
  // long ms_d[4] = {masks_accessor.stride(0), masks_accessor.stride(1), masks_accessor.stride(2), masks_accessor.stride(3)};
  // long ts_d[4] = {tfms_accessor.stride(0), tfms_accessor.stride(1), tfms_accessor.stride(2), tfms_accessor.stride(3)};
  // long *ps = ps_d;
  // long *ms = ms_d;
  // long *ts = ts_d;

  // // Get data pointers
  // // float *points_data    = points->data<float>();
  // // float *masks_data     = masks->data<float>();
  // // float *tfms_data      = tfms->data<float>();
  // // float *tfmpoints_data = tfmpoints->data<float>();




  // // std::cout << "Hello Cuda" << '\n';
  // // std::cout << batchSize << '\n';
  // // std::cout << ndim << '\n';
  // // std::cout << nrows << '\n';
  // // std::cout << ncols << '\n';
  // // std::cout << nTfmParams << '\n';


  // // std::cout << *points << '\n';

