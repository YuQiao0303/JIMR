#include <torch/extension.h>
#include "ext.h"

void sec_rand_k(torch::Tensor &rand_num_tensor,torch::Tensor &offsets_tensor,torch::Tensor &indices_tensor,torch::Tensor &out_tensor, int nProposal, int k) {
     int *offsets =  (int *)offsets_tensor.data_ptr();
     float *rand_nums = (float *)rand_num_tensor.data_ptr();
     float *indices = (float *)indices_tensor.data_ptr();
     float *out = (float *)out_tensor.data_ptr();
     sec_rand_k_cuda(nProposal, k, rand_nums, offsets,indices, out);
     //       should (int,     int, float*, int*,   int*,   int*)
     //          but (int,     int, int,    float*, int*,   float*)
     //invalid conversion from _float*_ to _int_
     //sec_top_k_cuda(int nProposal, int k, float *scores, int *offsets, int *indices, int *output){

}


void sec_top_k(torch::Tensor &scores_tensor,torch::Tensor &offsets_tensor,torch::Tensor &indices_tensor,torch::Tensor &out_tensor, int nProposal, int k) {
     int *offsets =  (int *)offsets_tensor.data_ptr();
     float *scores = (float *)scores_tensor.data_ptr();
     float *indices = (float *)indices_tensor.data_ptr();
     float *out = (float *)out_tensor.data_ptr();
     sec_top_k_cuda(nProposal, k, scores, offsets,indices, out);
     //       should (int,     int, float*, int*,   int*,   int*)
     //          but (int,     int, int,    float*, int*,   float*)
     //invalid conversion from _float*_ to _int_
     //sec_top_k_cuda(int nProposal, int k, float *scores, int *offsets, int *indices, int *output){

}

void sec_first_k(torch::Tensor &inp_tensor,torch::Tensor &offsets_tensor,torch::Tensor &out_tensor, int nProposal, int C, int k) {
     int *offsets =  (int *)offsets_tensor.data_ptr();
     float *inp = (float *)inp_tensor.data_ptr();
     float *out = (float *)out_tensor.data_ptr();
     sec_first_k_cuda(nProposal, C,k, inp, offsets, out);
}


void sec_argsort(torch::Tensor &inp_tensor,torch::Tensor &offsets_tensor,torch::Tensor &out_tensor, int nProposal, int C) {
     int *offsets =  (int *)offsets_tensor.data_ptr();
     float *inp = (float *)inp_tensor.data_ptr();
     float *out = (float *)out_tensor.data_ptr();
     sec_argsort_cuda(nProposal, C, inp, offsets, out);
}

void sec_weighted_mean(torch::Tensor &inp_tensor, torch::Tensor &weights_tensor,  float weights_sum, torch::Tensor &offsets_tensor,torch::Tensor &out_tensor, int nProposal, int C){
    int *offsets =  (int *)offsets_tensor.data_ptr();
    float *inp =  (float *)inp_tensor.data_ptr();
    float *out = (float *) out_tensor.data_ptr();
    float *weights = (float *)  weights_tensor.data_ptr();
    //float *weights_sum = (float *)  weights_sum_tensor.data_ptr();
    sec_weighted_mean_cuda( nProposal,  C,  inp,  weights, weights_sum, offsets,  out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sec_rand_k",&sec_rand_k,"sec_rand_k");
    m.def("sec_top_k",&sec_top_k,"sec_top_k");
    m.def("sec_first_k",&sec_first_k,"sec_first_k");
    m.def("sec_argsort",&sec_argsort,"sec_argsort");
    m.def("sec_weighted_mean",&sec_weighted_mean,"sec_weighted_mean");
}

//TORCH_LIBRARY(add2, m) {
//    m.def("torch_launch_add2", torch_launch_add2);
//}