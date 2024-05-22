#include <stdio.h>
// #include <math.h>
#include <thrust/sort.h>

/* ================================== sec_randK ================================== */
// This kernel function is to get indices with top k  scores of each proposal.
__global__ void sec_rand_k_cuda_(int nProposal,int k, float *rand_nums,  int *offsets, float *indices, float *output){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];
        int count = end - start;
        //thrust::sort_by_key(thrust::device,  scores + start , scores + end, indices+ start);
        for(int i=0;i<k;i++)
        {
         
            output[p_id*k + i] = indices[start+((int)(rand_nums[i]* count))%count];
//             output[p_id*k + i] = start+(int)(rand_nums[i]);
        }

    }
}
//input: scores (N) float
//input: offsets (nProposal + 1) int
//input: indices (N) int
//input: output (nProposal, K) int

void sec_rand_k_cuda(int nProposal, int k, float *rand_nums, int *offsets, float *indices, float *output){
    sec_rand_k_cuda_<<<std::min(nProposal, (int)32768), 1>>>(nProposal, k, rand_nums, offsets, indices,output);
}

/* ================================== sec_topK ================================== */
// This kernel function is to get indices with top k  scores of each proposal.
__global__ void sec_top_k_cuda_(int nProposal,int k, float *scores,  int *offsets, float *indices, float *output){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];
        int count = end - start;
        thrust::sort_by_key(thrust::device,  scores + start , scores + end, indices+ start);
        for(int i=0;i<k;i++)
        {
            output[p_id*k + i] = indices[start+i%count];
        }

    }
}
//input: scores (N) float
//input: offsets (nProposal + 1) int
//input: indices (N) int
//input: output (nProposal, K) int

void sec_top_k_cuda(int nProposal, int k, float *scores, int *offsets, float *indices, float *output){
    sec_top_k_cuda_<<<std::min(nProposal, (int)32768), 1>>>(nProposal, k, scores, offsets, indices,output);
}


/* ================================== sec_first_k ================================== */
// This is for indices as inputs, so C must be 1.
__global__ void sec_first_k_cuda_(int nProposal, int C, int k, float *inp,  int *offsets, float *out)
{
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x)
    {
        int start = offsets[p_id];
        int end = offsets[p_id + 1];
        int count = end - start;

        for(int i = 0; i < k; i++)
        {
            out[p_id*k + i] = inp[start+i%count];
        }
    }

}
//input: inp (N, C) float (it should be indices and C == 1)
//input: out (nProposal, k) int
//input: offsets (nProposal + 1) int
void sec_first_k_cuda(int nProposal, int C, int k, float *inp, int *offsets, float *out){
    sec_first_k_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C,k, inp, offsets, out);
}



/* ================================== sec_argsort ================================== */
// This is to sort indices by 1-dim scores.  So C must be 1.
__global__ void sec_argsort_cuda_(int nProposal, int C, float *inp,  int *offsets, float *indices){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];
        float count = (float)(end - start);
        thrust::sort_by_key(thrust::device,  inp + start , inp + end, indices+ start);

    }
}
//input: inp (N, C) float (it should be scores and C == 1)
//input: indices (N, 1) int
//input: offsets (nProposal + 1) int
void sec_argsort_cuda(int nProposal, int C, float *inp, int *offsets, float *indices){
    sec_argsort_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, inp, offsets, indices);
}

/* ================================== sec_weighted_mean ================================== */
__global__ void sec_weighted_mean_cuda_(int nProposal, int C, float *inp, float *weights, float weights_sum, int *offsets, float *out){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];

        float count = (float)(end - start);

        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            float mean = 0;
            float weights_sum = 0;
            for(int i = start; i < end; i++){
//                 mean += (inp[i * C + plane] * weights[i * C + plane] / weights_sum);

//                 mean += (inp[i * C + plane] * weights[i * C + plane] );
//                 weights_sum += weights[i * C + plane];

                mean += (inp[i * C + plane] * weights[i  ] );
                weights_sum += weights[i ];
            }
            out[p_id * C + plane] = mean / weights_sum;
        }
    }
}

//input: inp (N, C) float
//input: offsets (nProposal + 1) int
//output: out (nProposal, C) float
void sec_weighted_mean_cuda(int nProposal, int C, float *inp, float *weights,float weights_sum, int *offsets, float *out){
    sec_weighted_mean_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, inp, weights,weights_sum, offsets, out);
}
