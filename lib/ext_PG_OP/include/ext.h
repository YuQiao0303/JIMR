void sec_rand_k_cuda(int nProposal, int k, float *rand_nums, int *offsets, float *indices, float *output);
void sec_top_k_cuda(int nProposal, int k, float *scores, int *offsets, float *indices, float *output);
void sec_first_k_cuda(int nProposal, int C, int k, float *inp, int *offsets, float *out);
void sec_argsort_cuda(int nProposal, int C, float *inp, int *offsets, float *indices);
void sec_weighted_mean_cuda(int nProposal, int C, float *inp, float *weights, float weights_sum,int *offsets, float *out);