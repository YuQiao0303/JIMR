import torch
from torch.autograd import Function
import ext_PG_OP


class SecRandKIdx(Function):
    @staticmethod
    def forward(ctx, k, offsets,rand_nums=None):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1

        N = offsets[-1]

        assert offsets.is_contiguous()
        if rand_nums == None:
            rand_nums = torch.rand(k).cuda()
        indices = torch.arange((N)).float().cuda()
        # print('before', indices)
        rand_k_indices = torch.cuda.FloatTensor(nProposal, k).zero_().float()
        # print('before offsets',offsets)
        # print('before out',out)
        ext_PG_OP.sec_rand_k(rand_nums, offsets.cuda(), indices, rand_k_indices, nProposal,k)
        # print('after offsets', offsets)
        # print('after out', out)
        # print('after',indices)
        rand_k_indices = rand_k_indices.long()
        return rand_k_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None

sec_rand_k_idx = SecRandKIdx.apply


class SecTopKIdx(Function):
    @staticmethod
    def forward(ctx, k, scores, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        device = scores.device
        N = scores.size(0)

        assert scores.is_contiguous()
        assert offsets.is_contiguous()



        indices = torch.arange((N)).float().to(device)
        # print('before', indices)
        top_k_indices = torch.cuda.FloatTensor(nProposal, k).zero_().float().to(device)


        # print('before offsets',offsets)
        # print('before out',out)

        ext_PG_OP.sec_top_k(-scores, offsets, indices, top_k_indices, nProposal,k)
        # print('after offsets', offsets)
        # print('after out', out)
        # print('after',indices)

        top_k_indices = top_k_indices.long()
        return top_k_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None

sec_top_k_idx = SecTopKIdx.apply


class SecFirstK(Function):
    @staticmethod
    def forward(ctx, k, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)
        N = inp.size(0)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()


        out = torch.cuda.FloatTensor(nProposal, k).zero_()


        # print('before offsets',offsets)
        # print('before out',out)
        ext_PG_OP.sec_first_k(inp, offsets, out, nProposal, C,k)
        # print('after offsets', offsets)
        # print('after out', out)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None

sec_first_k = SecFirstK.apply


class SecArgsort(Function):
    @staticmethod
    def forward(ctx, inp, offsets,descend=False):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)
        N = inp.size(0)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        if descend:
            inp_clone = -inp
        else:
            inp_clone = inp.clone()
        # out = torch.cuda.FloatTensor(nProposal, C).zero_()
        out = torch.arange((N)).float().to(inp.device) # indices

        # print('before offsets',offsets)
        # print('before out',out)
        ext_PG_OP.sec_argsort(inp_clone, offsets, out, nProposal, C)
        # print('after offsets', offsets)
        # print('after out', out)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None

sec_argsort = SecArgsort.apply

class SecWeightedMean(Function):
    @staticmethod
    def forward(ctx, inp,weights, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)
        N = inp.size(0)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()
        assert weights.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()
        # out = torch.arange((N)).float().to(inp.device)

        ext_PG_OP.sec_weighted_mean(inp, weights, weights.sum(),offsets, out, nProposal, C)
        # print(offsets)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

sec_weighted_mean = SecWeightedMean.apply

