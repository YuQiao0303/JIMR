import time
import argparse
import numpy as np
import torch
from pytorch import ext_pointgroup_ops


# real_data 1
# offset = torch.load('offsets.torch').int() # the key is this .int(). Don't ever forget it
# scores_1 = torch.load('scores.torch')

# # real_data 2
# offset = torch.load('offsets_torch2').int() # the key is this .int(). Don't ever forget it
# scores_1 = torch.load('scores_torch2')
# bboxes = torch.load('bbox_torch2')

# # real_data 3
# offset = torch.load('offsets_torch3').int() # the key is this .int(). Don't ever forget it
# scores_1 = torch.load('scores_torch3')

# # real_data 4
# offset = torch.load('offsets_torch4').int() # the key is this .int(). Don't ever forget it
# scores_1 = torch.load('scores_torch4')

# # real_data 5
# offset = torch.load('offsets_torch5').int() # the key is this .int(). Don't ever forget it
# scores_1 = torch.load('scores_torch5')
# bboxes = torch.load('bbox_torch5')

# real_data 6
offset = torch.load('offsets_torch6').int() # the key is this .int(). Don't ever forget it
scores_1 = torch.load('scores_torch6')
bboxes = torch.load('bbox_torch6')

# cluster_point_num = offset[1:]-offset[:-1]
# print(cluster_point_num.max())
# print(cluster_point_num.min())
# print(offset.shape)
# 魔改
# cut_n = 15
# # offset[1] = 6
# # offset[2] = 15
# offset = offset[0:cut_n+1]
# scores_1 = scores_1[:offset[cut_n]]
# bboxes = bboxes[:offset[cut_n]]

# same test data
# scores_1 = torch.tensor([0.9947, 0.9982, 0.9972, 0.9981, 0.9979, 0.9969, 0.9982, 0.9979, 0.9960,\
#         0.9982, 0.9981, 0.9981, 0.9981, 0.9981, 0.9980], device='cuda:0')
#
# scores_1 = torch.rand((15)).float().cuda()
# offset = torch.tensor([ 0,  6, 15], device='cuda:0').int()


# test_data
# scores_1 = torch.rand((100)).float().cuda()
# offset = torch.tensor([0, 3, 30, 60, 100]).cuda().int()
# bboxes = torch.rand((100,7)).float().cuda()
# print('bboxes[:3]',bboxes[:3])
# print('scores_1[:3]',scores_1[:3])


n_Points = scores_1.shape[0]
scores = scores_1.unsqueeze(1)
n_Proposals = offset.shape[0]-1
ntest = 1
k = 2048

rand_nums = torch.rand(k).cuda()
def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for i in range(ntest):
        res = func()

    for i in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

def run_weighted_mean_cuda():
    print('bboxes.shape',bboxes.shape)
    print('scores.shape',scores.shape)
    results = ext_pointgroup_ops.sec_weighted_mean(bboxes,scores,offset)
    # results = ext_pointgroup_ops.sec_weighted_mean(scores,scores,offset)
    return results

def run_weighted_mean_for():
    manual_results = torch.zeros((n_Proposals,7)).cuda()
    for i in range(n_Proposals):
        start = offset[i]
        end = offset[i + 1]


        # manual_results[i] = ((scores_1[start:end] * scores_1[start:end]).sum() / scores_1[start:end].sum())
        manual_results[i] = (bboxes[start:end] * scores_1[start:end].unsqueeze(1).repeat(1,7)).sum(0) / scores_1[start:end].sum()
    return manual_results

def run_argsort_cuda():
    results = ext_pointgroup_ops.sec_argsort(scores,offset,True) #descend =
    return results

def run_argsort_for():
    # print('offset',offset)
    manual_results = []
    for i in range(n_Proposals):
        start = offset[i]
        end = offset[i + 1]
        # sorted_values, sorted_indices = torch.sort(-scores[start:end].squeeze(1) )
        sorted_values, sorted_indices = torch.sort(-scores_1[start:end] )
        # print(i, sorted_indices)
        # print(i, sorted_values)
        # print(i, start,end)

        sorted_indices +=start
        # print(i,)

        # print(i,sorted_values)
        manual_results.append(sorted_indices) #0 values, 1 indices
        # print(i,scores[sorted_indices].squeeze(1))
    manual_results = torch.cat(manual_results,dim=0)

    return manual_results

def run_first_k_cuda():
    results = ext_pointgroup_ops.sec_first_k( 5, scores,offset)
    return results

def run_first_k_for():
    results = torch.zeros((n_Proposals,k))
    for i in range(n_Proposals):
        start = offset[i]
        end = offset[i + 1]
        count = (end-start).item()

        # results[i,:] = scores[start:start+k.fmod(count)].squeeze(1) # if k
        # results[i,:] = scores[start:start+k%count].squeeze(1)
        # results[i,:] = scores[start:start+k].squeeze(1)
        results[i,:] = scores_1[start:start+k]
    return results


def run_top_k_cuda():

    # top_k_indices = ext_pointgroup_ops.sec_top_k_idx( 5, scores.squeeze(1),offset)
    # print('scores',scores_1)
    # print(scores_1.shape)
    # print('offset',offset)
    # print(offset.shape)
    top_k_indices = ext_pointgroup_ops.sec_top_k_idx( k, scores_1,offset)
    # results = scores_1[top_k_indices]
    # results = bboxes[top_k_indices]
    # return results
    return top_k_indices

def run_top_k_for():
    manual_sorted_indices = []
    for i in range(n_Proposals):
        start = offset[i]
        end = offset[i + 1]

        sorted_values, sorted_indices = torch.sort(-scores_1[start:end])
        sorted_indices += start
        # print(i,)
        # print(i,sorted_indices)
        # print(i,sorted_values)
        manual_sorted_indices.append(sorted_indices)  # 0 values, 1 indices

        # print(i, scores[sorted_indices].squeeze(1))
    manual_sorted_indices = torch.cat(manual_sorted_indices, dim=0)

    indices = torch.arange(scores_1.shape[0]).cuda()

    results = torch.zeros((n_Proposals,k))
    # results = torch.zeros((n_Proposals,k,7))
    for i in range(n_Proposals):
        start = offset[i]
        end = offset[i + 1]
        count = (end-start).item()


        # results[i,:] = scores_1[manual_sorted_indices][start:start+k]
        # results[i,:,:] = bboxes[manual_sorted_indices][start:start+k]
        results[i,:] = indices[manual_sorted_indices][start:start+k]

    return results
    # return manual_sorted_indices


def run_rand_k_cuda():
    # top_k_indices = ext_pointgroup_ops.sec_top_k_idx( 5, scores.squeeze(1),offset)
    # print('scores',scores_1)
    # print(scores_1.shape)
    # print('offset',offset)
    # print(offset.shape)
    rand_k_indices = ext_pointgroup_ops.sec_rand_k_idx( k,offset,rand_nums)
    # results = scores_1[top_k_indices]
    # results = bboxes[top_k_indices]
    # return results
    return rand_k_indices

def run_rand_k_for():
    manual_sorted_indices = []
    # rand_nums = torch.rand(k).cuda()
    for i in range(n_Proposals):
        start = offset[i]
        end = offset[i + 1]
        count = end-start

        # sorted_indices = torch.round(rand_nums*count).long()
        sorted_indices = (rand_nums*count).long()

        sorted_indices += start
        # print(i,)
        # print(i,sorted_indices)
        # print(i,sorted_values)
        manual_sorted_indices.append(sorted_indices)  # 0 values, 1 indices

        # print(i, scores[sorted_indices].squeeze(1))
    # manual_sorted_indices = torch.cat(manual_sorted_indices, dim=0)
    manual_sorted_indices = torch.vstack(manual_sorted_indices)

    return manual_sorted_indices


def test_compare_cuda_and_for(cuda_func, for_func):
    print("Running cuda...")

    # test weighted mean
    cuda_time, cuda_res = show_time(cuda_func)
    for_time, for_res = show_time(for_func)


    print('cuda rsults', cuda_res)
    print('manual rsults', for_res)

    # print('differnece', cuda_res-for_res)
    print('differnece', (cuda_res-for_res).abs().max())


    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time))) #134.397us
    print("For  time:  {:.3f}us".format(np.mean(for_time)))  #1038.551us
    print("Kernel test passed.")







if __name__ == "__main__":
    # print('offset',offset )
    # print('offset',offset.shape)
    # print('scores_1',scores_1)
    # print('scores_1',scores_1.shape)
    # print('bboxes', bboxes)
    # print('bboxes', bboxes.shape)



    # test_compare_cuda_and_for(run_rand_k_cuda,run_rand_k_for)
    # test_compare_cuda_and_for(run_top_k_cuda,run_top_k_for) # can't run on too much data
    # test_compare_cuda_and_for(run_first_k_cuda,run_first_k_for)
    # test_compare_cuda_and_for(run_argsort_cuda,run_argsort_for) # can't run on too much data
    test_compare_cuda_and_for(run_weighted_mean_cuda,run_weighted_mean_for)

    # result = (0.1428 * 0.6566 + 0.9471 * 0.7805 + 0.3391 *0.3599) / (0.6566 +0.7805 + 0.3599 )
    # print(result)

