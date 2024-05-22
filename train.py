import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import open3d as o3d

import torch
import torch.optim as optim
import torch.nn as nn  # added
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np

from util.config import cfg

if hasattr(cfg, 'gpu_ids'):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

from util.log import logger
import util.utils as utils


def init():
    print('init: copying back up files')
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    vis_dir = os.path.join(cfg.exp_path, 'vis')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))
    os.system('cp {} {}'.format(cfg.model_pipeline_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.model_loss_dir, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    iteration_num_per_epoch = len(train_loader)

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        if hasattr(cfg, 'vis_step'):
            vis_now = (i + 1) % cfg.vis_step == 0
        else:
            vis_now = False
        # print('vis_now',vis_now)
        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        if vis_now:
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch, vis=vis_now)
        else:
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)
        # print('-'*100)
        # for k,v in preds.items():
        #     print(k)
        # print('-' * 100)


        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        if loss.grad_fn is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # #####  visualize intermediate results.
        if vis_now:

            voxels_out = preds['voxels_out']
            # sample_ids = np.random.choice(voxels_out.shape[0], 3, replace=False) if voxels_out.shape[0] >= 3 else range(
            #     voxels_out.shape[0])

            # n_shapes_per_batch = self.cfg.config['data']['completion_limit_in_train'] #
            for idx in range(3):
                # print(idx)
                ### pred voxel
                voxel_path = os.path.join(cfg.exp_path, 'vis', '%d_%d_shape%d.png' % (epoch, i, idx))
                utils.visualize_voxels(voxels_out[idx].cpu().numpy(), voxel_path)
                ### gt voxel
                # batch_index = i // n_shapes_per_batch
                # in_batch_id = shape_id % n_shapes_per_batch
                # box_id = proposal_to_gt_box_w_cls_list[batch_index][in_batch_id][1].item()
                # cls_id = proposal_to_gt_box_w_cls_list[batch_index][in_batch_id][2].item()
                #
                # voxels_gt = data['object_voxels'][batch_index][box_id].cpu().numpy()
                # voxel_path = os.path.join(self.cfg.config['log']['vis_path'],
                #                           '%s_%s_%s_%03d_gt_cls%d.png' % (epoch, phase, iter, idx, cls_id))
                # vis.visualize_voxels(voxels_gt, voxel_path)

        ##### print step
        print_step = 3
        if hasattr(cfg, 'print_step'):
            print_step = cfg.print_step

        # Phase 1:
        if epoch <= cfg.prepare_epochs and not cfg.use_gt_seg_and_bbox:
            print_str = \
                "{}/{}, {}/{} | L: {:.4f} lr:{:.6f} | sem: {:.4f} off: {:.4f}/{:.4f} \nbbox_loss:{:.4f} bbox_iou:{:.4f} ".format \
                    (epoch, cfg.epochs,
                     i + 1, iteration_num_per_epoch,
                     am_dict['loss'].avg,
                     optimizer.param_groups[0]['lr'],
                     am_dict['semantic_loss'].avg,
                     am_dict['offset_norm_loss'].avg,
                     am_dict['offset_dir_loss'].avg,
                     am_dict['bbox_loss'].avg,
                     am_dict['bbox_iou'].avg,)
            if cfg.use_bbox_reg_loss:
                print_str += \
                    "| bbox_reg_loss:{:.4f} \n".format \
                            (
                            am_dict['bbox_reg_loss'].avg
                        )

            if cfg.angle_parameter != 'bin':
                print_str += \
                "angle_flip_loss:{:.4f} | angle_flip_acc:{:.4f}| T: {remain_time}\n\n".format \
                    (
                     am_dict['angle_flip_loss'].avg,
                     am_dict['angle_flip_accuracy'].avg,
                     remain_time=remain_time)
            else:
                print_str += \
                    "angle_label_loss:{:.4f} | angle_residual_loss:{:.4f} | T: {remain_time}\n\n".format \
                            (
                            am_dict['angle_label_loss'].avg,
                            am_dict['angle_residual_loss'].avg,
                            remain_time=remain_time)

        # Phase 1.5
        elif epoch <= cfg.prepare_epochs_2 and not cfg.use_gt_seg_and_bbox:
            print_str = \
                "{}/{}, {}/{} | L: {:.4f} lr:{:.6f} | sem: {:.4f} off: {:.4f}/{:.4f} \nbbox_loss:{:.4f} bbox_iou:{:.4f}  ".format \
                    (epoch, cfg.epochs,
                     i + 1, iteration_num_per_epoch,
                     am_dict['loss'].avg,
                     optimizer.param_groups[0]['lr'],
                     am_dict['semantic_loss'].avg,
                     am_dict['offset_norm_loss'].avg,
                     am_dict['offset_dir_loss'].avg,
                     am_dict['bbox_loss'].avg,
                     am_dict['bbox_iou'].avg, )
            if cfg.use_bbox_reg_loss:
                print_str += \
                    "| bbox_reg_loss:{:.4f} \n".format \
                            (
                            am_dict['bbox_reg_loss'].avg
                        )
            if cfg.angle_parameter != 'bin':
                print_str += \
                    "angle_flip_loss:{:.4f} | angle_flip_acc:{:.4f}| T: {remain_time}\n\n".format \
                            (
                            am_dict['angle_flip_loss'].avg,
                            am_dict['angle_flip_accuracy'].avg,
                            remain_time=remain_time)
            else:
                print_str += \
                    "angle_label_loss:{:.4f} | angle_residual_loss:{:.4f} | T: {remain_time}\n\n".format \
                            (
                            am_dict['angle_label_loss'].avg,
                            am_dict['angle_residual_loss'].avg,
                            remain_time=remain_time)

        # Phase 2
        else:
            print_str = ''
            if not cfg.use_gt_seg_and_bbox:
                print_str += \
                    "{}/{}, {}/{} | L: {:.4f} lr:{:.6f} | sem: {:.4f} off: {:.4f}/{:.4f} \nbbox_loss:{:.4f} bbox_iou:{:.4f} | ".format \
                        (epoch, cfg.epochs,
                         i + 1, iteration_num_per_epoch,
                         am_dict['loss'].avg,
                         optimizer.param_groups[0]['lr'],
                         am_dict['semantic_loss'].avg,
                         am_dict['offset_norm_loss'].avg,
                         am_dict['offset_dir_loss'].avg,
                         am_dict['bbox_loss'].avg,
                         am_dict['bbox_iou'].avg, )
                if cfg.use_bbox_reg_loss:
                    print_str += \
                        "| bbox_reg_loss:{:.4f} \n".format \
                                (
                                am_dict['bbox_reg_loss'].avg
                            )
                if cfg.angle_parameter != 'bin':
                    print_str += \
                        "angle_flip_loss:{:.4f}| angle_flip_acc:{:.4f} \n".format \
                                (
                                am_dict['angle_flip_loss'].avg,
                                am_dict['angle_flip_accuracy'].avg,
                                remain_time=remain_time)
                else:
                    print_str += \
                        "angle_label_loss:{:.4f}| angle_residual_loss:{:.4f}\n".format \
                                (
                                am_dict['angle_label_loss'].avg,
                                am_dict['angle_residual_loss'].avg,
                                remain_time=remain_time)


                print_str += 'seg_score: {:.4f}| cls_score:{:.4f}| bbox_score:{:.4f}|mesh_score:{:.4f} \n'.format \
                        (

                        am_dict['instance_bbox_score_loss'].avg,
                        am_dict['instance_score_loss'].avg,
                        am_dict['instance_semantic_loss'].avg,
                        am_dict['instance_mesh_score_loss'].avg,

                    )

            print_str += 'completion: {:.4f}| z: {:.4f} | mesh: {:.4f} |mesh_acc:{:.4f} T: {remain_time}\n\n'.format \
                            (
                            am_dict['completion_loss'].avg,
                            am_dict['z_loss'].avg,
                            am_dict['mesh_loss'].avg,
                            am_dict['mesh_accuracy'].avg,

                            remain_time=remain_time
                            )

            # sys.stdout.write(
            # if cfg.use_gt_seg_and_bbox:
            #     print_str = \
            #         "{}/{}, {}/{} | L: {:.4f} lr:{:.6f} |  completion: {:.4f}, z: {:.4f}  mesh: {:.4f}| T: {remain_time}".format \
            #             (epoch, cfg.epochs,
            #              i + 1, iteration_num_per_epoch,
            #              am_dict['loss'].avg,
            #              optimizer.param_groups[0]['lr'],
            #              am_dict['completion_loss'].avg,
            #              am_dict['z_loss'].avg,
            #              am_dict['mesh_loss'].avg,
            #              remain_time=remain_time)

        if (i + 1) % print_step == 0:
            logger.info(print_str)  # save in log file
            '''log in tensorboard'''
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_train', am_dict[k].avg, epoch * iteration_num_per_epoch + i)
        else:
            print(print_str)

        if (i == len(train_loader) - 1): print()
    else:
        print('no CAD clusters, skip this iteration')


    # out of iteration for loop
    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                     time.time() - start_epoch))

    utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)

    # '''log in tensorboard'''
    # for k in am_dict.keys():
    #     if k in visual_dict.keys():
    #         writer.add_scalar(k+'_train', am_dict[k].avg, epoch)


def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val,
                                                                         am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                       time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


if __name__ == '__main__':

    ##### init

    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')

    if not hasattr(cfg, 'network') or cfg.network == 'rfs':
        from model.rfs import RfSNet as Network
        from model.rfs import model_fn_decorator
    elif cfg.network == 'network':
        from model.network import Network
        from model.model_fn_decorator import model_fn_decorator

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    # # gpus
    # if hasattr(cfg, 'gpu_ids'):
    #     logger.info('GPU Ids: %s used.' % cfg.gpu_ids)
    device = torch.device("cuda")
    model = model.cuda()
    # model = nn.DataParallel(model).to(device) #model = model.cuda() # cudnn error
    # model = nn.DataParallel(model,device_ids=[0]) #model = model.cuda() # give device_ids or cudnn error

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator()

    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst

            dataset = data.scannetv2_inst.Dataset()
            dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    ##### resume
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5],
                                           use_cuda)  # resume from the latest epoch, or specify the epoch to restore

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch):
            eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
