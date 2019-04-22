# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

from __future__ import print_function, division
import os
import sys
import logging
import pprint
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from upsnet.config.config import config
from upsnet.config.parse_args import parse_args
from lib.utils.logging import create_logger
from lib.utils.timer import Timer


args = parse_args()
logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

from upsnet.dataset import *
from upsnet.models import *
from upsnet.bbox.bbox_transform import bbox_transform, clip_boxes, expand_boxes
from lib.utils.callback import Speedometer
from lib.utils.data_parallel import DataParallel
from pycocotools.mask import encode as mask_encode

cv2.ocl.setUseOpenCL(False)

cudnn.enabled = True
cudnn.benchmark = False

def im_detect(output_all, data, im_infos):

    scores_all = []
    pred_boxes_all = []
    pred_masks_all = []
    pred_ssegs_all = []
    pred_panos_all = []
    pred_pano_cls_inds_all = []
    cls_inds_all = []

    if len(data) == 1:
        output_all = [output_all]

    output_all = [{k: v.data.cpu().numpy() for k, v in output.items()} for output in output_all]

    for i in range(len(data)):
        im_info = im_infos[i]
        scores_all.append(output_all[i]['cls_probs'])
        pred_boxes_all.append(output_all[i]['pred_boxes'][:, 1:] / im_info[2])
        cls_inds_all.append(output_all[i]['cls_inds'])

        if config.network.has_mask_head:
            pred_masks_all.append(output_all[i]['mask_probs'])
        if config.network.has_fcn_head:
            pred_ssegs_all.append(output_all[i]['fcn_outputs'])
        if config.network.has_panoptic_head:
            pred_panos_all.append(output_all[i]['panoptic_outputs'])
            pred_pano_cls_inds_all.append(output_all[i]['panoptic_cls_inds'])

    return {
        'scores': scores_all,
        'boxes': pred_boxes_all,
        'masks': pred_masks_all,
        'ssegs': pred_ssegs_all,
        'panos': pred_panos_all,
        'cls_inds': cls_inds_all,
        'pano_cls_inds': pred_pano_cls_inds_all,
    }


def im_post(boxes_all, masks_all, scores, pred_boxes, pred_masks, cls_inds, num_classes, im_info):

    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0

    M = config.network.mask_size

    scale = (M + 2.0) / M


    ref_boxes = expand_boxes(pred_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    for idx in range(1, num_classes):
        segms = []
        cls_boxes = np.hstack([pred_boxes[idx == cls_inds, :], scores.reshape(-1, 1)[idx == cls_inds]])
        cls_pred_masks = pred_masks[idx == cls_inds]
        cls_ref_boxes = ref_boxes[idx == cls_inds]
        for _ in range(cls_boxes.shape[0]):

            if pred_masks.shape[1] > 1:
                padded_mask[1:-1, 1:-1] = cls_pred_masks[_, idx, :, :]
            else:
                padded_mask[1:-1, 1:-1] = cls_pred_masks[_, 0, :, :]
            ref_box = cls_ref_boxes[_, :]

            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)
            im_mask = np.zeros((im_info[0], im_info[1]), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_info[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_info[0])

            im_mask[y_0:y_1, x_0:x_1] = mask[
                                        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                        (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                        ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            rle['counts'] = rle['counts'].decode()
            segms.append(rle)

            mask_ind += 1

        cls_segms[idx] = segms
        boxes_all[idx].append(cls_boxes)
        masks_all[idx].append(segms)

class PGDAttack(object):
  def __init__(self, loss_fn=None, num_steps=10, step_size=0.01, epsilon=0.1):
    """
    Attack a network by Project Gradient Descent. The attacker performs
    k steps of gradient descent of step size a, while always staying
    within the range of epsilon from the input image.

    Args:
      loss_fn: loss function used for the attack
      num_steps: (int) number of steps for PGD
      step_size: (float) step size of PGD
      epsilon: (float) the range of acceptable samples
               for our normalization, 0.1 ~ 6 pixel levels
    """
    self.loss_fn = loss_fn
    self.num_steps = num_steps
    self.step_size = step_size
    self.epsilon = epsilon

  def perturb(self, model, input):
    """
    Given input image X (torch tensor), return an adversarial sample
    (torch tensor) using PGD of the least confident label.

    See https://openreview.net/pdf?id=rJzIBfZAb

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) an adversarial sample of the given network
    """
    # clone the input tensor and disable the gradients
    #output = input[0]['data'].clone()
    #input[0]['data'].requires_grad = False
    input[0]['data'].requires_grad = True
    input[0]['data_4x'].requires_grad = True

    #output.requires_grad = True
    inputori = input[0]['data'].clone()


    #net = model(inputori) # [To Check] what's the output format? the same as hw2?
    net = model(input) # [To Check] what's the output format? the same as hw2?

    #print('\n\n\n\n\nnet = {}\n\n\n\n\n'.format(net))
    '''print('\n\n\n\n\n')
    for k, v in net.items():
        print('net_key = {}'.format(k))
        if k == 'cls_probs':
            print('cls_probs = {}'.format(v))
    print('\n\n\n\n\n')'''

    print('\n\n\n\n\n')
    for k, v in net.items():
        print('net_key = {}'.format(k))
        if not(k == 'mask_probs'):
            print('net_val = {}\n\n\n\n'.format(v))
        else:
            print('net_val.size = {}\n\n\n\n'.format(len(v)))

    print('\n\n\n\n\n')



    #pred = torch.min(net.data, 1)[1]

    #[To do] merge pred to input

    # loop over the number of steps
    for _ in range(self.num_steps):
      #################################################################################
      # Fill in the code here
      #################################################################################
      #net = model(output, pred)
      #net = model(input, pred)
      net = model(input)



      #loss = self.loss_fn(net, pred)
      loss = 0
      if config.network.has_rpn:
          loss = loss + net['rpn_cls_loss'].mean() + net['rpn_bbox_loss'].mean()
      if config.network.has_rcnn:
          loss = loss + net['cls_loss'].mean() + net['bbox_loss'].mean()
      if config.network.has_mask_head:
          loss = loss + net['mask_loss'].mean()
      if config.network.has_fcn_head:
          loss = loss + net['fcn_loss'].mean() * config.train.fcn_loss_weight
          if config.train.fcn_with_roi_loss:
              loss = loss + net['fcn_roi_loss'].mean() * config.train.fcn_loss_weight * 0.2
      if config.network.has_panoptic_head:
          loss = loss + net['panoptic_loss'].mean() * config.train.panoptic_loss_weight
      loss.backward()



      print("input[0]['data'].grad = {}".format(input[0]['data'].grad))
      temp = input[0]['data'] - self.step_size * torch.sign(input[0]['data'].grad) - inputori
      temp = torch.clamp(temp, min = -self.epsilon, max=self.epsilon)
      input[0]['data'] = temp + inputori
      #output = input.clone()
      #output = torch.tensor(output.data, requires_grad=True)




    #input[0]['data'] = output
    #return output
    return input

default_attack = PGDAttack

def upsnet_test():

    pprint.pprint(config)
    logger.info('test config:{}\n'.format(pprint.pformat(config)))

    # create models
    gpus = [int(_) for _ in config.gpus.split(',')]
    test_model = eval(config.symbol)().cuda(device=gpus[0])

    # create data loader
    test_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False,
                                                result_path=final_output_path)#, phase='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False,
                                              num_workers=0, drop_last=False, pin_memory=False, collate_fn=test_dataset.collate)

    #num_gpus = 1 if config.train.use_horovod else len(gpus)
    #test_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.image_set.split('+'), flip=config.train.flip, result_path=final_output_path)
    #val_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False, result_path=final_output_path, phase='val')
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=num_gpus * 4 if not config.debug_mode else num_gpus * 4, drop_last=False, collate_fn=test_dataset.collate)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=num_gpus * 4 if not config.debug_mode else num_gpus * 4, drop_last=False, collate_fn=val_dataset.collate)



    if args.eval_only:
        results = pickle.load(open(os.path.join(final_output_path, 'results', 'results_list.pkl'), 'rb'))
        if config.test.vis_mask:
            test_dataset.vis_all_mask(results['all_boxes'], results['all_masks'], os.path.join(final_output_path, 'results', 'vis'))
        if config.network.has_rcnn:
            test_dataset.evaluate_boxes(results['all_boxes'], os.path.join(final_output_path, 'results'))
        if config.network.has_mask_head:
            test_dataset.evaluate_masks(results['all_boxes'], results['all_masks'], os.path.join(final_output_path, 'results'))
        if config.network.has_fcn_head:
            test_dataset.evaluate_ssegs(results['all_ssegs'], os.path.join(final_output_path, 'results', 'ssegs'))
            # logging.info('combined pano result:')
            # test_dataset.evaluate_panoptic(test_dataset.get_combined_pan_result(results['all_ssegs'], results['all_boxes'], results['all_masks'], stuff_area_limit=config.test.panoptic_stuff_area_limit), os.path.join(final_output_path, 'results', 'pans_combined'))
        if config.network.has_panoptic_head:
            logging.info('unified pano result:')
            test_dataset.evaluate_panoptic(test_dataset.get_unified_pan_result(results['all_ssegs'], results['all_panos'], results['all_pano_cls_inds'], stuff_area_limit=config.test.panoptic_stuff_area_limit), os.path.join(final_output_path, 'results', 'pans_unified'))
        sys.exit()

    # preparing
    curr_iter = config.test.test_iteration
    test_model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0]),
                               '_'.join(config.dataset.image_set.split('+')), config.model_prefix+str(curr_iter)+'.pth'))), resume=True)

    print(os.path.join(os.path.join(os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0]),
                                       '_'.join(config.dataset.image_set.split('+')), config.model_prefix+str(curr_iter)+'.pth')))

    #for p in test_model.parameters():
    #    p.requires_grad = False

    test_model = DataParallel(test_model, device_ids=gpus, gather_output=False).to(gpus[0])

    # start training
    test_model.eval()

    i_iter = 0
    idx = 0
    test_iter = test_loader.__iter__()
    all_boxes = [[] for _ in range(test_dataset.num_classes)]
    if config.network.has_mask_head:
        all_masks = [[] for _ in range(test_dataset.num_classes)]
    if config.network.has_fcn_head:
        all_ssegs = []
    if config.network.has_panoptic_head:
        all_panos = []
        all_pano_cls_inds = []
        panos = []


    data_timer = Timer()
    net_timer = Timer()
    post_timer = Timer()
    attacker=default_attack()

    while i_iter < len(test_loader):
        data_timer.tic()
        batch = []
        labels = []
        for gpu_id in gpus:
            try:
                data, label, _ = test_iter.next()

                print('\n\n\n\n\n')
                #print('label = {}\n\n\n\n\n\n\n\n\n'.format(label))
                for k, v in label.items():
                    print('label_key = {}\n'.format(k))

                    if k == 'roidb':
                        for k2, v2 in v.items():
                            print('label_val_key = {}'.format(k2))

                            if not(k2 == 'segms'):
                                print('label_val_val = {}\n\n\n\n'.format(v2))

                            if k2 == 'segms':
                                print('segms.val.size = {}\n\n\n\n'.format(len(v2)))
                    else:
                        print('label_val = {}\n\n\n\n'.format(v))

                print('\n\n\n\n\n')
                print('=================================================================')

                print('\n\n\n\n\n')
                #print('label = {}\n\n\n\n\n\n\n\n\n'.format(label))
                for k, v in data.items():
                    print('data_key = {}'.format(k))
                    print('data_val = {}\n\n\n\n'.format(v))
                print('\n\n\n\n\n')
                print('=================================================================')

                if label is not None:
                    data['roidb'] = label['roidb']
                for k, v in data.items():
                    data[k] = v.pin_memory().to(gpu_id, non_blocking=True) if torch.is_tensor(v) else v



                if attacker is not None:
                    for k, v in label.items():
                        label[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)


            except StopIteration:
                data = data.copy()
                for k, v in data.items():
                    data[k] = v.pin_memory().to(gpu_id, non_blocking=True) if torch.is_tensor(v) else v
            



            if attacker is not None:
                batch.append((data, label))
            else:
                batch.append((data, None))




            labels.append(label)
            i_iter += 1

        im_infos = [_[0]['im_info'][0] for _ in batch]

        data_time = data_timer.toc()
        if i_iter > 10:
            net_timer.tic()



        #with torch.no_grad():

        #print("data['data'] = {}".format(data['data']))
        #print("data['data'].size() = {}".format(data['data'].size()))
        #print(data['data_4x'].size())
        #print(batch)
        
        # generate adversarial samples
        if attacker is not None:
            print("attacking: {}".format(i_iter))
            # generate adversarial samples
            adv_input = attacker.perturb(test_model, *batch)
            # forward the model
            output = test_model(adv_input)
        else:
            # forward the model
            output = test_model(*batch)



        torch.cuda.synchronize()
        if i_iter > 10:
            net_time = net_timer.toc()
        else:
            net_time = 0
        output = im_detect(output, batch, im_infos)




        post_timer.tic()
        for score, box, mask, cls_idx, im_info in zip(output['scores'], output['boxes'], output['masks'], output['cls_inds'], im_infos):
            im_post(all_boxes, all_masks, score, box, mask, cls_idx, test_dataset.num_classes, np.round(im_info[:2] / im_info[2]).astype(np.int32))
            idx += 1
        if config.network.has_fcn_head:
            for i, sseg in enumerate(output['ssegs']):
                sseg = sseg.squeeze(0).astype('uint8')[:int(im_infos[i][0]), :int(im_infos[i][1])]
                all_ssegs.append(cv2.resize(sseg, None, None, fx=1/im_infos[i][2], fy=1/im_infos[i][2], interpolation=cv2.INTER_NEAREST))
        if config.network.has_panoptic_head:
            pano_cls_inds = []
            for i, (pano, cls_ind) in enumerate(zip(output['panos'], output['pano_cls_inds'])):
                pano = pano.squeeze(0).astype('uint8')[:int(im_infos[i][0]), :int(im_infos[i][1])]
                panos.append(cv2.resize(pano, None, None, fx=1/im_infos[i][2], fy=1/im_infos[i][2], interpolation=cv2.INTER_NEAREST))
                pano_cls_inds.append(cls_ind)

            all_panos.extend(panos)
            panos = []
            all_pano_cls_inds.extend(pano_cls_inds)
        post_time = post_timer.toc()
        s = 'Batch %d/%d, data_time:%.3f, net_time:%.3f, post_time:%.3f' % (idx, len(test_dataset), data_time, net_time, post_time)
        logging.info(s)

    results = []

    # trim redundant predictions
    for i in range(1, test_dataset.num_classes):
        all_boxes[i] = all_boxes[i][:len(test_loader)]
        if config.network.has_mask_head:
            all_masks[i] = all_masks[i][:len(test_loader)]
    if config.network.has_fcn_head:
        all_ssegs = all_ssegs[:len(test_loader)]
    if config.network.has_panoptic_head:
        all_panos = all_panos[:len(test_loader)]

    os.makedirs(os.path.join(final_output_path, 'results'), exist_ok=True)

    results = {'all_boxes': all_boxes,
               'all_masks': all_masks if config.network.has_mask_head else None,
               'all_ssegs': all_ssegs if config.network.has_fcn_head else None,
               'all_panos': all_panos if config.network.has_panoptic_head else None,
               'all_pano_cls_inds': all_pano_cls_inds if config.network.has_panoptic_head else None,
               }

    with open(os.path.join(final_output_path, 'results', 'results_list.pkl'), 'wb') as f:
        pickle.dump(results, f, protocol=2)

    if config.test.vis_mask:
        test_dataset.vis_all_mask(all_boxes, all_masks, os.path.join(final_output_path, 'results', 'vis'))
    else:
        ''' 
        test_dataset.evaluate_boxes(all_boxes, os.path.join(final_output_path, 'results'))
        if config.network.has_mask_head:
            test_dataset.evaluate_masks(all_boxes, all_masks, os.path.join(final_output_path, 'results'))
        '''
        if config.network.has_panoptic_head:
            logging.info('unified pano result:')
            test_dataset.evaluate_panoptic(test_dataset.get_unified_pan_result(all_ssegs, all_panos, all_pano_cls_inds, stuff_area_limit=config.test.panoptic_stuff_area_limit), os.path.join(final_output_path, 'results', 'pans_unified'))
        '''
        if config.network.has_fcn_head:
            test_dataset.evaluate_ssegs(all_ssegs, os.path.join(final_output_path, 'results', 'ssegs'))
            # logging.info('combined pano result:')
            # test_dataset.evaluate_panoptic(test_dataset.get_combined_pan_result(all_ssegs, all_boxes, all_masks, stuff_area_limit=config.test.panoptic_stuff_area_limit), os.path.join(final_output_path, 'results', 'pans_combined'))
        '''

if __name__ == "__main__":
    upsnet_test()
