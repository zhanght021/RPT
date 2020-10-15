from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
the code is modified from pysot(https://github.com/STVIR/pysot) 
and RepPoints(https://github.com/microsoft/RepPoints), Thanks.
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from siamreppoints.core.xcorr import xcorr_fast, xcorr_depthwise

from siamreppoints.models.layers import DeformConv

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.gradient_mul = 0.1
        self.dcn_kernel = 3
        self.dcn_pad = 1
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor([dcn_base_offset]).view(1, -1, 1, 1)
        
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.cls_conv3x3 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.pts_conv3x3 = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
                
        self.pts_out_dim = 2 * 9
        self.reppoints_cls_conv = DeformConv(256, 256, self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_bn = nn.BatchNorm2d(256)
        self.reppoints_cls_out = nn.Conv2d(256, 1, 1, 1, 0)
        
        self.reppoints_pts_init_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.reppoints_pts_init_bn = nn.BatchNorm2d(256)
        self.reppoints_pts_init_out = nn.Conv2d(256, self.pts_out_dim, 1, 1, 0)
        
        self.reppoints_pts_refine_conv = DeformConv(256, 256, self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_pts_refine_bn = nn.BatchNorm2d(256)
        self.reppoints_pts_refine_out = nn.Conv2d(256, self.pts_out_dim, 1, 1, 0)
        
    def forward(self, kernel, search):
        dcn_base_offset = self.dcn_base_offset.type_as(kernel)
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)

        cls_feat = self.cls_conv3x3(feature)  
        pts_feat = self.pts_conv3x3(feature)
        
        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_bn(self.reppoints_pts_init_conv(pts_feat))))   
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        
        cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_bn(self.reppoints_cls_conv(cls_feat, dcn_offset))))
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_bn(self.reppoints_pts_refine_conv(pts_feat, dcn_offset))))
        pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine

        
class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.normalCorr = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        
    def forward(self, z_f, x_f):
        cls_out, pts_out_init, pts_out_refine = self.normalCorr(z_f, x_f)
        return cls_out, pts_out_init, pts_out_refine


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.pts_init_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.pts_refine_weight = nn.Parameter(torch.ones(len(in_channels)))
        
        self.moment_transfer = nn.Parameter(torch.zeros(2))
        self.moment_mul = 0.1
        self.exemplar_size = 127
        
    def get_xy_ctr(self, score_size, score_offset, total_stride):
        batch, fm_height, fm_width = 1, score_size, score_size

        y_list = torch.linspace(0., fm_height - 1., fm_height).reshape(
            1, fm_height, 1, 1).repeat(1, 1, fm_width,
                                       1)  # .broadcast([1, fm_height, fm_width, 1])
        x_list = torch.linspace(0., fm_width - 1., fm_width).reshape(
            1, 1, fm_width, 1).repeat(1, fm_height, 1,
                                      1)  # .broadcast([1, fm_height, fm_width, 1])
        xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
        xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(
            batch, -1,
            2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
        xy_ctr = xy_ctr.type(torch.Tensor)
        return xy_ctr
    
    def offset_to_pts(self, location, pred_list):
        pts_lvl = []
        for i_img in range(pred_list.shape[0]):
            pts_center = location.repeat(1, 9)
            yx_pts_shift = pred_list[i_img].permute(1, 2, 0).reshape(-1, 2*9)
            y_pts_shift = yx_pts_shift[..., 0::2]
            x_pts_shift = yx_pts_shift[..., 1::2]
            xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
            xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
            pts = xy_pts_shift * 8 + pts_center
            pts_lvl.append(pts)
        pts_coordinate_preds = torch.stack(pts_lvl, 0)
        return pts_coordinate_preds
    
    def points2bbox(self, pts, y_first=True):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        
        pts_y_mean = pts_y.mean(dim=1, keepdim=True)
        pts_x_mean = pts_x.mean(dim=1, keepdim=True)
        pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
        pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
        moment_transfer = (self.moment_transfer * self.moment_mul) + (
            self.moment_transfer.detach() * (1 - self.moment_mul))
        moment_width_transfer = moment_transfer[0]
        moment_height_transfer = moment_transfer[1]
        half_width = pts_x_std * torch.exp(moment_width_transfer)
        half_height = pts_y_std * torch.exp(moment_height_transfer)
        bbox = torch.cat([
            pts_x_mean - half_width, pts_y_mean - half_height,
            pts_x_mean + half_width, pts_y_mean + half_height
        ],
                         dim=1)
        
        # bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        # bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        # bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        # bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        # bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                        # dim=1)
        
        return bbox
    
    def forward(self, z_fs, x_fs, instance_size):
        score_size = (instance_size - self.exemplar_size) // 8 + 1 + 8
        score_offset = (instance_size - 1 - (score_size - 1) * 8) // 2
        self.location = self.get_xy_ctr(score_size, score_offset, 8).cuda().squeeze(0)
        
        cls = []
        pts_init = []
        pts_refine = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l, s = rpn(z_f, x_f)
            cls.append(c)
            pts_init.append(l)
            pts_refine.append(s)
            
        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            pts_init_weight = F.softmax(self.pts_init_weight, 0)
            pts_refine_weight = F.softmax(self.pts_refine_weight, 0)
            
        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s
        
        if self.weighted:
            cls = weighted_avg(cls, cls_weight)
            pts_init = weighted_avg(pts_init, pts_init_weight)
            pts_refine = weighted_avg(pts_refine, pts_refine_weight)
        else:
            cls = avg(cls)
            pts_init = avg(pts_init)
            pts_refine = avg(pts_refine)

        pts_coordinate_preds_init = self.offset_to_pts(self.location, pts_init)

        pts_preds_init = self.points2bbox(pts_coordinate_preds_init.reshape(-1, 18), y_first=False)
        pts_preds_init = pts_preds_init.reshape(cls.shape[0], -1, 4)
        
        pts_coordinate_preds_refine = self.offset_to_pts(self.location, pts_refine)
        pts_preds_refine = self.points2bbox(pts_coordinate_preds_refine.reshape(-1, 18), y_first=False)
        pts_preds_refine = pts_preds_refine.reshape(cls.shape[0], -1, 4)
        return cls, pts_preds_init, pts_preds_refine
        