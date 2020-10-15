# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from siamreppoints.core.config import cfg
from siamreppoints.models.backbone import get_backbone
from siamreppoints.models.head import get_rpn_head
from siamreppoints.models.neck import get_neck

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)
    
    def instance(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        #self.cf = xf[cfg.ADJUST.LAYER-1]
        self.cf = torch.cat([xf[2], xf[1]], dim=1)
    
    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
    
    def track(self, x, instance_size):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, pts_preds_init, pts_preds_refine = self.rpn_head(self.zf, xf, instance_size)
        
        cls = cls.permute(0, 2, 3, 1)
        cls = cls.reshape(cls.shape[0], -1, 1)
        cls = torch.sigmoid(cls)
        
        #self.cf = xf[cfg.ADJUST.LAYER-1]
        self.cf = torch.cat([xf[2], xf[1]], dim=1)
        return {
                'score': cls,
                'bbox': pts_preds_refine,
               }
    
