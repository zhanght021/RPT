# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamreppoints.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN
       }

def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)

