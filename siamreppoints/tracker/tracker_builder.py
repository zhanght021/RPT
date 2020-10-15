# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamreppoints.core.config import cfg
from siamreppoints.tracker.siamreppoints_tracker import SiamReppointsTracker


TRACKS = {
          'SiamReppointsTracker': SiamReppointsTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
