'''
modified from pysot(https://github.com/STVIR/pysot)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import numpy as np

from siamreppoints.core.config import cfg
from siamreppoints.tracker.base_tracker import SiameseTracker
        
class SiamReppointsTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamReppointsTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.reshape(-1)
        self.model = model
        self.model.eval()

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.frame_num = 0
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        with torch.no_grad():   
            self.model.template(z_crop)
        
        ##control the ratio of window_influence using speed
        ##not important for results
        self.dist = []
        self.dist_x = []
        self.dist_y = []
        self.speed = 0
        self.speed_x = 0
        self.speed_y = 0
             
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        self.frame_num += 1
        
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z) * cfg.TRACK.EXPANSION ##best1.02
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
              
        with torch.no_grad():
            outputs = self.model.track(x_crop, cfg.TRACK.INSTANCE_SIZE)

        scores_siamese = outputs['score'].view(-1).cpu().detach().numpy()
        pred_bbox = outputs['bbox'].cpu().detach().numpy().squeeze(0)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
        
        # scale penalty
        s_c = change(sz((pred_bbox[:, 2]-pred_bbox[:, 0]), (pred_bbox[:, 3]-pred_bbox[:, 1])) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     ((pred_bbox[:, 2]-pred_bbox[:, 0])/(pred_bbox[:, 3]-pred_bbox[:, 1])))

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        score = scores_siamese.copy()
        scores_siamese = penalty * scores_siamese

        pscore = scores_siamese
        
        ##unnecessary for VOT2018...
        if self.speed > cfg.TRACK.SPEED_INFLUENCE * max(self.size) or \
           self.speed_x > cfg.TRACK.SPEED_INFLUENCE * self.size[0] or \
           self.speed_y > cfg.TRACK.SPEED_INFLUENCE * self.size[1]:
            window_influence = cfg.TRACK.WINDOW_INFLUENCE_FAST
        elif self.speed > max(self.size) or self.speed_x > self.size[0] or self.speed_y > self.size[1]:
            window_influence = cfg.TRACK.WINDOW_INFLUENCE_MEDIUM
        else:
            window_influence = cfg.TRACK.WINDOW_INFLUENCE
            
        pscore = pscore * (1 - window_influence) + self.window * window_influence
        best_idx = np.argmax(pscore)
        
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        pred_bbox = pred_bbox[best_idx]
        box = np.array([0.0, 0.0, 0.0, 0.0])
        box[0] = (pred_bbox[0] + pred_bbox[2]) / 2 - cfg.TRACK.INSTANCE_SIZE // 2
        box[1] = (pred_bbox[1] + pred_bbox[3]) / 2 - cfg.TRACK.INSTANCE_SIZE // 2
        box[2] = (pred_bbox[2] - pred_bbox[0] + 1)
        box[3] = (pred_bbox[3] - pred_bbox[1] + 1)
        final_bbox = box.copy()
        bbox = box / scale_z
        
        cx = self.center_pos[0] + bbox[0]
        cy = self.center_pos[1] + bbox[1]
        
        self.dist_x.append(bbox[0])
        self.dist_y.append(bbox[1])
        self.dist.append(math.sqrt(bbox[0]**2 + bbox[1]**2))
        self.speed = max(self.dist[-3:])
        self.speed_x = max(self.dist_x[-3:])
        self.speed_y = max(self.dist_y[-3:])
        
        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        
        return {
                'bbox': bbox,
                'best_score': score[best_idx]
               }

               