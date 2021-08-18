from siamban.core.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from siamban.core.config import cfg
from siamban.utils.bbox import lt2corner_rect
import torch
import numpy as np


class GetPrroiPoolFeature:
    def __init__(self):
        super(GetPrroiPoolFeature, self).__init__()

    def __call__(self, feature, bboxes, origin_size, type=None):
        avg_pool = PrRoIPool2D(7, 7, 1 / cfg.POINT.STRIDE)
        roi_feature_temp = []
        # batch_size = cfg.TRAIN.BATCH_SIZE
        batch_size = feature[0].shape[0]
        if type == 'template':
            # dim = 5,(batch_id, x1, y1, x2, y2)
            _bboxes = torch.zeros(batch_size, 5)
            _bboxes[:, 0] = torch.tensor(range(0, batch_size))
            _bboxes[:, 1:] = bboxes
            for i in range(len(feature)):
                roi_feature_temp.append(avg_pool(feature[i], _bboxes.cuda()))
        else:
            if type == 'track':
                # bbox is lt-based, need to change to corner-base
                bboxes_num_per_batch = bboxes.shape[-1]
                bboxes = lt2corner_rect(bboxes.transpose(0, 2, 1).reshape(batch_size * bboxes_num_per_batch, 4))
            elif type == 'search':
                bboxes_num_per_batch = bboxes.shape[1]
                bboxes = lt2corner_rect(bboxes).reshape(batch_size * bboxes_num_per_batch, 4)

            # bboxes_num_per_batch = bboxes.shape[-1]
            # bboxes = bboxes.transpose(0, 2, 1).reshape(batch_size * bboxes_num_per_batch, 4)

            # bboxes = bboxes.reshape(batch_size * bboxes_num_per_batch, 4)
            # change to corner-base

            # check value (<0 or >255/127)
            bboxes[np.where(bboxes < 0)] = 0
            bboxes[np.where(bboxes > origin_size)] = origin_size

            _bboxes = torch.zeros(batch_size * bboxes_num_per_batch, 4+1)
            _bboxes[:, 0] = \
                torch.tensor(list(range(batch_size)) * bboxes_num_per_batch).view(-1, batch_size).permute(1, 0).reshape(-1)
            _bboxes[:, 1:] = torch.from_numpy(bboxes)
            for i in range(len(feature)):
                roi_feature_temp.append(avg_pool(feature[i], _bboxes.cuda()))
        return roi_feature_temp
