import random
import torch
import numpy as np

from siamban.core.config import cfg


class GenerateRoiLabel:
    def __init__(self):
        super(GenerateRoiLabel, self).__init__()

    def __call__(self, pos_sup_roi, neg_sup_roi, query_gt_roi, pos_proposals_rois, neg_proposals_rois):
        length = len(pos_sup_roi)
        batch_size = pos_sup_roi[0].shape[0]
        final_support_rois = []
        final_proposal_rois = []
        final_label = []
        wh = pos_sup_roi[0].size(-1)
        pos_proposals_rois = [roi.reshape(batch_size, -1, 256, wh, wh) for roi in pos_proposals_rois]
        neg_proposals_rois_1 = [roi.reshape(batch_size, -1, 256, wh, wh)[:, :16, ...] for roi in neg_proposals_rois]
        neg_proposals_rois_2 = [roi.reshape(batch_size, -1, 256, wh, wh)[:, 16:32, ...] for roi in neg_proposals_rois]
        neg_proposals_rois_3 = [roi.reshape(batch_size, -1, 256, wh, wh)[:, 32:48, ...] for roi in neg_proposals_rois]
        pos_or_neg = pos_proposals_rois if random.random() > 0.5 else neg_proposals_rois_3
        pos_sup_rois = [roi[:, None, :, :, :].repeat(1, 16, 1, 1, 1) for roi in pos_sup_roi]
        neg_sup_rois = [roi[:, None, :, :, :].repeat(1, 16, 1, 1, 1) for roi in neg_sup_roi]

        if cfg.TRAIN.CE_LOSS:
            training_pairs = [(1, pos_sup_rois, pos_proposals_rois), (0, pos_sup_rois, neg_proposals_rois_1),
                              (0, pos_sup_rois, neg_proposals_rois_2), (0, neg_sup_rois, pos_or_neg)]
        elif cfg.TRAIN.MSE_LOSS:
            training_pairs = [([1, 0], pos_sup_rois, pos_proposals_rois), ([0, 1], pos_sup_rois, neg_proposals_rois_1),
                              ([0, 1], pos_sup_rois, neg_proposals_rois_2),
                              ([0, 1], neg_sup_rois, pos_or_neg)]
        for id in range(batch_size):
            random.shuffle(training_pairs)
            if id == 0:
                final_support_rois = [
                    torch.cat((training_pairs[0][1][i][id], training_pairs[1][1][i][id], training_pairs[2][1][i][id],
                               training_pairs[3][1][i][id]), dim=0) for i in range(length)]
                final_proposal_rois = [
                    torch.cat((training_pairs[0][2][i][id], training_pairs[1][2][i][id], training_pairs[2][2][i][id],
                               training_pairs[3][2][i][id]), dim=0) for i in range(length)]
            else:
                final_support_rois = [
                    torch.cat((final_support_rois[i], training_pairs[0][1][i][id], training_pairs[1][1][i][id],
                               training_pairs[2][1][i][id],
                               training_pairs[3][1][i][id]), dim=0) for i in range(length)]
                final_proposal_rois = [
                    torch.cat((final_proposal_rois[i], training_pairs[0][2][i][id], training_pairs[1][2][i][id],
                               training_pairs[2][2][i][id],
                               training_pairs[3][2][i][id]), dim=0) for i in range(length)]

            tmp_label = [[training_pairs[0][0]] * 16, [training_pairs[1][0]] * 16,
                         [training_pairs[2][0]] * 16, [training_pairs[3][0]] * 16]
            final_label.append(tmp_label)
        return final_support_rois, final_proposal_rois, np.array(final_label)
