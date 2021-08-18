from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from siamban.core.xcorr import xcorr_fast, xcorr_depthwise
from siamban.core.config import cfg
from siamban.models.neck.non_local import NONLocalBlock2D


class Detector(nn.Module):
    def __init__(self, ):
        super(Detector, self).__init__()

    def forward(self, target_feat, box_feat):
        raise NotImplementedError


def agg_feat(f):
    for i in range(len(f)):
        if i == 0:
            tmp = f[i]
        else:
            tmp += f[i]
    return tmp


class MultiDetector(Detector):
    def __init__(self, input_wh, in_channels, weighted=False, fusion=False):
        super(MultiDetector, self).__init__()
        self.weighted = weighted
        self.matching_mode = ['global_det', 'local_det', 'patch_det']
        self.fusion = fusion
        if self.fusion:
            in_channels = [in_channels[0]]
        for i in range(len(in_channels)):
            self.add_module('global_det' + str(i + 2),
                            GlobalDetector(input_wh, in_channels[i]))
            self.add_module('local_det' + str(i + 2),
                            LocalDetector(input_wh, in_channels[i]))
            self.add_module('patch_det' + str(i + 2),
                            PatchDetector(input_wh, in_channels[i]))
            self.add_module('non_local' + str(i + 2), NONLocalBlock2D(int(2 * in_channels[i])))
        if self.weighted:
            self.matching_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, type, support_rois, proposal_rois):
        matching_scores = []
        if self.fusion:
            support_rois = [agg_feat(support_rois)]
            proposal_rois = [agg_feat(proposal_rois)]
        if type == 'train':
            for idx, (support_roi, proposal_roi) in enumerate(zip(support_rois, proposal_rois)):
                support_roi = support_roi.reshape(-1, 256, 8, 8)
                proposal_roi = proposal_roi.reshape(-1, 256, 8, 8)
                nl_block = getattr(self, 'non_local' + str(idx + 2))
                detector1 = getattr(self, self.matching_mode[0] + str(idx + 2))
                detector2 = getattr(self, self.matching_mode[1] + str(idx + 2))
                detector3 = getattr(self, self.matching_mode[2] + str(idx + 2))
                cat_rois = nl_block(torch.cat((support_roi, proposal_roi), dim=1))
                score1 = detector1(cat_rois)
                # score1 = detector1(support_roi, proposal_roi)
                score2 = detector2(support_roi, proposal_roi)
                # score3 = detector3(support_roi, proposal_roi)
                score3 = detector3(cat_rois)
                matching_scores.append(score1 + score2 + score3)
        elif type == 'test':
            for idx, (target_feat, box_feat) in enumerate(zip(support_rois, proposal_rois)):
                _target_feat = target_feat.expand_as(box_feat)
                nl_block = getattr(self, 'non_local' + str(idx + 2))
                detector1 = getattr(self, self.matching_mode[0] + str(idx + 2))
                detector2 = getattr(self, self.matching_mode[1] + str(idx + 2))
                detector3 = getattr(self, self.matching_mode[2] + str(idx + 2))
                cat_rois = nl_block(torch.cat((_target_feat, box_feat), dim=1))
                score1 = detector1(cat_rois)
                # score1 = detector1(_target_feat, box_feat)
                score2 = detector2(_target_feat, box_feat)
                # score3 = detector3(_target_feat, box_feat)
                score3 = detector3(cat_rois)
                matching_scores.append(score1 + score2 + score3)

        if self.weighted:
            matching_weight = F.softmax(self.matching_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:  # average cls/reg
            return weighted_avg(matching_scores, matching_weight)
        else:
            return avg(matching_scores)


class GlobalDetector(Detector):
    def __init__(self, feat_size, in_channel):
        super(GlobalDetector, self).__init__()
        wh = feat_size
        c = in_channel
        self.avg_pool = nn.AvgPool2d(kernel_size=wh)
        self.MLP = nn.Sequential(
            nn.Linear(c * 2, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(c, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, cat_rois):  # target_feat, box_feat
        # box_feat = self.avg_pool(box_feat).squeeze(3).squeeze(2)
        # target_feat = self.avg_pool(target_feat).squeeze(3).squeeze(2).expand_as(box_feat)
        # concat_feat = torch.cat((target_feat, box_feat), dim=1)
        concat_feat = self.avg_pool(cat_rois).squeeze(3).squeeze(2)
        x = self.MLP(concat_feat)
        score = self.fc(x)
        return score


class LocalDetector(Detector):
    def __init__(self, feat_size, in_channel):
        super(LocalDetector, self).__init__()
        wh = feat_size
        c = in_channel
        self.conv = nn.Conv2d(c, c, 1, padding=0, bias=False)
        self.fc = nn.Linear(c, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, target_feat, box_feat):
        z_feat = self.conv(target_feat)
        x_feat = self.conv(box_feat)
        x = xcorr_depthwise(z_feat, x_feat)
        x = F.relu(x, inplace=True).squeeze(3).squeeze(2)
        score = self.fc(x)
        return score


class PatchDetector(Detector):
    def __init__(self, feat_size, in_channel):
        super(PatchDetector, self).__init__()
        wh = feat_size
        c = in_channel
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1 = nn.Conv2d(c * 2, int(c / 4), 1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(int(c / 4), int(c / 4), 3, padding=0, bias=False)
        self.conv_3 = nn.Conv2d(int(c / 4), c, 1, padding=0, bias=False)
        self.fc = nn.Linear(c, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, cat_rois):
        # concat_feat = torch.cat((target_feat, box_feat), dim=1)
        concat_feat = cat_rois
        x = self.conv_1(concat_feat)
        x = F.relu(x, inplace=True)
        x = self.avg_pool(x)
        x = self.conv_2(x)
        x = F.relu(x, inplace=True)
        x = self.conv_3(x)
        x = F.relu(x, inplace=True)
        x = self.avg_pool2(x)
        x = x.squeeze(3).squeeze(2)
        score = self.fc(x)
        return score
