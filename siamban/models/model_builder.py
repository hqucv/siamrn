# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss, mse_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head, get_dect_head
from siamban.models.neck import get_neck
from siamban.datasets.multidect_target import GetMultidectTarget


# from siamban.utils.refine_cls import RefineCls


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

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

        # get multidect
        if cfg.MULDECT.MULDECT:
            self.get_multidect_target = GetMultidectTarget()
            # build multi dect head
            self.multi_det = get_dect_head(cfg.MULDECT.TYPE,
                                           **cfg.MULDECT.KWARGS)
            if cfg.MULDECT.TRAINHEAD:
                self.head4dect = get_dect_head(cfg.MULDECT.TYPE,
                                               **cfg.MULDECT.KWARGS)

    def template(self, z, gt_bbox=None):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf_ = self.neck(zf)
            zf = [f[0] for f in zf_]
            zf_o = [f[1] for f in zf_]
        if cfg.MULDECT.MULDECT:
            self.gt_roi_feature = self.get_multidect_target.prroi_pool(zf_o, torch.from_numpy(gt_bbox),
                                                                       cfg.TRAIN.EXEMPLAR_SIZE,
                                                                       type='template')
        self.zf = zf

    def instance(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER - 1]
            else:
                self.cf = xf

    def track(self, x, points=None):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER - 1]
            else:
                self.cf = xf

        cls, loc = self.head(self.zf, xf)

        if cfg.MULDECT.MULDECT:
            all_proposals_roi, origin_score = self.get_multidect_target(cls=cls, loc=loc, points=points, xf=xf,
                                                                        type='track', x=x)
            matching_scores = self.multi_det('test', self.gt_roi_feature, all_proposals_roi)
            matching_scores = F.softmax(matching_scores, dim=1).data[:, 1].cpu().numpy()
            # use new response map to build cls map
            cls, _, origin_corr_cls, corr_cls = self.head(self.zf, xf, matching_scores)

        return {
            'cls': cls,
            'loc': loc,
            'original_cls_map': origin_score if cfg.MULDECT.MULDECT else None,
            'corr_cls': corr_cls if cfg.MULDECT.MULDECT else None,
            'origin_corr_cls': origin_corr_cls if cfg.MULDECT.MULDECT else None,
            'matching_scores': matching_scores if cfg.MULDECT.MULDECT else None
        }

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data, epoch):
        """ only used in training
        """
        # ----------get data----------
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bs = cfg.TRAIN.BATCH_SIZE
        if cfg.MULDECT.MULDECT:
            # ----------get data----------
            neg_support_image = data['neg_support_image'].cuda()
            query_img = data['query_img'].cuda()
            template_bbox = data['template_bbox'].cuda()
            neg_support_bbox = data['neg_support_bbox'].cuda()
            query_bbox = data['query_bbox']
            points = data['points']
            # pos = data['pos']   # the positive sample position in regression
            # ----------get feature----------
            zf_nf = self.backbone(torch.cat((template, neg_support_image), dim=0))
            zf, nf = [f[:bs] for f in zf_nf], [f[bs:] for f in zf_nf]
            xf_qf = self.backbone(torch.cat((search, query_img), dim=0))
            xf, qf = [f[:bs] for f in xf_qf], [f[bs:] for f in xf_qf]
            # qf = self.backbone(query_img)
            # nf = self.backbone(neg_support_image)
            if cfg.ADJUST.ADJUST:
                zf_ = self.neck(zf)
                zf = [f[0] for f in zf_]
                zf_o = [f[1] for f in zf_]
                xf = self.neck(xf)
            cls, loc = self.head(zf, xf)

            qf = self.neck(qf)
            # nf = self.neck(nf)
            nf_ = self.neck(nf)
            nf, nf_o = [f[0] for f in nf_], [f[1] for f in nf_]

            cls_zq = None
            loc_zq = None
            if epoch >= cfg.TRAIN.HNM_EPOCH:
                cls_zq, loc_zq = self.head(zf, qf)
            support_rois, proposal_rois, matching_label, pos_proposals, neg_proposals = \
                self.get_multidect_target(qimg=query_img, points=points, loc=loc_zq, cls=cls_zq,
                                          zf=zf_o, qf=qf, nf=nf_o,
                                          template_bbox=template_bbox, query_bbox=query_bbox,
                                          neg_support_bbox=neg_support_bbox, epoch=epoch, type='train')
            # multi-relation detector
            matching_scores = self.multi_det('train', support_rois, proposal_rois)

            # add trained redect head (head4dect)

            if cfg.MULDECT.TRAINHEAD:
                refined_cls = self.get_multidect_target(points=points, cls=cls, loc=loc, zf=zf, xf=xf,
                                                        template_bbox=template_bbox, detector=self.multi_det)

            # ----------get loss----------
            matching_scores = F.log_softmax(
                matching_scores.view(cfg.TRAIN.BATCH_SIZE, -1, 2), dim=2)
            if cfg.TRAIN.CE_LOSS:
                matching_loss = select_cross_entropy_loss(matching_scores, torch.from_numpy(matching_label).cuda())
            elif cfg.TRAIN.MSE_LOSS:
                matching_loss = mse_loss(matching_scores, torch.from_numpy(matching_label).cuda())
        else:
            # ----------get feature----------
            zf = self.backbone(template)
            xf = self.backbone(search)
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)
            cls, loc = self.head(zf, xf)

        # ----------get loss----------
        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        if cfg.MULDECT.MULDECT:
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                    cfg.TRAIN.LOC_WEIGHT * loc_loss + matching_loss
            outputs['matching_loss'] = matching_loss
        else:
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                    cfg.TRAIN.LOC_WEIGHT * loc_loss

        if cfg.TRAIN.VISUAL:
            vis_data = {'template_img': data['template'],
                        'search_img': data['search'],
                        'query_img': data['query_img'],
                        'neg_sup_img': data['neg_support_image'],
                        'template_bbox': data['template_bbox'],
                        'search_bbox': data['search_bbox'],
                        'neg_support_bbox': data['neg_support_bbox'],
                        'query_bbox': data['query_bbox'],
                        'neg_proposals': neg_proposals,
                        'pos_proposals': pos_proposals,
                        'matching_label': matching_label
                        }
            return outputs, vis_data
        else:
            return outputs
