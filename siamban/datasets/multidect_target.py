import random

from siamban.utils.get_prroi_pool import GetPrroiPoolFeature
from siamban.utils.gen_proposal import SampleGenerator
from siamban.utils.ohnm import GetHN
from siamban.datasets.proposals_target import ProposalTarget
from siamban.datasets.maching_net_rois_label import GenerateRoiLabel
from siamban.utils.bbox import corner2center
from siamban.utils.point import Point
from siamban.core.config import cfg


class GetMultidectTarget:

    def __init__(self):
        # buid roi_pool layer
        self.prroi_pool = GetPrroiPoolFeature()
        self.get_proposals = ProposalTarget()
        self.generate_roi_label = GenerateRoiLabel()
        self.select_hn_proposal = GetHN()
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels

    def __call__(self, qimg=None, points=None, cls=None, loc=None, zf=None, xf=None, qf=None, nf=None,
                 template_bbox=None, query_bbox=None, neg_support_bbox=None, x=None, epoch=None, type='train'):

        if type == 'train':
            bs = qimg.shape[0]
            qimg_size = qimg.shape[-2:]

            pos_proposals = SampleGenerator(cfg.MULDECT.POS_SAMPLETYPE, qimg_size,
                                            cfg.MULDECT.TRANS_POS, cfg.MULDECT.SCALE_POS)(qimg, query_bbox,
                                                                                          cfg.MULDECT.N_POS_INIT,
                                                                                          cfg.MULDECT.OVERLAP_POS_INIT)

            if loc is not None:
                # get all proposals
                all_proposals = self._convert_bbox(loc, points, type='train_lt')
                select_proposals_pos, select_proposals_neg, pos_num, neg_num = self.get_proposals(all_proposals,
                                                                                                  query_bbox)
                # fill the pos samples
                for b, n in enumerate(pos_num):
                    if n < cfg.TRAIN.POS_NUM:
                        select_proposals_pos[b][n:, :] = pos_proposals[b][:cfg.TRAIN.POS_NUM - n, :]
                pos_proposals = select_proposals_pos
                neg_proposals = select_proposals_neg
            else:
                if cfg.TRAIN.HNM:
                    pass
                else:
                    # ----------get neg proposals   [lt-base]
                    neg_proposals = SampleGenerator(cfg.MULDECT.NEG_SAMPLETYPE, qimg_size,
                                                    cfg.MULDECT.TRANS_NEG_INIT, cfg.MULDECT.SCALE_NEG_INIT)(qimg,
                                                                                                            query_bbox,
                                                                                                            int(
                                                                                                                0.5 * cfg.MULDECT.N_NEG_INIT),
                                                                                                            cfg.MULDECT.OVERLAP_NEG_INIT)

            # ohnm
            if epoch >= cfg.TRAIN.HNM_EPOCH:
                # # get all proposals
                # all_proposals = self._convert_bbox(loc, points, type='train_lt')
                # get cls scores
                cls_score = self._convert_score(cls)

                hn_proposals = self.select_hn_proposal(all_proposals, cls_score,
                                                       query_bbox)
                neg_proposals[:, :int((cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM) / 2), :] = hn_proposals

            # ----------get roi pool feature
            # pos_support (support)
            pos_sup_roi = self.prroi_pool(zf, template_bbox, cfg.TRAIN.EXEMPLAR_SIZE,
                                          type='template')
            # neg_support (support)
            neg_sup_roi = self.prroi_pool(nf, neg_support_bbox, cfg.TRAIN.EXEMPLAR_SIZE,
                                          type='template')
            # pos and neg proposals (query)
            pos_proposals_rois = self.prroi_pool(qf, pos_proposals, cfg.TRAIN.SEARCH_SIZE, type='search')
            neg_proposals_rois = self.prroi_pool(qf, neg_proposals, cfg.TRAIN.SEARCH_SIZE, type='search')

            # GT (query)
            query_gt_roi = \
                self.prroi_pool(qf, query_bbox, cfg.TRAIN.SEARCH_SIZE,
                                type='template')

            support_rois, proposal_rois, matching_label = self.generate_roi_label(pos_sup_roi, neg_sup_roi,
                                                                                  query_gt_roi, pos_proposals_rois,
                                                                                  neg_proposals_rois)

            return support_rois, proposal_rois, matching_label, pos_proposals, neg_proposals
        elif type == 'track':
            origin_score = self._convert_score(cls)
            all_proposals = self._convert_bbox(loc, points, type='track_lt')
            all_proposals_roi = \
                self.prroi_pool(xf, all_proposals[None, :], cfg.TRAIN.SEARCH_SIZE, type='track')
            return all_proposals_roi, origin_score

    def _convert_bbox(self, delta, point, type='center'):
        if 'train' in type:
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1, delta.shape[0]).permute(2, 0, 1)
            delta = delta.detach().cpu().numpy()
            point = point.permute(1, 2, 3, 0).contiguous().view(2, -1, delta.shape[0]).permute(2, 0, 1)
            point = point.data.cpu().numpy()
            delta[:, 0, :] = point[:, 0, :] - delta[:, 0, :]
            delta[:, 1, :] = point[:, 1, :] - delta[:, 1, :]
            delta[:, 2, :] = point[:, 0, :] + delta[:, 2, :]
            delta[:, 3, :] = point[:, 1, :] + delta[:, 3, :]
            if type == 'train_center':
                # center based
                delta_c = delta.copy()
                delta_c[:, 2, :] = delta[:, 2, :] - delta[:, 0, :]
                delta_c[:, 3, :] = delta[:, 3, :] - delta[:, 1, :]
                delta_c[:, 0, :] = delta[:, 0, :] + 1 / 2 * delta_c[:, 2, :]
                delta_c[:, 1, :] = delta[:, 1, :] + 1 / 2 * delta_c[:, 3, :]
                return delta_c
            elif type == 'train_lt':
                # left top based
                delta_lt = delta.copy()
                delta_lt[:, 2, :] = delta_lt[:, 2, :] - delta_lt[:, 0, :]
                delta_lt[:, 3, :] = delta_lt[:, 3, :] - delta_lt[:, 1, :]
                return delta_lt
        elif 'track' in type:
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
            delta = delta.detach().cpu().numpy()
            point = point.reshape(2, -1)
            delta[0, :] = point[0, :] - delta[0, :]
            delta[1, :] = point[1, :] - delta[1, :]
            delta[2, :] = point[0, :] + delta[2, :]
            delta[3, :] = point[1, :] + delta[3, :]
            if type == 'track_center':
                delta_c = delta.copy()
                delta_c[0, :] = (delta[0, :] + delta[2, :]) / 2
                delta_c[1, :] = (delta[1, :] + delta[3, :]) / 2
                delta_c[2, :] = delta[2, :] - delta[0, :] + 1
                delta_c[3, :] = delta[3, :] - delta[1, :] + 1
                return delta_c
            if type == 'track_lt':
                delta_lt = delta.copy()
                delta_lt[2, :] = delta_lt[2, :] - delta_lt[0, :]
                delta_lt[3, :] = delta_lt[3, :] - delta_lt[1, :]
                return delta_lt
        else:
            assert False, 'convert box is no type matching'

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            if score.shape[0] == 1:
                score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
                score = score.softmax(1).detach()[:, 1].cpu().numpy()
            else:
                score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1, score.shape[0]).permute(
                    2, 1, 0)
                score = score.softmax(2).detach()[:, :, 1].cpu().numpy()
        return score
