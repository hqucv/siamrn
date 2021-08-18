import numpy as np

from siamban.utils.bbox import corner2lt, rect_iou


class GetHN:
    def __init__(self):
        super(GetHN, self).__init__()

    def __call__(self, all_proposals, cls_score, gt):
        '''
        :param all_proposals: (28, 4, 625) left-top based
        :param cls_score: (28, 625)
        :param gt: [28, 4] corner-based
        :return:
        '''
        # compute iou between all proposals and gt
        threshold = 0.1
        batchsize = all_proposals.shape[0]
        num = all_proposals.shape[2]
        ious = self.get_iou(all_proposals, gt, batchsize, num).reshape(batchsize, num)  # [28, 625]
        # sort cls score
        sort_cls_idx = np.argsort(-cls_score, axis=1)   # big to small
        # select proposal
        select_proposal = []
        b_idx = np.repeat(np.array(range(0, batchsize)), num)
        sort_ious_idx = (b_idx, sort_cls_idx.reshape(-1))
        sort_ious = ious[sort_ious_idx].reshape(batchsize, num)

        for b in range(batchsize):
            # idx = np.argwhere(ious[b][sort_cls_idx[b]] < 0.1)
            idx = np.where(sort_ious[b] < threshold)
            # select the hard negative samples idx
            selected_cls_idx = sort_cls_idx[b][idx]
            hn_proposals = all_proposals[b, :, selected_cls_idx[:24]]
            # get the related proposal
            select_proposal.append(hn_proposals)
        return np.array(select_proposal)


    def get_iou(self, proposals, gt, batchsize, num):
        proposals = proposals.transpose(0, 2, 1).reshape(-1, 4)
        # proposals_cen = np.array([center2corner(proposals_cen[x]) for x in range(proposals_cen.shape[0])])
        gt = np.tile(corner2lt(gt.cpu().numpy())[:, None, :], (1, num, 1)).reshape(-1, 4)
        assert gt.shape == proposals.shape, "proposals size don't match the gt size"
        ious = rect_iou(proposals, gt)
        ious = ious.reshape(batchsize, num)
        return ious