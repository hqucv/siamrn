from siamban.utils.bbox import center2corner_rect, corner2lt, rect_iou
from siamban.core.config import cfg
import numpy as np
import random


class ProposalTarget:
    def __init__(self):
        super(ProposalTarget, self).__init__()

    def __call__(self, all_proposals, gt):
        '''
        :param all_proposals: left top based
        :param gt: corner based
        :return:
        '''
        batchsize, num = all_proposals.shape[0], all_proposals.shape[2]     # 28, 625
        select_proposals_pos, select_proposals_neg, pos_num, neg_num = self.get_select_proposal_label(all_proposals, gt, batchsize, num)
        return select_proposals_pos.transpose(0, 2, 1), \
               select_proposals_neg.transpose(0, 2, 1), \
               pos_num, neg_num


    def get_iou(self, proposals, gt, batchsize, num):
        proposals = proposals.transpose(0, 2, 1).reshape(-1, 4)
        # proposals_cen = np.array([center2corner(proposals_cen[x]) for x in range(proposals_cen.shape[0])])
        gt = np.tile(corner2lt(gt.cpu().numpy()), num).reshape(-1, 4)
        assert gt.shape == proposals.shape, "proposals size don't match the gt size"
        ious = rect_iou(proposals, gt)
        return ious

    def get_box_center(self, box):
        return (box[2] - box[0]) / 2, (box[3] - box[1]) / 2

    def dist_2_points(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def get_select_proposal_label(self, all_proposals, gt, batchsize, num):
        '''
        :param all_proposals: left-top based
        :param gt:
        :param batchsize:
        :param num:
        :return:
        '''
        ious = self.get_iou(all_proposals, gt, batchsize, num).reshape(batchsize, num)
        # label = -1 * np.ones((batchsize, num), dtype=np.int64)

        def select(position, keep_num=2, type='pos'):
            count = {}
            _pos = []
            _neg = []
            # init count
            for i in range(cfg.TRAIN.BATCH_SIZE):
                count[i] = []
            # fill in count
            for i in position:
                count[i[0]].append(i[1])
            if type == 'pos':
                for j in count:
                    if len(count[j]) <= keep_num:
                        continue
                    else:
                        tmp = count[j]
                        count[j] = random.sample(tmp, keep_num)
                for i in count.keys():
                    for j in count[i]:
                        _pos.append([i, j])
                return count        # (np.array(_pos)[:, 0], np.array(_pos)[:, 1]),
            elif type == 'neg':
                for j in count:
                    tmp = count[j]
                    # print(keep_num + (2 - len(self.count_pos[j])))
                    # select neg samples around target
                    gt_center = self.get_box_center(gt[j].cpu().numpy())
                    proposal_center = [self.get_box_center(all_proposals[j].transpose(1, 0)[i]) for i in tmp]
                    dist = [self.dist_2_points(p, gt_center) for p in proposal_center]
                    neg_sort_id = np.argsort(np.array(dist))    # small to big
                    select_num = keep_num  # + (cfg.TRAIN.PROPOSAL_POS - len(self.count_pos[j]))
                    select = [tmp[s] for s in neg_sort_id[:select_num]]
                    count[j] = select

                    # select_num = keep_num  # + (cfg.TRAIN.PROPOSAL_POS - len(self.count_pos[j]))
                    # count[j] = random.sample(tmp, select_num)

                for i in count.keys():
                    for j in count[i]:
                        _neg.append([i, j])
                return count    # (np.array(_neg)[:, 0], np.array(_neg)[:, 1])

        pos = np.argwhere(ious > 0.6)
        neg = np.argwhere(ious < 0.2)
        self.count_pos = select(pos, cfg.TRAIN.PROPOSAL_POS, 'pos')    # max 16
        self.count_neg = select(neg, cfg.TRAIN.PROPOSAL_NEG, 'neg')  # 3n

        #proposals_id = [[0 for i in range(2)] for j in range(batchsize)]

        # pos proposals

        # neg proposals

        # for i in range(batchsize):
        #     tmp = [x for x in self.count_pos[i]] + [x for x in self.count_neg[i]]
        #     proposals_id[i] = [x for x in tmp]   # [pos, neg] or [neg, neg]


        select_proposals_pos = np.zeros((batchsize, 4, cfg.TRAIN.PROPOSAL_POS), dtype=np.float)
        select_proposals_neg = np.zeros((batchsize, 4, cfg.TRAIN.PROPOSAL_NEG), dtype=np.float)
        # for i in range(batchsize):
        neg_num = []
        for i in self.count_neg:
            idx = self.count_neg[i]
            neg_num.append(len(idx))
            select_proposals_neg[i][:, :len(idx)] = all_proposals[i][:, idx]


        # for i in range(batchsize):
        pos_num = []
        for i in self.count_pos:
            idx = self.count_pos[i]
            pos_num.append(len(idx))
            select_proposals_pos[i][:, :len(idx)] = all_proposals[i][:, idx]

        return select_proposals_pos, select_proposals_neg, pos_num, neg_num
