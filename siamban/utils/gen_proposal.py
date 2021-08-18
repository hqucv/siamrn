import numpy as np
import cv2
import torch

from siamban.utils.bbox import center2corner_rect, corner2lt


# test sample
trans_pos = 0.1
scale_pos = 1.3
n_pos_init = 1
n_neg_init = 2
trans_neg_init = 1
scale_neg_init = 1.5
overlap_pos_init = [0.7, 1]
overlap_neg_init = [0, 0.1]


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

class SampleGenerator():
    def __init__(self, type_, img_size, trans=1, scale=1, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size)  # (w, h)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def _gen_samples(self, bb, n):
        # bathed bb (new)
        # bb: target bbox (min_x,min_y,xmax,ymax)
        bb = np.array(bb, dtype='float32')
        # lt2center (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None, :], (n, 1))  # (n,1) is dimension

        # vary aspect ratio
        if self.aspect is not None:
            ratio = np.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect ** ratio  # w,h

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == 'uniform':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def __call__(self, imgs, bboxes, n, overlap_range=None, scale_range=None):
        '''
        :param bbox:  [b, xmin,ymin,xmax,ymax]
        :param n:
        :param overlap_range:
        :param scale_range:
        :return:
        '''
        if isinstance(bboxes, torch.Tensor):
            bboxes = np.array(bboxes)
        bs = bboxes.shape[0]
        ss = np.zeros((bs, n, 4))
        for i in range(bs):
            bbox = bboxes[i]
            bbox = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])  # corner2lt
            if overlap_range is None and scale_range is None:
                return self._gen_samples(bbox, n)

            else:
                samples = None
                remain = n
                factor = 2
                while remain > 0 and factor < 16:
                    samples_ = self._gen_samples(bbox, remain * factor)

                    idx = np.ones(len(samples_), dtype=bool)
                    if overlap_range is not None:
                        r = overlap_ratio(samples_, bbox)
                        idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
                    if scale_range is not None:
                        s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                        idx *= (s >= scale_range[0]) * (s <= scale_range[1])

                    samples_ = samples_[idx, :]
                    samples_ = samples_[:min(remain, len(samples_))]
                    if samples is None or len(samples) == 0:
                        samples = samples_
                    else:
                        samples = np.concatenate([samples, samples_])
                    remain = n - len(samples)
                    factor = factor * 2
                try:
                    ss[i, ...] = samples
                except ValueError:
                    print(bbox)
        return ss


if __name__ == '__main__':
    img = cv2.imread('../../0005.jpg')
    _img_size = img.shape[:2]
    # box = np.array([198, 214, 34, 81])  # left-top w,h
    # box = np.array([335, 160, 26, 61])[None, :]  # left-top w,h
    box = np.array([335, 160, 361, 221])[None, :]  # corner

    #_img_size = (255, 255)
    #box = np.array(range(28*4)).reshape(28, 4)
    pos_examples = SampleGenerator('gaussian', _img_size, trans_pos, scale_pos)(
        box, n_pos_init, overlap_pos_init)

    neg_examples = SampleGenerator('uniform', _img_size, trans_neg_init, scale_neg_init)(
            box, int(n_neg_init * 0.5), overlap_neg_init)
        # np.concatenate([
        # SampleGenerator('uniform', _img_size, trans_neg_init, scale_neg_init)(
        #     box, int(n_neg_init * 0.5), overlap_neg_init),
        # SampleGenerator('whole', _img_size)(
        #     box, int(n_neg_init * 0.5), overlap_neg_init)])
    neg_examples = np.random.permutation(neg_examples)
    # plot box
    # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)    # gt
    for b in neg_examples[0]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (255, 0, 255), 2)
    for b in pos_examples[0]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (255, 0, 0), 2)
    cv2.imwrite('../../sample1.jpg', img)
    print(len(neg_examples[0]))
