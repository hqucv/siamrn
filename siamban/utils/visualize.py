import visdom
import numpy as np
import matplotlib.pyplot as plt


plt.switch_backend('agg')
import matplotlib.patches as patches
import cv2
import torch
from siamban.core.config import cfg
from siamban.utils.bbox import corner2center

class Visual:
    def __init__(self, port=8097):
        self.vis = visdom.Visdom(port=port)
        self.counter = 0
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.var = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def get_im_patch(self, img, template_bbox, limit_size):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy().astype('uint8')
        if isinstance(template_bbox, torch.Tensor):
            template_bbox = template_bbox.cpu().detach().numpy()
        for i in range(len(template_bbox)):
            if template_bbox[i] < 0:
                template_bbox[i] = 0
            elif template_bbox[i] > limit_size:
                template_bbox[i] = limit_size
        x1, y1, x2, y2 = template_bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        patch = img.transpose(1, 2, 0)[y1:y2, x1:x2]
        patch = cv2.resize(patch, (80, 80))
        return patch.transpose(2, 0, 1)[::-1, :, :]

    def denormalize(self, img):
        img = img.cpu().detach().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def plot_img_bbox(self, img, bboxes, win, name, box_color):
        """
        :param img:  [3, *, *] tensor
        :param bboxes: [4, 5]  cx,cy,w,h    [4,] xmin,ymin,xmax,ymax
        :param win:
        :param name:
        :return:
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy().astype('uint8')[::-1, :, :]
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().detach().numpy()
        box_img = self.add_box_to_img(img.transpose(1, 2, 0), bboxes, box_color)
        self.plot_img(box_img.transpose(2, 0, 1), win, name)

    def plot_text(self, context, win):
        text = context
        self.vis.text(text, win)


    def plot_error(self, errors, win=0, id_val=1):
        if not hasattr(self, 'plot_data'):
            self.plot_data = [{'X': [], 'Y': [], 'legend': list(errors.keys())}]
        elif len(self.plot_data) != id_val:
            self.plot_data.append({'X': [], 'Y': [], 'legend': list(errors.keys())})
        id_val -= 1
        self.plot_data[id_val]['X'].append(self.counter)
        self.plot_data[id_val]['Y'].append([errors[k] for k in self.plot_data[id_val]['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data[id_val]['X'])] * len(self.plot_data[id_val]['legend']), 1),
            Y=np.array(self.plot_data[id_val]['Y']),
            opts={
                'legend': self.plot_data[id_val]['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'}, win=win)
        self.counter += 1

    def plot_img(self, img, win=1, name='img'):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy().astype('uint8')
        self.vis.image(img, win=win, opts={'title': name})

    def plot_img_list(self, img, win=2, name='img'):
        self.vis.images(img, 2, 20, win=win, opts={'title': name})

    def plot_box(self, im1, gt_box1, im2, gt_box2, box, name='img', win=1):
        im1 = self.denormalize(im1)
        im2 = self.denormalize(im2)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(im1)
        p = patches.Rectangle(
            (gt_box1[0], gt_box1[1]), gt_box1[2] - gt_box1[0], gt_box1[3] - gt_box1[1],
            fill=False, clip_on=False, color='r'
        )
        ax.add_patch(p)

        ax = fig.add_subplot(122)
        ax.imshow(im2)
        p = patches.Rectangle(
            (gt_box2[0], gt_box2[1]), gt_box2[2] - gt_box2[0], gt_box2[3] - gt_box2[1],
            fill=False, clip_on=False, color='r'
        )
        ax.add_patch(p)
        box = box.copy()
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]
        for i in range(box.shape[0]):
            p = patches.Rectangle(
                (box[i, 0], box[i, 1]), box[i, 2], box[i, 3],
                fill=False, clip_on=False, color='b'
            )
            ax.add_patch(p)
        self.vis.matplot(fig, win=win, opts={'title': name})
        plt.clf()

    def add_box_to_img(self, img, boxes, color=(255, 0, 0)):
        """
        :param img:  (3, 255, 255)
        :param boxes: (cx,cy,w,h)
        :param color:
        :return:
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        for box in boxes:
            point_1 = [- box[2] / 2 + box[0] + 0.5, - box[3] / 2 + box[1] + 0.5]
            point_2 = [box[2] / 2 + box[0] - 0.5, box[3] / 2 + box[1] - 0.5]
            point_1[0] = np.clip(point_1[0], 0, img.shape[0])
            point_2[0] = np.clip(point_2[0], 0, img.shape[0])
            point_1[1] = np.clip(point_1[1], 0, img.shape[1])
            point_2[1] = np.clip(point_2[1], 0, img.shape[1])
            img = cv2.rectangle(cv2.UMat(img).get(),
                                (int(point_1[0]), int(point_1[1])),
                                (int(point_2[0]), int(point_2[1])),
                                color,
                                1)
        return img

    def modulate(self, score, out_size):
        score_per_temp = int(np.prod(score.shape) / (np.prod(out_size)))
        score_im = score.reshape(score_per_temp, *out_size)
        score_mean = np.mean(score_im, axis=0).reshape(1, *out_size)
        score_norm = score_mean / np.max(score_mean)
        return score_norm

    def viz_score_map(self, im, score, out_size):

        def to_numpy(tensor):
            if torch.is_tensor(tensor):
                return tensor.detach().cpu().numpy()
            elif type(tensor).__module__ != 'numpy':
                raise ValueError("Cannot convert {} to numpy array"
                                 .format(type(tensor)))
            return tensor

        def torch_to_img(img):
            img = to_numpy(torch.squeeze(img, 0))
            img = np.transpose(img, (1, 2, 0))  # H*W*C
            return img

        score_viz = self.modulate(to_numpy(score), out_size)
        im = cv2.resize(torch_to_img(im), (cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE)).astype(np.uint8)
        canvas = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)

        # calculate the color map
        score_im_base = cv2.resize(score_viz[0], im.shape[:2])
        score_im_base = (255 * score_im_base).astype(np.uint8)
        im_color = cv2.applyColorMap(score_im_base, cv2.COLORMAP_JET)

        # show the image
        overlayed_im = cv2.addWeighted(im, 0.8, im_color, 0.7, 0)
        canvas[:, :im.shape[1], :] = overlayed_im
        return canvas
