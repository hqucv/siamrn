# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
import random

import cv2
import numpy as np
from annoy import AnnoyIndex
from torch.utils.data import Dataset

from siamban.utils.bbox import center2corner, Center
from siamban.datasets.point_target import PointTarget
from siamban.datasets.augmentation import Augmentation
from siamban.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in list(meta_data[video]):
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]
                # del less than 3 frames videos
                if len(frames) < 1:
                    logger.warning("{}/{} less than 1 frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 1 or h <= 1:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        if 'ILSVRC' in video_name:
            retrieve_key = video_name.split('/')[-1] + '_' + str(int(track))
        else:
            retrieve_key = video_name.split('/')[-1]
        frames = track_info['frames']       # frames maybe less than 2
        # reset index
        '''
        while len(frames) < 2:
            index = random.sample(range(0, index), 1)[0]
            video_name = self.videos[index]
            video = self.labels[video_name]
            track = np.random.choice(list(video.keys()))
            track_info = video[track]
            retrieve_key = video_name.split('/')[-1] + '_' + str(int(track))
            frames = track_info['frames']  # frames maybe less than 2
        '''
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame),\
               retrieve_key


    def get_positive_triple(self, index):
        # three frame extract from the same video
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        if 'ILSVRC' in video_name:
            retrieve_key = video_name.split('/')[-1] + '_' + str(int(track))
        else:
            retrieve_key = video_name.split('/')[-1]
        frames = track_info['frames']  # frames maybe less than 2
        # reset index
        self.info_frames_ = '''
        while len(frames) < 2:
            index = random.sample(range(0, index), 1)[0]
            video_name = self.videos[index]
            video = self.labels[video_name]
            track = np.random.choice(list(video.keys()))
            track_info = video[track]
            retrieve_key = video_name.split('/')[-1] + '_' + str(int(track))
            frames = track_info['frames']
        '''
        self.frames_ = self.info_frames_
        _template_frame = np.random.randint(0, len(frames))
        left = max(_template_frame - self.frame_range, 0)
        right = min(_template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[_template_frame]
        search_frame = np.random.choice(search_range)
        query_range = \
            (list(range(len(frames)))[0:_template_frame] +
             list(range(len(frames)))[_template_frame+1:]) if len(frames) > 1 else [0]
        try:
            query_frame = frames[random.sample(query_range, 1)[0]]
        except IndexError:
            print("frames: ", frames, "query_range: ", query_range)
        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame), \
               self.get_image_anno(video_name, track, query_frame), \
               retrieve_key

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class BANDataset(Dataset):
    def __init__(self,):
        super(BANDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create point target
        self.point_target = PointTarget()

        # hard example mining
        if cfg.TRAIN.HNM:
            logger.info("loading index file")
            self.index_file = json.load(open('../../training_dataset/data_list_integrated_all.json', 'r'))
            # build inverse index file
            self.inv_ind_file = self.build_inv_ind_file(self.index_file)
            # load index file
            f = 32768  # vector dimension
            self.u = AnnoyIndex(f, 'angular')
            self.u.load('../../training_dataset/hnm_tree_all.ann')

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def build_inv_ind_file(self, ind_file):
        file_path = '../../training_dataset/inv_ind_all.json'
        if os.path.exists(file_path):
            inv_ind = json.load(open(file_path, 'r'))
        else:
            logger.info('loading inverse index file')
            inv_ind = {}
            for idx in ind_file:
                seq_name = ind_file[idx]['path']
                # if 'VID' not in seq_name:
                #     continue
                if 'LaSOT' in seq_name:
                    _seq_name = seq_name.split('/')[-3]
                    relative_path = seq_name.split('/')[-4] + '/' + seq_name.split('/')[-3] + \
                                    '/' + seq_name.split('/')[-2]
                elif 'VID' in seq_name:
                    _seq_name = seq_name.split('/')[-2]
                    relative_path = seq_name.split('/')[-3] + '/' + seq_name.split('/')[-2]
                elif 'GOT' in seq_name:
                    _seq_name = seq_name.split('/')[-2]
                    relative_path = seq_name.split('/')[-3] + '/' + seq_name.split('/')[-2]
                if _seq_name not in inv_ind:
                    inv_ind[_seq_name] = {}
                    inv_ind[_seq_name]['path'] = relative_path
                    if 'id' not in inv_ind[_seq_name]:
                        inv_ind[_seq_name]['id'] = []
                        inv_ind[_seq_name]['id'].append(idx)
                else:
                    inv_ind[_seq_name]['id'].append(idx)
            with open(file_path, 'w+') as f:
                json.dump(inv_ind, f)
            print('save the inverse index json file to ' + file_path)
        return inv_ind

    def get_hd_samples(self, hn_based_ids):
        based_item = int(random.sample(hn_based_ids, 1)[0])  # based sample's item, from 2 choose 1
        # print(based_item)
        nearest_n = self.u.get_nns_by_item(based_item, 5)  # return a list with items
        nearest = random.sample(nearest_n[1:], 1)  # 1 for re-dect, 4 for mosaic
        hn_samples = []
        # get image
        all_train = self.index_file
        for idx in nearest:
            im_path = all_train[str(idx)]['path']
            im_anno = all_train[str(idx)]['anno']
            # convert box to corner based
            im_anno = [im_anno[0], im_anno[1], im_anno[0] + im_anno[2], im_anno[1] + im_anno[3]]
            if 'VID' in im_path:
                # ------ only for vid -------
                tmp = im_path.split('/')
                tmp[-2] = '_'.join(tmp[-2].split('_')[:-1])
                im_path = '/'.join(tmp)
                # ---------------------------
            hn_samples.append((im_path, im_anno))
        return hn_samples

    def crop_like_SiamFC(self, image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
        def pos_s_2_bbox(pos, s):
            return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]

        def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
            a = (out_sz - 1) / (bbox[2] - bbox[0])
            b = (out_sz - 1) / (bbox[3] - bbox[1])
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
            return crop

        target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
        target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        #z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
        x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
        return x

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()


        if neg:
            if cfg.MULDECT.MULDECT:
                template, query, retrieve_key = dataset.get_positive_pair(index)
            else:
                template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            if cfg.MULDECT.MULDECT:
                template, search, query, retrieve_key = dataset.get_positive_triple(index)
            else:
                template, search, _ = dataset.get_positive_pair(index)

        # get negative support sample
        if cfg.MULDECT.MULDECT:
            query_img = cv2.imread(query[0])
            query_box = self._get_bbox(query_img, query[1])
            query_img, query_bbox = self.search_aug(query_img,
                                                    query_box,
                                                    cfg.TRAIN.SEARCH_SIZE,
                                                    gray=gray)
            query_img = query_img.transpose((2, 0, 1)).astype(np.float32)
            hnm_flag = False
            if cfg.TRAIN.HNM and dataset.name != 'DET' and dataset.name != 'COCO' and \
                    dataset.name != 'YOUTUBEBB':
                hnm_flag = True
            if hnm_flag:
                try:
                    # get hn item ids
                    hn_based_ids = self.inv_ind_file[retrieve_key]['id']
                    # get hn image set(img path and anno)
                    hd_samples = self.get_hd_samples(hn_based_ids)
                    hd_sample_imgs = [cv2.imread(im[0]) for im in hd_samples]
                    hd_sample_bboxes = [b[1] for b in hd_samples]
                    # get imgs
                    hd_sample_imgs = [
                        self.crop_like_SiamFC(img, hd_samples[idx][1], context_amount=0.5, exemplar_size=127, instanc_size=511,
                                              padding=np.mean(img, axis=(0, 1))) for idx, img in enumerate(hd_sample_imgs)]
                    # get bounding box
                    hd_sample_boxes = [self._get_bbox(img, hd_sample_bboxes[idx]) for idx, img in enumerate(hd_sample_imgs)]
                    # augmentation  (choose 1 img from the hnm set, the others for other function (mosaic augmentation))
                    hd_sample_image, hd_sample_box = self.template_aug(hd_sample_imgs[0],
                                                                       hd_sample_boxes[0],
                                                                       cfg.TRAIN.EXEMPLAR_SIZE,
                                                                       gray=gray)
                    # unified name
                    neg_support_img = hd_sample_image
                    neg_support_box = hd_sample_box
                    neg_support_img = neg_support_img.transpose((2, 0, 1)).astype(np.float32)
                    # logger.info('success to select hnm sample !')
                except KeyError:
                    logger.warning('can not find key {}, change to choose random target'.format(retrieve_key))
                    hnm_flag = False
            if not hnm_flag:
                rnd_neg_support = np.random.choice(self.all_dataset).get_random_target()
                rnd_neg_support_image = cv2.imread(rnd_neg_support[0])
                rnd_neg_support_image_bbox = self._get_bbox(rnd_neg_support_image, rnd_neg_support[1])
                # augmentation
                rnd_neg_support_image, rnd_neg_support_bbox = self.template_aug(rnd_neg_support_image,
                                                                        rnd_neg_support_image_bbox,
                                                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                                                        gray=gray)
                # unified name
                neg_support_img = rnd_neg_support_image
                neg_support_box = rnd_neg_support_bbox
                neg_support_img = neg_support_img.transpose((2, 0, 1)).astype(np.float32)

        # use MOSAIC (on search img)
        if cfg.TRAIN.MOSAIC:
            pass

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, template_bbox = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels
        cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        if cfg.MULDECT.MULDECT:
            return {
                    'template': template,
                    'template_bbox': np.array(template_bbox),
                    'search': search,
                    'search_bbox': np.array(bbox),
                    'label_cls': cls,
                    'label_loc': delta,
                    'query_img': query_img,
                    'query_bbox': np.array(query_bbox),
                    'neg_support_image': neg_support_img,
                    'neg_support_bbox': np.array(neg_support_box),
                    'points': self.point_target.points.points,
                   # 'pos': pos
                    }
        else:
            return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'bbox': np.array(bbox)
            }

