# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
python main.py --gpu_devices 1 --dataset Occluded_Duke --dataset_root /home/user3/ZSJ/data --dataset_filename Occluded_Duke --save_dir Occluded_Duke_save_10 --save_checkpoint

python main.py --gpu_devices 0 --dataset Occluded_Duke --dataset_root /home/user3/ZSJ/data --dataset_filename Occluded_Duke --save_dir Occluded_Duke_save_20 --save_checkpoint
"""

import glob
import re
import urllib
import zipfile

import os.path as osp


from dataset.base_image_dataset import BaseImageDataset

# encoding: utf-8
import glob
import re
import os.path as osp

from dataset.base_image_dataset import BaseImageDataset


class Occluded_Duke(BaseImageDataset):
    """
    Occluded-DukeMTMC-reID / DukeMTMC-reID (Occluded Duke)
    Reference:
      - Ristani et al. ECCVW 2016 (DukeMTMC)
      - Zheng et al. ICCV 2017 (DukeMTMC-reID)

    Dataset statistics (typical Occluded-Duke):
      # cameras: 8
      # images:   bounding_box_train / query / bounding_box_test
    """

    def __init__(self, dataset_root='data', dataset_filename='Occluded_Duke', verbose=True, **kwargs):
        super(Occluded_Duke, self).__init__()
        self.dataset_dir = osp.join(dataset_root, dataset_filename)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # Óë Market1501 ±£³ÖÒ»ÖÂµÄ´æÔÚÐÔ¼ì²é
        self.check_before_run(required_files=[self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir])

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Occluded_Duke loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # ÎÄ¼þÃûÐÎÈç 0002_c2_XXXX.jpg
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            m = pattern.search(osp.basename(img_path))
            if m is None:
                # Ìø¹ý²»·ûºÏÃüÃû¹æ·¶µÄÎÄ¼þ
                continue
            pid, _ = map(int, m.groups())
            if pid == -1:
                # Óë Market ±£³ÖÒ»ÖÂ£¬ºöÂÔ junk Í¼Æ¬
                continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            m = pattern.search(osp.basename(img_path))
            if m is None:
                continue
            pid, camid = map(int, m.groups())
            if pid == -1:
                continue
            # Duke ÓÐ 8 ¸öÏà»ú
            assert 1 <= camid <= 8, f"Unexpected camid {camid} in {img_path}"
            camid -= 1  # ´Ó 0 ¿ªÊ¼
            if relabel:
                pid = pid2label[pid]
            # **Óë Market ¶ÔÆë£º·µ»ØÈýÔª×é (img_path, pid, camid)**
            dataset.append((img_path, pid, camid))

        return dataset
