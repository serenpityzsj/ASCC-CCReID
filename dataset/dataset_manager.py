from __future__ import print_function, absolute_import

from dataset.LTCC import LTCC
from dataset.CelebreID import CelebreID
from dataset.occ_duke import Occluded_Duke
from dataset.occ_reid import OccludedREID

__img_factory = {
    'ltcc': LTCC,
    'celeb': CelebreID,
    'Occluded_Duke':Occluded_Duke,
    'OccludedREID':OccludedREID
}


def get_dataset(args):
    name = args.dataset
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))

    dataset = __img_factory[name](dataset_root=args.dataset_root, dataset_filename=args.dataset_filename)
    return dataset