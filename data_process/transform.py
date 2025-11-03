from torchvision import transforms as T
import random, math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

import random
import numpy as np
from PIL import Image

class RandomOccludeImage:
    """
    ¸´ÏÖ CrossViT-ReID µÄ 8 ÖÖÕÚµ²ÐÎÌ¬£¨ºÚÉ«¾ØÐÎÕÚµ²£©£¬
    ÊÊÅä LTCC µÄµ¥Ö¡ RGB Í¼Ïñ (H, W, 3)
    """
    def __init__(self, p=0.5, occlude_bounds=(0.25, 0.5), color=(0, 0, 0)):
        self.p = p
        self.occlude_bounds = occlude_bounds  # ±ÈÀý·¶Î§£¬Èç(0.25, 0.5)
        self.color = color

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        x = np.array(img).copy()  # H, W, 3
        h, w = x.shape[:2]
        mode = random.randint(1, 8)
        # Èç¹û¹Ø±ÕÕÚµ²£¬Ö±½Ó·µ»Ø
        if self.occlude_bounds[0] == 0.0 and self.occlude_bounds[1] == 0.0:
            return x
        # if mode==0:
        #     pass
        if mode in [1, 2, 3, 4]:
            rh = random.uniform(*self.occlude_bounds)
            rw = random.uniform(*self.occlude_bounds)
            oh = int(h * rh)
            ow = int(w * rw)

            if mode == 1:  # ×óÉÏ
                x[0:oh, 0:ow, :] = self.color
            elif mode == 2:  # ÓÒÉÏ
                x[0:oh, w-ow:w, :] = self.color
            elif mode == 3:  # ×óÏÂ
                x[h-oh:h, 0:ow, :] = self.color
            else:            # ÓÒÏÂ
                x[h-oh:h, w-ow:w, :] = self.color

        # ÉÏ/ÏÂ °ëÕÚµ²£ºËæ»ú¸ß¶È
        elif mode in [5, 6]:
            rh = random.uniform(*self.occlude_bounds)
            oh = int(h * rh)
            if mode == 5:    # ÉÏ
                x[0:oh, :, :] = self.color
            else:            # ÏÂ
                x[h-oh:h, :, :] = self.color

        # ×ó/ÓÒ °ëÕÚµ²£ºËæ»ú¿í¶È
        else:  # 7, 8
            rw = random.uniform(*self.occlude_bounds)
            ow = int(w * rw)
            if mode == 7:    # ×ó
                x[:, 0:ow, :] = self.color
            else:            # ÓÒ
                x[:, w-ow:w, :] = self.color

        return Image.fromarray(x)

# ÓÃ ImageNet ¾ùÖµ×÷Îª"Ìî³äÉ«"£¬±ÜÃâºÚÉ«¿é´øÀ´·Ö²¼Í»±ä
IM_MEAN_255 = (int(0.485*255), int(0.456*255), int(0.406*255))

def get_transform(args):
    transform_train = T.Compose([
        T.Resize((args.height, args.width)),
        # RandomOccludeImage(p=0.5, occlude_bounds=(0.25, 0.5)),
        T.RandomHorizontalFlip(p=args.horizontal_flip_pro),
        T.Pad(padding=args.pad_size),
        T.RandomCrop((args.height, args.width)),
        # RandomOccludeImage(
        #     p=0.5,  # ´Ó 0.3 Æð²½£¬¹Û²ìÊÕÁ²ÔÙµ÷
        #     occlude_bounds=(0.25, 0.5),  # Ãæ»ý 15%~35% ¸üÎÈ
        #     # color=IM_MEAN_255  # ÓÃÊý¾Ý¾ùÖµ¶ø·Ç´¿ºÚ
        # ),

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=args.random_erasing_pro, mean=[0.0, 0.0, 0.0])
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test