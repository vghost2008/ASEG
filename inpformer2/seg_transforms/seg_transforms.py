import random
import copy
import numbers
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop as TV_RandomResizedCrop, \
                                   Resize as TV_Resize, \
                                   ColorJitter as TV_ColorJitter, \
                                   ToTensor as TV_ToTensor, \
                                   Normalize as TV_Normalize


class BaseSegTransform(object):
    def __init__(self) -> None:
        pass

    def apply_image(self, image):
        return image

    def apply_semantic(self, mask):
        return mask

    def __call__(self, data):
        assert len(data) == 2, 'data length must be 2, (image, mask)'
        image, mask = data
        t_image = self.apply_image(image)
        if mask is not None:
            t_mask = self.apply_semantic(mask)
        else:
            t_mask = mask
        return (t_image, t_mask)
    

class Compose(object):
    def __init__(self, transforms:List[BaseSegTransform]) -> None:
        self.transforms = transforms if isinstance(transforms, list) else [transforms]
        pass

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    

class Resize(BaseSegTransform):
    """
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
    """
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True) -> None:
        super().__init__()
        self._applyer = TV_Resize(size, interpolation, max_size, antialias)
    
    def apply_image(self, image):
        return self._applyer(image)

    def apply_semantic(self, mask):
        return self._applyer(mask)


class RandomResizedCrop(BaseSegTransform):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
            ) -> None:
        super().__init__()
        self._applyer = TV_RandomResizedCrop(size, scale, ratio, interpolation, antialias)
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio
        self._t_params = None
    
    def apply_image(self, image):
        self._t_params = self._applyer.get_params(image, self.scale, self.ratio)
        i, j, h, w = self._t_params
        return F.resized_crop(image, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

    def apply_semantic(self, mask):
        assert self._t_params is not None
        i, j, h, w = self._t_params
        return F.resized_crop(mask, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
    

class RandomVerticalFlip(BaseSegTransform):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self._t_p = -1

    def apply_image(self, image):
        self._t_p = torch.rand(1)
        if self._t_p < self.p:
            return F.hflip(image)
        return image

    def apply_semantic(self, mask):
        assert self._t_p != -1
        if self._t_p < self.p:
            return F.hflip(mask)
        return mask
    

class RandomHorizontalFlip(BaseSegTransform):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self._t_p = -1

    def apply_image(self, image):
        self._t_p = torch.rand(1)
        if self._t_p < self.p:
            return F.vflip(image)
        return image

    def apply_semantic(self, mask):
        assert self._t_p != -1
        if self._t_p < self.p:
            return F.vflip(mask)
        return mask

    
class ColorJitter(BaseSegTransform):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
            ) -> None:
        super().__init__()    
        self._applyer = TV_ColorJitter(brightness, contrast, saturation, hue)

    def apply_image(self, image):
        return self._applyer(image)


class ToTensor(BaseSegTransform):
    def __init__(self) -> None:
        super().__init__()
        self._applyer = TV_ToTensor()
    
    def apply_image(self, image):
        return self._applyer(image)

    def apply_semantic(self, mask):
        return self._applyer(mask)

class Normalize(BaseSegTransform):
    def __init__(self, mean, std, inplace=False) -> None:
        super().__init__()
        self._applyer = TV_Normalize(mean, std, inplace)
    
    def apply_image(self, image):
        return self._applyer(image)

    