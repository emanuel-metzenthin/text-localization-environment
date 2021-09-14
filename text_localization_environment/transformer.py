import numpy as np
from PIL.Image import MAX_IMAGE_PIXELS
from abc import ABC, abstractmethod
from .utils import box_size, box_area, scale_bboxes


class BBoxTransformer(ABC):
     # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = None

    def __init__(self):
        self.bbox = None

    @property
    @abstractmethod
    def action_set(self):
        """
        Mapping from action indices (predicted by the agent) to action methods.
        Note: Only provide actions that transform the agent's bounding box here
        """
        pass

    def reset(self, max_width, max_height):
        self.bbox = np.array([0, 0, max_width, max_height])

    def __len__(self):
        return len(self.action_set)

    def _adjust_bbox(self, directions):
        width, height = box_size(self.bbox)
        ah = round(self.ALPHA * height)
        aw = round(self.ALPHA * width)

        adjustments = np.array([aw, ah, aw, ah])
        delta = directions * adjustments

        new_box = self.bbox + delta

        if box_area(new_box) < MAX_IMAGE_PIXELS:
            self.bbox = new_box


class BaseBBoxTransformer(BBoxTransformer):
     # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = 0.2

    @property
    def action_set(self):
        return {
            0: self.right,
            1: self.left,
            2: self.up,
            3: self.down,
            4: self.bigger,
            5: self.smaller,
            6: self.fatter,
            7: self.taller,
        }

    def up(self):
        self._adjust_bbox(np.array([0, -1, 0, -1]))

    def down(self):
        self._adjust_bbox(np.array([0, 1, 0, 1]))

    def left(self):
        self._adjust_bbox(np.array([-1, 0, -1, 0]))

    def right(self):
        self._adjust_bbox(np.array([1, 0, 1, 0]))

    def bigger(self):
        self._adjust_bbox(np.array([-0.5, -0.5, 0.5, 0.5]))

    def smaller(self):
        self._adjust_bbox(np.array([0.5, 0.5, -0.5, -0.5]))

    def fatter(self):
        self._adjust_bbox(np.array([0, 0.5, 0, -0.5]))

    def taller(self):
        self._adjust_bbox(np.array([0.5, 0, -0.5, 0]))

    def get_opposite_action(self, action):
        if action % 2 == 0:
            return action + 1
        else:
            return action - 1


class WangBBoxTransformer(BBoxTransformer):
     # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = 0.18888

    @property
    def action_set(self):
        return {
            # Note: tl_stop_br_stop is covered by trigger (not moving)
            0: self.tl_stop_br_left,
            1: self.tl_stop_br_up,
            2: self.tl_right_br_stop,
            3: self.tl_right_br_left,
            4: self.tl_right_br_up,
            5: self.tl_down_br_stop,
            6: self.tl_down_br_left,
            7: self.tl_down_br_up,
        }

    def tl_stop_br_left(self):
        self._adjust_bbox(np.array([0, 0, -1, 0]))

    def tl_stop_br_up(self):
        self._adjust_bbox(np.array([0, 0, 0, -1]))

    def tl_right_br_stop(self):
        self._adjust_bbox(np.array([1, 0, 0, 0]))

    def tl_right_br_left(self):
        self._adjust_bbox(np.array([1, 0, -1, 0]))

    def tl_right_br_up(self):
        self._adjust_bbox(np.array([1, 0, 0, -1]))

    def tl_down_br_stop(self):
        self._adjust_bbox(np.array([0, 1, 0, 0]))

    def tl_down_br_left(self):
        self._adjust_bbox(np.array([0, 1, -1, 0]))

    def tl_down_br_up(self):
        self._adjust_bbox(np.array([0, 1, 0, -1]))


def create_bbox_transformer(name):
    transformers = {
        'base': BaseBBoxTransformer,
        'wang': WangBBoxTransformer,
    }
    return transformers[name]()
