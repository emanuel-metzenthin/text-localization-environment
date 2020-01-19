import gym
from gym import spaces
import random
from gym.utils import seeding
from chainer.backends import cuda
from PIL import Image, ImageDraw
from PIL.Image import LANCZOS, MAX_IMAGE_PIXELS
import numpy as np
from text_localization_environment.ImageMasker import ImageMasker


class TextLocEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'box']}

    DURATION_PENALTY = 0.03
    HISTORY_LENGTH = 10
    # ⍺: factor relative to the current box size that is used for every transformation action
    ALPHA = 0.2
    # η: Reward of the trigger action
    ETA = 7.0
    # p: Probability for masking a bounding box in a new observation (applied separately to boxes 0..N-1 during premasking)
    P_MASK = 0.5
    # Reward for next image trigger action
    ETA2 = 8.0

    def __init__(self, image_paths, true_bboxes, gpu_id=-1,
        playout_episode=False, premasking=True, mode='train',
        max_steps_per_image=200
    ):
        """
        :param image_paths: The paths to the individual images
        :param true_bboxes: The true bounding boxes for each image
        :param gpu_id: The ID of the GPU to be used. -1 if CPU should be used instead
        :type image_paths: String or list
        :type true_bboxes: numpy.ndarray
        :type gpu_id: int
        """
        self.action_space = spaces.Discrete(10)
        self.action_set = {0: self.right,
                           1: self.left,
                           2: self.up,
                           3: self.down,
                           4: self.bigger,
                           5: self.smaller,
                           6: self.fatter,
                           7: self.taller,
                           8: self.trigger,
                           9: self.next_image_trigger
                           }
        # 224*224*3 (RGB image) + 9 * 10 (on-hot-enconded history) = 150618
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=256, shape=(224,224,3)), spaces.Box(low=0,high=1,shape=(10,9))])
        self.gpu_id = gpu_id
        if type(image_paths) is not list: image_paths = [image_paths]
        self.image_paths = image_paths
        self.true_bboxes = [[TextLocEnv.to_standard_box(b) for b in bboxes] for bboxes in true_bboxes]
        # Determines whether the agent is training or testing
        # Optimizations can be applied during training that are not allowed for testing
        self.mode = mode
        # Whether IoR markers will be placed upfront after loading the image
        self.premasking = premasking
        # Whether an episode terminates after a single trigger or is played out until the end
        self.playout_episode = playout_episode
        # Episodes will be terminated automatically after reaching max steps
        self.max_steps_per_image = max_steps_per_image

        self.seed()

        # Image for the current episode
        self.episode_image = Image.new("RGB", (256, 256))

        # Ground truth bounding boxes for the current episode image
        self.episode_true_bboxes = None
        # Predicted bounding boxes for the current episode image
        self.episode_pred_bboxes = None
        # IoU values for each trigger in the current episode
        self.episode_trigger_ious = None
        # List of indices of masked bounding boxes for the current episode image
        self.episode_masked_indices = []
        # The agent's current window represented as [x0, y0, x1, y1]
        self.bbox = None

        self.num_detected_texts = 0
        # For registering a handler that will be executed once after a step
        self.post_step_handler = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Execute an action and return
            state - the next state,
            reward - the reward,
            done - whether a terminal state was reached,
            info - any additional info"""
        assert self.action_space.contains(action), "%r (%s) is an invalid action" % (action, type(action))

        self.current_step += 1

        self.action_set[action]()

        reward = self.calculate_reward(action)
        self.max_iou = max(self.iou, self.max_iou)

        self.history.insert(0, self.to_one_hot(action))
        self.history.pop()

        self.state = self.compute_state()

        # Execute and remove any registered post-step handler
        if self.post_step_handler is not None:
            self.post_step_handler()
            self.post_step_handler = None

        # Terminate episode after reaching step limit (if set)
        # Prevents the agent from running into an infinite loop
        if self.max_steps_per_image != -1 and self.current_step >= self.max_steps_per_image:
            self.done = True

        return self.state, reward, self.done, {}

    def calculate_reward(self, action):
        reward = 0

        if self.action_set[action] == self.next_image_trigger:
            if self.evaluate_detected_instances() < 1.0:
                return -10
            else:
                return 10

        if self.action_set[action] == self.trigger:
            reward = 10 * self.ETA * self.iou - (self.current_step * self.DURATION_PENALTY)
        else:
            self.iou = self.compute_best_iou()

        return reward

    def create_empty_history(self):
        flat_history = np.repeat([False], self.HISTORY_LENGTH * self.action_space.n)
        history = flat_history.reshape((self.HISTORY_LENGTH, self.action_space.n))

        return history.tolist()

    @staticmethod
    def to_standard_box(bbox):
        """
        Transforms a given bounding box into a standardized representation.

        :param bbox: Bounding box given as [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
        :return: Bounding box represented as [x0, y0, x1, y1]
        """
        from typing import Iterable
        if isinstance(bbox[0], Iterable):
            bbox = [xy for p in bbox for xy in p]
        return bbox

    def create_ior_mark(self, bbox):
        """
        Creates an IoR (inhibition of return) mark that crosses out the given bounding box.
        This is necessary to find multiple objects within one image

        :param bbox: Bounding box given as [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
        """
        bbox = self.to_standard_box(bbox)
        masker = ImageMasker(self.episode_image, bbox)
        self.episode_image = masker.mask()

    @property
    def episode_true_bboxes_unmasked(self):
        """
        Returns the bounding boxes in the current episode image that are not masked.
        """
        bboxes_unmasked = []

        for index, bbox in enumerate(self.episode_true_bboxes):
            is_masked = index in self.episode_masked_indices
            if not is_masked:
                bboxes_unmasked.append(bbox)

        return bboxes_unmasked

    def compute_best_iou(self):
        max_iou = 0

        # Only consider boxes that have not been masked yet
        # Ensures that the agent is not rewarded for visiting the same location
        for box in self.episode_true_bboxes_unmasked:
            max_iou = max(max_iou, self.compute_iou(box))

        return max_iou

    def compute_iou(self, other_bbox):
        """Computes the intersection over union of the argument and the current bounding box."""
        intersection = self.compute_intersection(other_bbox)

        area_1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area_2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
        union = area_1 + area_2 - intersection

        return intersection / union

    def compute_intersection(self, other_bbox):
        left = max(self.bbox[0], other_bbox[0])
        top = max(self.bbox[1], other_bbox[1])
        right = min(self.bbox[2], other_bbox[2])
        bottom = min(self.bbox[3], other_bbox[3])

        if right < left or bottom < top:
            return 0

        return (right - left) * (bottom - top)

    def up(self):
        self.adjust_bbox(np.array([0, -1, 0, -1]))

    def down(self):
        self.adjust_bbox(np.array([0, 1, 0, 1]))

    def left(self):
        self.adjust_bbox(np.array([-1, 0, -1, 0]))

    def right(self):
        self.adjust_bbox(np.array([1, 0, 1, 0]))

    def bigger(self):
        self.adjust_bbox(np.array([-0.5, -0.5, 0.5, 0.5]))

    def smaller(self):
        self.adjust_bbox(np.array([0.5, 0.5, -0.5, -0.5]))

    def fatter(self):
        self.adjust_bbox(np.array([0, 0.5, 0, -0.5]))

    def taller(self):
        self.adjust_bbox(np.array([0.5, 0, -0.5, 0]))

    def trigger(self):
        self.num_detected_texts += 1
        self.episode_pred_bboxes.append(self.bbox)
        # IoU values are only updated after trigger action is executed
        # Therefore we need to track them lazily
        self.post_step_handler = self._register_trigger_iou

        if not self.playout_episode:
            # Terminate episode after first trigger action
            self.done = True
            return

        if self.mode == 'train':
            if len(self.episode_true_bboxes_unmasked) > 0:
                index, bbox = self.closest_unmasked_true_bbox()
                self.create_ior_mark(bbox)
                self.episode_masked_indices.append(index)
        else:
            self.create_ior_mark(self.bbox)

        self.reset_bbox()

    def _register_trigger_iou(self):
        self.episode_trigger_ious.append(self.iou)

    def closest_unmasked_true_bbox(self):
        max_iou = None
        best_box = None
        best_box_index = None

        for index, box in enumerate(self.episode_true_bboxes):
            if index in self.episode_masked_indices:
                continue
            iou = self.compute_iou(box)
            if not max_iou or iou > max_iou:
                max_iou = iou
                best_box = box
                best_box_index = index

        return (best_box_index, best_box)

    # trigger that should be used when all text instances have been detected by the agent
    def next_image_trigger(self):
        self.done = True
        # self.reset()

    @staticmethod
    def box_size(box):
        width = box[2] - box[0]
        height = box[3] - box[1]

        return width * height

    def adjust_bbox(self, directions):
        ah = round(self.ALPHA * (self.bbox[3] - self.bbox[1]))
        aw = round(self.ALPHA * (self.bbox[2] - self.bbox[0]))

        adjustments = np.array([aw, ah, aw, ah])
        delta = directions * adjustments

        new_box = self.bbox + delta

        if self.box_size(new_box) < MAX_IMAGE_PIXELS:
            self.bbox = new_box

    def reset_bbox(self):
        self.bbox = np.array([0, 0, self.episode_image.width, self.episode_image.height])

    def reset(self, image_index=None):
        """Reset the environment to its initial state (the bounding box covers the entire image)"""
        self.num_detected_texts = 0

        self.history = self.create_empty_history()
        if self.episode_image is not None:
            self.episode_image.close()

        if image_index is None:
            # Pick random next image if not specified otherwise
            image_index = self.np_random.randint(len(self.image_paths))
        self.episode_image = Image.open(self.image_paths[image_index])
        self.episode_true_bboxes = self.true_bboxes[image_index]

        if self.episode_image.mode != 'RGB':
            self.episode_image = self.episode_image.convert('RGB')

        self.episode_masked_indices = []

        # Mask bounding boxes randomly with probability P_MASK
        if self.mode == 'train' and self.premasking:
            num_unmasked = self.episode_num_true_bboxes
            for idx, box in enumerate(self.episode_true_bboxes):
                # Ensure at least 1 non-masked instance per observation
                if num_unmasked > 1 and np.random.random() <= self.P_MASK:
                    self.create_ior_mark(box)
                    self.episode_masked_indices.append(idx)
                    num_unmasked -= 1

        self.episode_pred_bboxes = []
        self.episode_trigger_ious = []
        self.current_step = 0
        self.reset_bbox()
        self.state = self.compute_state()
        self.done = False
        self.iou = self.compute_best_iou()
        self.max_iou = self.iou

        return self.state

    def render(self, mode='human', return_as_file=False, include_true_bboxes=False):
        """Render the current state"""

        image = self.episode_image
        if include_true_bboxes:
            image = self.episode_image_with_true_bboxes

        if mode == 'human':
            copy = image.copy()
            draw = ImageDraw.Draw(copy)
            draw.rectangle(self.bbox.tolist(), outline=(255, 255, 255))
            if return_as_file:
                return copy
            copy.show()
            copy.close()
        elif mode is 'box':
            # Renders what the agent currently sees
            # i.e. the section of the image covered by the agent's current window (warped to standard size)
            warped = self.get_warped_bbox_contents()
            if return_as_file:
                return warped
            warped.show()
            warped.close()
        elif mode is 'rgb_array':
            copy = image.copy()
            draw = ImageDraw.Draw(copy)
            draw.rectangle(self.bbox.tolist(), outline=(255, 255, 255))
            return np.array(copy)
        else:
            super(TextLocEnv, self).render(mode=mode)

    def get_warped_bbox_contents(self):
        cropped = self.episode_image.crop(self.bbox)
        return cropped.resize((224, 224), LANCZOS)

    def compute_state(self):
        warped = self.get_warped_bbox_contents()
        return (np.array(warped, dtype=np.float32), np.array(self.history))

    def to_one_hot(self, action):
        line = np.zeros(self.action_space.n, np.bool)
        line[action] = 1

        return line

    @property
    def episode_image_with_true_bboxes(self, true_bbox_color=(255, 0, 0)):
        if not self.episode_true_bboxes:
            return self.episode_image

        copy = self.episode_image.copy()
        draw = ImageDraw.Draw(copy)
        for box in self.episode_true_bboxes:
            draw.rectangle(box, outline=true_bbox_color)
        return copy

    @property
    def episode_num_true_bboxes(self):
        """Number of bounding boxes available in the current episode image."""
        if not self.episode_true_bboxes:
            return None
        return len(self.episode_true_bboxes)

    def evaluate_detected_instances(self):
        return (self.num_detected_texts / len(self.episode_true_bboxes))
