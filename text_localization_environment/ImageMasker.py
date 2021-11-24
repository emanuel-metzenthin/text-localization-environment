import numpy as np
import scipy.stats as st
from PIL import Image, ImageDraw


class ImageMasker:
    """
        This class implements a CPU/GPU method for adding a mask on an array. Think of it as drawing a polygon that is defined
        by a bounding box, given by four points.
    """
    def __init__(self, image, bbox, strategy):
        self.image = image
        self.bbox = bbox
        self.strategies = {
            'fill': self.fill,
            'cross': self.cross,
            'gauss': self.gauss,
            'noise': self.noise,
            'average': self.average,
            'alpha': self.alpha,
        }
        self.strategy = self.strategies[strategy]

    def fill(self, color=(0, 0, 0)):
        draw = ImageDraw.Draw(self.image)
        p0 = (self.bbox[0], self.bbox[1])
        p1 = (self.bbox[2], self.bbox[3])
        draw.rectangle([p0, p1], fill=color)
        return self.image

    def cross(self, color=0):
        draw = ImageDraw.Draw(self.image)

        x0, y0 = self.bbox[0], self.bbox[1]
        x1, y1 = self.bbox[2], self.bbox[3]
        ym = y0 + (y1 - y0) / 2
        xm = x0 + (x1 - x0) / 2

        cross_height = (y1 - y0) / 2
        cross_width = (x1 - x0) / 2

        draw.rectangle([(x0, ym - cross_height / 2), (x1, ym + cross_height / 2)], fill=color)
        draw.rectangle([(xm - cross_width / 2, y0), (xm + cross_width / 2, y1)], fill=color)

        return self.image

    def average(self):
        radius_x = (self.bbox[2] - self.bbox[0]) * 0.3
        radius_y = (self.bbox[3] - self.bbox[1]) * 0.3
        crop_above = [self.bbox[0] - radius_x, self.bbox[1] - radius_y, self.bbox[2] + radius_x, self.bbox[1]]
        crop_right = [self.bbox[2], self.bbox[1] - radius_y, self.bbox[2] + radius_x, self.bbox[3] + radius_y]
        crop_bottom = [self.bbox[0] - radius_x, self.bbox[3], self.bbox[2] + radius_x, self.bbox[3] + radius_y]
        crop_left = [self.bbox[0] - radius_x, self.bbox[1] - radius_y, self.bbox[0], self.bbox[1]]

        crops = [crop_above, crop_right, crop_bottom, crop_left]
        avg_lum = 0
        avg_color = (0, 0, 0)

        for crop in crops:
            if crop[2] - crop[0] > 0 and crop[3] - crop[1] > 0:
                crop = np.array(self.image.crop(crop))
                if crop.size > 1:
                    avg_lum += int(crop.mean(axis=0).mean(axis=0)[0])
                # avg_color += np.array(self.image.crop(crop).convert("RGB")).mean(axis=0).mean(axis=0).astype(int)

        avg_lum /= 4
        # avg_color = (avg_color / 4).astype(int)

        return self.cross(color=int(avg_lum))

    def alpha(self):
        return self.cross(color=(0, 0, 0, 0))

    def noise(self):
        self.bbox = list(map(int, self.bbox))
        x0, y0 = self.bbox[0], self.bbox[1]
        x1, y1 = self.bbox[2], self.bbox[3]
        ym = y0 + (y1 - y0) / 2
        xm = x0 + (x1 - x0) / 2
        cross_height = int((y1 - y0) / 2)
        cross_width = int((x1 - x0) / 2)

        img = np.array(self.image)
        y0_slice = max(0, min(self.image.height, int(ym - cross_height / 2)))
        y1_slice = max(0, min(self.image.height, int(ym + cross_height / 2)))
        x0_slice = max(0, min(self.image.width, int(xm - cross_height / 2)))
        x1_slice = max(0, min(self.image.width, int(xm + cross_height / 2)))
        x0 = max(0, min(self.image.width, x0))
        x1 = max(0, min(self.image.width, x1))
        y0 = max(0, min(self.image.height, y0))
        y1 = max(0, min(self.image.height, y1))

        img[y0_slice:y1_slice, x0:x1, :3] = np.random.rand(y1_slice-y0_slice, x1-x0, 3) * 255
        img[y0:y1, x0_slice:x1_slice, :3] = np.random.rand(y1-y0, x1_slice-x0_slice, 3) * 255

        self.image = Image.fromarray(img)

        return self.image

    def _gauss_kernel(self, kernel_size, nsig):
        """Returns a 2D Gaussian kernel."""

        x = np.linspace(-nsig, nsig, kernel_size + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d / kern2d.sum()

    def gauss(self, kernel_size=10, nsig=1):
        kernel = self._gauss_kernel(kernel_size, nsig)
        img = np.array(self.image.convert("RGB"), dtype=np.int32) / 255.0
        self.bbox = list(map(int, self.bbox))
        for y in range(self.bbox[1], self.bbox[3]):
            for x in range(self.bbox[0], self.bbox[2]):
                acc = np.array([0.0, 0.0, 0.0])

                for k_y, k_row in enumerate(kernel):
                    for k_x, _ in enumerate(k_row):
                        y_ = y + k_y - 1
                        x_ = x + k_x - 1
                        if y_ >= 0 and y_ < len(img) and x_ >= 0 and x_ < len(img[y_]):
                            acc += kernel[k_y][k_x] * img[y_][x_]

                img[y][x] = acc

        return Image.fromarray((img * 255.0).astype(np.uint8))

    def mask(self):
        return self.strategy()
