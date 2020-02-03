import numpy as np
import scipy.stats as st
from PIL import Image, ImageDraw


class ImageMasker:
    """
        This class implements a CPU/GPU method for adding a mask on an array. Think of it as drawing a polygon that is defined
        by a bounding box, given by four points.
    """

    def __init__(self, image, bbox, strategy='fill'):
        self.image = image
        self.bbox = bbox
        self.strategies = {
            'fill': self.fill,
            'cross': self.cross,
            'gauss': self.gauss,
        }
        self.strategy = self.strategies[strategy]

    def fill(self, color=(0, 0, 0)):
        draw = ImageDraw.Draw(self.image)
        draw.rectangle(self.bbox, fill=color)
        return self.image

    def cross(self, color=(0, 0, 0)):
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

    def _gauss_kernel(self, kernel_size, nsig):
        """Returns a 2D Gaussian kernel."""

        x = np.linspace(-nsig, nsig, kernel_size + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d / kern2d.sum()

    def gauss(self, kernel_size=10, nsig=1):
        kernel = self._gauss_kernel(kernel_size, nsig)
        img = np.array(self.image, dtype=np.int32) / 255.0

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
