
def box_size(box):
    """
    :param bbox: Bounding box given as [x0, y0, x1, y1]
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width, height

def box_area(box):
    """
    :param bbox: Bounding box given as [x0, y0, x1, y1]
    """
    width, height = box_size(box)
    return width * height

def scale_bboxes(bboxes, image_size, factor_w, factor_h):
    """
    Scales up the bounding boxes by the given factor relative to their size
    while respecting image boundaries.

    :param bboxes: list of bounding box given as [x0, y0, x1, y1]
    :param image_size: (width,height) tuple for image boundary
    :param factor: scaling factor relative to current bbox size
    """
    scaled_bboxes = []
    max_width, max_height = image_size
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        width, height = box_size(bbox)
        n_x0 = max(x0 - factor_w * width / 2, 0)
        n_y0 = max(y0 - factor_h * height / 2, 0)
        n_x1 = min(x1 + factor_w * width / 2, max_width)
        n_y1 = min(y1 + factor_h * height / 2, max_height)
        scaled_bbox = [n_x0, n_y0, n_x1, n_y1]
        scaled_bboxes.append(scaled_bbox)
    return scaled_bboxes
