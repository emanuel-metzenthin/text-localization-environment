
def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def intersection(bbox1, bbox2):
    left = max(bbox1[0], bbox2[0])
    top = max(bbox1[1], bbox2[1])
    right = min(bbox1[2], bbox2[2])
    bottom = min(bbox1[3], bbox2[3])

    if right < left or bottom < top:
        return 0

    return (right - left) * (bottom - top)

def intersection_over_union(bbox1, bbox2):
    _intersection = intersection(bbox1, bbox2)
    area_1 = area(bbox1)
    area_2 = area(bbox2)

    union = area_1 + area_2 - _intersection

    return _intersection / union
