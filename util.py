import numpy as np

# pt: (x, y)
def inside(pt, rect):
    x, y = pt
    x_ul, y_ul, w, h = rect
    return (x >= x_ul and x <= x_ul + w and y >= y_ul and y <= y_ul + h)


# rect: (x, y, w, h)
# 记录rect1的四个顶点在rect2内部的
def intersect(rect1, rect2):
    pt_intersect = {'ul': rect1[:2],
                'ur': (rect1[0]+rect1[2], rect1[1]),
                'bl': (rect1[0], rect1[1]+rect1[3]),
                'br': (rect1[0]+rect1[2], rect1[1]+rect1[3])}
    
    keep_corner = []
    for name in pt_intersect.keys():
        if inside(pt_intersect[name], rect2):
            keep_corner.append(name)

    return {key: pt_intersect[key] for key in keep_corner}


def opposite_corner(rect, name):
    if name.lower() == 'ul':
        return (rect[0]+rect[2], rect[1]+rect[3])
    elif name.lower() == 'ur':
        return (rect[0], rect[1]+rect[3])
    elif name.lower() == 'bl':
        return (rect[0]+rect[2], rect[1])
    elif name.lower() == 'br':
        return (rect[0], rect[1])
    else:
        print('wrong corner identifier')
        return (-1, -1)


def iou_gt(box1, box2):
    pt_intersect = intersect(box1, box2)
    cnt = len(pt_intersect.keys())
    size_box2 = box2[3] * box2[2]
    if cnt == 0:
        return 0
    elif cnt == 1:
        name = list(pt_intersect.keys())[0]
        pt_corner = opposite_corner(box2, name)
        return np.fabs(pt_corner[0] - pt_intersect[name][0]) * np.fabs(pt_corner[1] - pt_intersect[name][1]) / size_box2
    elif cnt == 2:
        if all(['l' in key for key in pt_intersect.keys()]):
            x_right = box2[0] + box2[2]
            return np.fabs(pt_intersect['ul'][1] - pt_intersect['bl'][1]) * (x_right - pt_intersect['ul'][0]) / size_box2
        if all(['r' in key for key in pt_intersect.keys()]):
            x_left = box2[0]
            return np.fabs(pt_intersect['ur'][1] - pt_intersect['br'][1]) * (pt_intersect['ur'][0] - x_left) / size_box2
        if all(['u' in key for key in pt_intersect.keys()]):
            y_bottom = box2[1] + box2[3]
            return np.fabs(pt_intersect['ul'][0] - pt_intersect['ur'][0]) * (y_bottom - pt_intersect['ul'][1]) / size_box2
        if all(['b' in key for key in pt_intersect.keys()]):
            y_up = box2[1]
            return np.fabs(pt_intersect['bl'][0] - pt_intersect['br'][0]) * (pt_intersect['bl'][1] - y_up) / size_box2
    elif cnt == 4:
        return (pt_intersect['br'][0] - pt_intersect['ul'][0]) * (pt_intersect['br'][1] - pt_intersect['ul'][1]) / size_box2


def measure(pred, gt):
    assert pred.shape == gt.shape
    pred = pred.detach().cpu().numpy()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    pred = pred.astype(np.bool)
    gt = gt.detach().cpu().numpy().astype(np.bool)
    tp = np.count_nonzero(pred * gt)
    fp = np.count_nonzero(pred * ~gt)
    tn = np.count_nonzero(~pred * ~gt)
    fn = np.count_nonzero(~pred * gt)
    
    size = 1
    for i in gt.shape:
        size = size * i
    assert size == (tp + fp + tn + fn)
    return tp, fp, tn, fn


class Measure(object):
    def __init__(self, tp=0, fp=0, tn=0, fn=0):
        self.tp, self.fp, self.tn, self.fn = tp, fp, tn, fn
        self.delta = 1e-4

    def __add__(self, m):
        self.tp += m[0]
        self.fp += m[1]
        self.tn += m[2]
        self.fn += m[3]
        return self

    def __radd__(self, m):
        return self.__add__(m)

    def sum(self):
        return self.tp + self.tn + self.fp + self.fn + self.delta

    def precision(self):
        return (self.tp + self.tn) / self.sum()

    def recall(self):
        return self.tp / (self.sum()-self.tn - self.fp)

    def f1(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

if __name__ == '__main__':
    # 测试通过
    '''
    # 两个矩形相交部分矩形的坐标
    rect2 = [127, 94, 54, 54]
    rect1 = [150, 100, 45, 43]
    print(intersect(rect1, rect2))
    print(iou_gt(rect1, rect2))
    '''
    m1 = Measure()
    m2 = Measure(1, 2, 3, 4)
    m1 += m2