import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as functional
import torchvision.transforms.transforms as transforms
from PIL import Image
import util

class BBox():
    def __init__(self, data):
        self.x = int(data[0])
        self.y = int(data[1])
        self.width = int(data[2])
        self.height = int(data[3])
    
    def to_array(self):
        return [self.x, self.y, self.width, self.height]

    def update(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

class LabelData():
    def __init__(self, label, root='MAFA/images/'):
        self.root = root
        self.filename = os.path.join(root, label[0])
        #self.img = cv2.imread(root+self.filename, -1)
        try:
            self.face_box = BBox(label[1: 5])
            self.eye_pos = label[5: 9].astype(int)
            self.occ_box = BBox(label[9: 13])
            self.occ_type = int(label[13])
            self.occ_degree = int(label[14])
            self.gender = label[15]
            self.race = label[16]
            self.orientation = label[17]
            self.translate_orientation()
            self.glasses_box = BBox(label[18:22])
        except ValueError:
            print('Error Label:', label)
        
    def translate_orientation(self):
        if self.orientation == '1':
            self.orientation = 'left'
        elif self.orientation == '2':
            self.orientation = 'left_frontal'
        elif self.orientation == '3':
            self.orientation = 'frontal'
        elif self.orientation == '4':
            self.orientation = 'right_frontal'
        elif self.orientation == '5':
            self.orientation = 'right'
            
    def translate_occ_type(self):
        if self.occ_type == 1:
            self.occ_type = 'simple'
        elif self.occ_type == 2:
            self.occ_type = 'complex'
        elif self.occ_type == 3:
            self.occ_type = 'human_body'
    
    def face_area(self):
        return self.img[self.face_box.y: self.face_box.y+self.face_box.height,
                        self.face_box.x: self.face_box.x+self.face_box.width, ...]


def parse_anno(file):
    lines = file.readlines()
    anno = np.empty((len(lines), 22), dtype=object)
    for i, line in enumerate(lines):
        anno[i, :] = line.split(' ')[:-1]
    return anno


def square(box, w, h):
    pad = abs(box.height - box.width)
    cut = 0
    x_l, x_r, y_u, y_b = box.x, box.x+box.width, box.y, box.y+box.height
    if box.width > box.height:
        # cut = max(0, box.height + pad - h)
        y_u -= pad//2
        y_b += (pad-pad//2)
    else:
        # cut = max(0, box.width + pad - w)
        x_l -= pad//2
        x_r += (pad-pad//2)
    assert (y_b-y_u) == (x_r-x_l)
    
    x_move = max(-x_l, 0) + min(w-x_r, 0) if x_r - x_l <= w else 0#x_l < 0 or x_r > w
    y_move = max(-y_u, 0) + min(h-y_b, 0) if y_b - y_u <= h else 0
    return (x_l+x_move, y_u+y_move, x_r-x_l, y_b-y_u)


class Faceset(data.Dataset):
    def __init__(self, anno_file, image_folder, in_size=128, test_mode=False, img_ckpt=None):
        assert anno_file.endswith('.txt')
        with open(anno_file, 'r') as f:
            self.anno = parse_anno(f)
        self.image_folder = image_folder
        self.in_size = in_size
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
        self.test_mode = test_mode
        self.img_ckpt = img_ckpt
        col_idx = 2 * (-0.5 + np.reshape(list(range(in_size)) * in_size, (in_size, in_size)).astype(np.float32) / (in_size-1))
        row_idx = col_idx.transpose()
        self.coord_channel = torch.cat([torch.Tensor(col_idx[np.newaxis,]), 
                                        torch.Tensor(row_idx[np.newaxis,])], 0)

    def __len__(self):
        return self.anno.shape[0]
    
    def __getitem__(self, idx):
        label = LabelData(self.anno[idx, :], self.image_folder)
        img = Image.open(label.filename).convert('RGB')
        width, height = img.size
        label.occ_box.x += label.face_box.x
        label.occ_box.y += label.face_box.y
        
        fo_x = min(label.face_box.x, label.occ_box.x)
        fo_y = min(label.face_box.y, label.occ_box.y)
        fo_w = max(label.face_box.x + label.face_box.width, label.occ_box.x + label.occ_box.width) - fo_x
        fo_h = max(label.face_box.y + label.face_box.height, label.occ_box.y + label.occ_box.height) - fo_y
        face_and_occ = BBox([fo_x, fo_y, fo_w, fo_h])
        face_and_occ = BBox(square(face_and_occ, width, height))
        # pad img if corners out of range
        padding = (0 if face_and_occ.x >= 0 else int(-face_and_occ.x), \
                    0 if face_and_occ.y >= 0 else int(-face_and_occ.y), \
                    0 if face_and_occ.x + face_and_occ.width < width else int(face_and_occ.x + face_and_occ.width + 1 - width), \
                    0 if face_and_occ.y + face_and_occ.height < height else int(face_and_occ.y + face_and_occ.height + 1 - height))
        img = functional.pad(img, padding)
        if padding[0] > 0 or padding[1] > 0:
            print('update bbox label.')
            label.face_box.update(padding[0], padding[1])
            label.occ_box.update(padding[0], padding[1])
            face_and_occ.update(padding[0], padding[1])

        img = functional.crop(img, face_and_occ.y, face_and_occ.x, face_and_occ.height, face_and_occ.width)
        if self.test_mode and self.img_ckpt is not None:
            fn = label.filename.rsplit('/', 1)[-1]
            img.save(os.path.join(self.img_ckpt, fn))
            print('save ', fn)
        width, height = img.size

        if label.occ_box.width + label.occ_box.height == 0:
            occ_box = [0]* 4
        else:
            occ_box = [(label.occ_box.x + label.occ_box.width/2 - face_and_occ.x) / width, \
                        (label.occ_box.y + label.occ_box.height/2 - face_and_occ.y) / height, \
                        label.occ_box.width / width, \
                        label.occ_box.height / height]
        cx, cy, w, h = ((label.face_box.x + label.face_box.width/2 - face_and_occ.x) / width, \
                        (label.face_box.y + label.face_box.height/2 - face_and_occ.y) / height, \
                        label.face_box.width / width, label.face_box.height / height
                         )
        '''
        eye_pos = label.eye_pos.tolist() # x_l, y_l, x_r, y_r
        eye_pos[0] = (eye_pos[0] + padding[0]) / width
        eye_pos[1] = (eye_pos[1] + padding[1]) / height
        eye_pos[2] = (eye_pos[2] + padding[0]) / width
        eye_pos[3] = (eye_pos[3] + padding[1]) / height
        '''
        y = np.array([cx, cy, w, h] + occ_box, dtype=np.float32)
        img_tensor = functional.resize(img, (self.in_size, self.in_size))
        img_tensor = self.transforms(img_tensor)
        img_tensor = torch.cat([self.coord_channel, img_tensor], 0)

        return (img_tensor, y) if not self.test_mode else (img_tensor, y, label.filename)


if __name__ == '__main__':
    box = BBox([5, 10, 5, 20])
    w, h = (50, 50)
    print(square(box, w, h))