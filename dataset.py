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

class Faceset(data.Dataset):
    def __init__(self, anno_file, image_folder, in_size=128):
        assert anno_file.endswith('.txt')
        with open(anno_file, 'r') as f:
            self.anno = parse_anno(f)
        self.image_folder = image_folder
        self.in_size = in_size
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
        
    def __len__(self):
        return self.anno.shape[0]
    
    def __getitem__(self, idx):
        label = LabelData(self.anno[idx, :], self.image_folder)
        img = Image.open(label.filename).convert('RGB')
        width, height = img.size
        pad = abs(height-width)
        padding = (pad//2, 0, pad-pad//2, 0) if height > width else (0, pad//2, 0, pad-pad//2)
        img = functional.pad(img, padding)
        #confidence = label.occ_box.width * label.occ_box.height / (label.face_box.width* label.face_box.height)
        if (np.array(label.occ_box.to_array())==-1).all():
            confidence = 1
        else:
            occ_box = label.occ_box.to_array()
            occ_box[0] += label.face_box.x
            occ_box[1] += label.face_box.y
            confidence = util.iou_gt(occ_box, label.face_box.to_array())
            try:
                assert confidence >= 0 and confidence <= 1
            except AssertionError:
                print(confidence, 'rect1:', occ_box, 'rect2:', label.face_box.to_array())
        width, height = img.size
        cx, cy, w, h = (padding[0] + label.face_box.x + label.face_box.width/2) / width, \
                        (padding[1] + label.face_box.y + label.face_box.height/2) / height, \
                        label.face_box.width / width, \
                        label.face_box.height / height
        
        eye_pos = label.eye_pos.tolist()
        eye_pos[0] = (eye_pos[0] + padding[0]) / width
        eye_pos[1] = (eye_pos[1] + padding[1]) / height
        eye_pos[2] = (eye_pos[2] + padding[0]) / width
        eye_pos[3] = (eye_pos[3] + padding[1]) / height

        y = np.array([cx, cy, w, h] + eye_pos + [confidence], dtype=np.float32)
        img = functional.resize(img, (self.in_size, self.in_size))
        img = self.transforms(img)

        return img, y