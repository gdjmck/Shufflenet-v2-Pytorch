import torch.utils.data.Dataset as Dataset

class BBox():
    def __init__(self, data):
        self.x = int(data[0])
        self.y = int(data[1])
        self.width = int(data[2])
        self.height = int(data[3])

class LabelData():
    def __init__(self, label, root='MAFA/images/'):
        self.root = root
        self.filename = label[0]
        self.img = cv2.imread(root+self.filename, -1)
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


class Faceset(Dataset):
    def __init__(self, anno_file, image_folder):
        super(self, Dataset).__init__()
        self.anno = np.loads(anno_file)
        self.image_folder = image_folder
        
    def __len__(self):
        return self.anno.shape[0]
    
    def __getitem__(self, idx):
        label = LabelData(self.anno[idx, :], self.image_folder)
        img = label
        height, width = img.shape[:2]
        pad = abs(height-width) // 2
        confidence = label.occ_box.width * label.occ_box.height / (label.face_box.width* label.face_box.height)
        cx, cy, w, h = (label.face_box.x + label.face_box.width/2) / width, 
                        (label.face_box.y + label.face_box.height/2) / height,
                        label.face_box.width / width,
                        label.face_box.height / height