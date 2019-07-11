import torch
import torch.nn as nn
import argparse
import ShuffleNetV2
import dataset

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch to count from')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    # annotation
    parser.add_argument('--anno', type=str, required=True, help='location of annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()


class Criterion(nn.Module):
    def __init__(self, weights=[0.4, 1, 0.5], batch_size=4):
        super(Criterion, self).__init__()
        self.loss_func = nn.SmoothL1Loss(reduction='none')
        self.face_mask = torch.Tensor([weights[0]]*4 + [weights[1]*4] + [weights[2]]).repeat(batch_size, 1)

    def __call__(self, gt, pred):
        loss = (self.loss_func(gt, pred) * self.face_mask).mean
        return loss


if __name__ == '__main__':
    args = get_args()
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size),
                                batch_size=args.batch, shuffle=True, num_workers=4) 
    model = ShuffleNetV2.ShuffleNetV2(n_class=9, input_size=args.in_size)

    for epoch in range(args.epoch_start, args.epochs):
        
        for i, batch in enumerate(data):
            x, y = batch
            print('x:', type(x), '\ty:', type(y))
            print('x shape:', x.shape, 'y shape:', y.shape)
            pred = model(x)
            print('pred shape:', pred.shape)