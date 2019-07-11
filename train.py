import torch
import torch.nn as nn
import numpy as np
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
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # annotation
    parser.add_argument('--anno', type=str, required=True, help='location of annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()


class Criterion(nn.Module):
    def __init__(self, weights=[0.5, 0.1, 1], batch_size=4, device=torch.device('cuda')):
        super(Criterion, self).__init__()
        self.loss_func = nn.SmoothL1Loss(reduction='none')
        assert len(weights) == 3
        self.w_face_bb, self.w_eye, self.w_conf = weights
        self.face_mask = torch.Tensor([1]*4 + [0]*5).repeat(batch_size, 1).to(device)
        self.eye_mask = torch.Tensor([0]*4+[1]*4+[0]).repeat(batch_size, 1).to(device)
        self.conf_mask = torch.Tensor([0]*8+[1]).repeat(batch_size, 1).to(device)

    def __call__(self, gt, pred):
        loss = self.loss_func(gt, pred)
        loss_face = (loss * self.face_mask).sum()
        loss_eye = (loss * self.eye_mask).sum()
        loss_conf = (loss * self.conf_mask).sum()
        loss_optim = (loss * (self.face_mask * self.w_face_bb + self.eye_mask * self.w_eye + self.conf_mask * self.w_conf)).mean()
        return loss_optim, loss_face, loss_eye, loss_conf


if __name__ == '__main__':
    args = get_args()
    # dataloader
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size),
                                batch_size=args.batch, shuffle=True, num_workers=4) 
    # init model
    model = ShuffleNetV2.ShuffleNetV2(n_class=9, input_size=args.in_size)
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model.to(device)
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = Criterion(weights=(0.5, 0.1, 1), batch_size=args.batch, device=device)

    best_conf_loss = np.Inf
    print('Start training from epoch %d'%args.epoch_start)
    for epoch in range(args.epoch_start, args.epochs):
        
        sum_loss, sum_face, sum_eye, sum_conf = 0, 0, 0, 0
        for i, batch in enumerate(data):
            x, y = batch
            x, y = x.to(device), y.to(device)
            #print('x:', x.dtype, '\ty:', y.dtype)
            #print('x shape:', x.shape, 'y shape:', y.shape)
            pred = model(x)
            #print('pred shape:', pred.shape, pred.dtype)

            optimizer.zero_grad()
            loss, loss_face, loss_eye, loss_conf = criterion(y, pred)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_face += loss_face.item()
            sum_eye += loss_eye.item()
            sum_conf += loss_conf.item()
            print('\tBatch %d total loss: %.2f\tface:%.2f\teye:%.2f\tconf:%.2f'% \
                (i, sum_loss/(1+i), sum_face/(1+i), sum_eye/(1+i), sum_conf/(1+i)))
        
        print('End of Epoch %d'%epoch)
        if best_conf_loss > sum_conf:
            best_conf_loss = sum_conf
