import torch
import torch.nn as nn
import numpy as np
import argparse
import ShuffleNetV2
import dataset
import os
import shutil

init_weight = (0.5, 0.1, 1)
update_weight = (0.25, 0.05, 2.2)

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch to count from')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint folder')
    parser.add_argument('--resume', action='store_true', help='load previous best model and resume training')
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

    def update_weights(self, weights):
        assert len(weights) == 3
        self.w_face_bb, self.w_eye, self.w_conf = weights

    def __call__(self, gt, pred):
        loss = self.loss_func(gt, pred)
        loss_face = (loss * self.face_mask).sum()
        loss_eye = (loss * self.eye_mask).sum()
        loss_conf = (loss * self.conf_mask).sum()
        loss_optim = (loss * (self.face_mask * self.w_face_bb + self.eye_mask * self.w_eye + self.conf_mask * self.w_conf)).sum()
        return loss_optim, loss_face, loss_eye, loss_conf


if __name__ == '__main__':
    args = get_args()
    best_conf_loss = np.Inf
    epoch_start = args.epoch_start
    # dataloader
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size),
                                batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True) 
    # init model
    model = ShuffleNetV2.ShuffleNetV2(n_class=9, input_size=args.in_size)
    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt, 'best_acc.pth'))
        model.load_state_dict(ckpt['state_dict'])
        best_conf_loss = ckpt['conf_loss']
        epoch_start = ckpt['epoch']
        print('Loaded epoch %d'%epoch_start)
        epoch_start += 1
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model.to(device)

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = Criterion(weights=init_weight, batch_size=args.batch, device=device)
    # reduce the loss of face bounding box and eye position after half the training procedure
    if epoch_start > args.epochs/2:
        criterion.update_weights(update_weight)

    print('Start training from epoch %d'%epoch_start)
    for epoch in range(epoch_start, args.epochs):
        if epoch == args.epochs//2 and epoch_start < args.epochs/2:
            criterion.update_weights(update_weight)
        
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
            print('\tBatch %d total loss: %.4f\tface:%.4f\teye:%.4f\tconf:%.4f'% \
                (i, sum_loss/(1+i), sum_face/(1+i), sum_eye/(1+i), sum_conf/(1+i)))
        
        print('End of Epoch %d'%epoch)
        sum_conf /= i
        if best_conf_loss > sum_conf:
            best_conf_loss = sum_conf
            torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch, 'conf_loss': sum_conf}, \
                        os.path.join(args.ckpt, '%d_epoch_ckpt.pth'%epoch))
            shutil.copy(os.path.join(args.ckpt, '%d_epoch_ckpt.pth'%epoch), os.path.join(args.ckpt, 'best_acc.pth'))
            print('Saved model.')
            model.to(device)