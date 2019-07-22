import torch
import torch.nn as nn
import numpy as np
import argparse
import ShuffleNetV2
import dataset
import os
import shutil
import test

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
    parser.add_argument('--anno_test', type=str, required=True, help='location of test data annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()


class Criterion(nn.Module):
    def __init__(self, batch_size=4, device=torch.device('cuda')):
        super(Criterion, self).__init__()
        self.loss_func = nn.SmoothL1Loss(reduction='none')
        self.weight = torch.Tensor(([0.25]*2+[0.75]*2)*2).repeat(batch_size, 1).to(device)
        #self.eye_mask = torch.Tensor([0]*4+[1]*4+[0]*4).repeat(batch_size, 1).to(device)

    def forward(self, gt, pred):
        assert len(gt.shape) == 4
        loss = (self.loss_func(gt, pred) * self.weight).sum() / gt.shape[0]
        return loss


class ContentLoss(nn.Module):
    def __init__(self, side_len=128, device=torch.device('cuda')):
        super(ContentLoss, self).__init__()
        self.side_len = side_len
        self.loss_func = nn.SmoothL1Loss()
        self.device = device

    def forward(self, pred_content, pred_label, gt_content, gt_label):
        loss = 0
        face_gt_label = (gt_label[:, :4].detach().cpu().numpy() * self.side_len).astype(int)
        face_pred_label = (pred_label[:, :4].detach().cpu().numpy() * self.side_len).astype(int)
        for i in range(face_gt_label.shape[0]):
            cx_gt, cy_gt, w_gt, h_gt = face_gt_label[i, :]
            cx_pred, cy_pred, w_pred, h_pred = face_pred_label[i, :]
            w, h = min(int(w_gt/2), int(w_pred/2)), min(int(h_gt/2), int(h_pred/2))
            w = min(min(self.side_len-1-cx_gt, min(w, cx_gt)), min(self.side_len-1-cx_pred, min(w, cx_pred)))
            h = min(min(self.side_len-1-cy_gt, min(h, cy_gt)), min(self.side_len-1-cy_pred, min(h, cy_pred)))
            patch_gt = gt_content[i, 2:, (cy_gt-h): (cy_gt+h), (cx_gt-w): (cx_gt+w)]
            patch_pred = pred_content[i, :, (cy_pred-h): (cy_pred+h), (cx_pred-w): (cx_pred+w)]
            try:
                assert patch_gt.shape == patch_pred.shape
            except AssertionError:
                print('gt: ', cx_gt-w, cx_gt+w, cy_gt-h, cy_gt+h)
                print('pred: ', cx_pred-w, cx_pred+w, cy_pred-h, cy_pred+h)
                print('gt:', patch_gt.shape, '\tpred:', patch_pred.shape)
            loss += self.loss_func(patch_gt, patch_pred)
            #mask[i, :, int(cy-h/2): int(cy+h/2), int(cx-w/2): int(cx+w/2)] = 1
        #mask_count = np.count_nonzero(mask)
        #mask = torch.Tensor(mask).to(self.device)

        return loss / pred_content.shape[0]



if __name__ == '__main__':
    args = get_args()
    best_bb_loss = np.Inf
    epoch_start = args.epoch_start
    # dataloader
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size),
                                batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True) 
    data_test = torch.utils.data.DataLoader(dataset.Faceset(args.anno_test, os.path.join(args.img_folder, 'test'), args.in_size, test_mode=True),
                                batch_size=args.batch, shuffle=False, num_workers=1, drop_last=args.batch!=1)
    # init model
    model = ShuffleNetV2.ShuffleNetV2(n_class=8, input_size=args.in_size)
    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt, 'best_acc.pth'))
        model.load_state_dict(ckpt['state_dict'])
        best_bb_loss = ckpt['occ_loss']
        epoch_start = ckpt['epoch']
        print('Loaded epoch %d'%epoch_start)
        epoch_start += 1
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model.to(device)

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = Criterion(batch_size=args.batch, device=device)
    # criterion_content = ContentLoss(side_len=args.in_size, device=device)
    # reduce the loss of face bounding box and eye position after half the training procedure
    #if epoch_start > args.epochs/2:
    #    criterion.update_weights(update_weight)

    print('Start training from epoch %d'%epoch_start)
    for epoch in range(epoch_start, args.epochs):
        model.train()
        '''
        if epoch == args.epochs//2 and epoch_start < args.epochs/2:
            criterion.update_weights(update_weight)
        '''
        sum_loss = 0
        for i, batch in enumerate(data):
            x, y = batch
            x, y = x.to(device), y.to(device)
            #print('x:', x.dtype, '\ty:', y.dtype)
            #print('x shape:', x.shape, 'y shape:', y.shape)
            pred = model(x)
            #print('pred shape:', pred.shape, pred.dtype)

            optimizer.zero_grad()
            loss = criterion(y, pred)
            # loss_recon = criterion_content(x_recon, pred, x, y)
            # loss += 0.5*loss_recon
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
        print('\tEpoch %d total loss: %.4f'%(epoch, sum_loss/(1+i)))

        test_loss, _ = test.test(model, data_test, criterion, device)
        if best_bb_loss > test_loss:
            best_bb_loss = test_loss
            torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch, 'loss': sum_loss}, \
                        os.path.join(args.ckpt, '%d_epoch_ckpt.pth'%epoch))
            shutil.copy(os.path.join(args.ckpt, '%d_epoch_ckpt.pth'%epoch), os.path.join(args.ckpt, 'best_acc.pth'))
            print('Saved model.')
            model.to(device)
