import argparse
import torch
import ShuffleNetV2
import dataset
import numpy as np
import os
import scipy.io as sio
from PIL import Image
from util import measure, Measure

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint folder')
    # annotation
    parser.add_argument('--anno', type=str, required=True, help='location of annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    parser.add_argument('--save_mat', type=str, default='result_test.mat', help='file to save result')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()

def test(model, data, loss_func, device, img_ckpt=None):
    model.eval()
    sum_loss = 0
    m = Measure()
    pred_rec = {}
    with torch.no_grad():
        for i, batch in enumerate(data):
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            m += measure(pred, y)

            loss = loss_func(y, pred)
            #loss_recon = loss_func[1](x_recon, pred, x, y)
            #pred_rec[fn] = np.hstack([pred.cpu().data.numpy()[0], y.cpu().numpy()[0]])

            sum_loss += loss.item()
            #sum_recon += loss_recon.item()
        f1 = m.f1()
        print('\t\tBatch %d total loss: %.4f\t f1: %.4f'% (i, sum_loss/(1+i), f1))
    
    return (sum_loss)/(i+1), f1

if __name__ == '__main__':
    from train import Criterion
    args = get_args()
    img_ckpt = os.path.join(args.ckpt, 'img_ckpt/')
    if not os.path.exists(img_ckpt):
        os.makedirs(img_ckpt)
    # dataloader
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size, test_mode=True, img_ckpt=img_ckpt),
                                batch_size=args.batch, shuffle=False, num_workers=1, drop_last=args.batch!=1)
    # init model
    model = ShuffleNetV2.ShuffleNetV2(n_class=1, input_size=args.in_size)
    ckpt = torch.load(os.path.join(args.ckpt, 'best_acc.pth'))
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model.to(device)

    loss_func = Criterion(batch_size=args.batch, device=device)

    conf_loss, f1 = test(model, data, loss_func, device, img_ckpt)
    print('loss: %.4f\tf1: %.4f'%(conf_loss, f1))
    #sio.savemat(os.path.join(args.ckpt, args.save_mat), pred_rec)
