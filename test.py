import argparse
import torch
import ShuffleNetV2
import dataset
import numpy as np
import os
import scipy.io as sio
from train import Criterion

weight = (0.25, 0.05, 2.2)

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

def test(model, data, loss_func, device):
    model.eval()
    sum_loss, sum_face, sum_eye, sum_conf, sum_recon = 0, 0, 0, 0, 0
    pred_rec = {}
    with torch.no_grad():
        for i, batch in enumerate(data):
            x, y, fn = batch
            x, y = x.to(device), y.to(device)
            pred, x_recon = model(x)

            loss, loss_face, loss_eye, loss_conf = loss_func[0](y, pred)
            loss_recon = loss_func[1](x_recon, x, y)
            pred_rec[fn] = np.hstack([pred.cpu().data.numpy()[0], y.cpu().numpy()[0]])

            sum_loss += loss.item()
            sum_face += loss_face.item()
            sum_eye += loss_eye.item()
            sum_conf += loss_conf.item()
            sum_recon += loss_recon.item()
            print('\t\tBatch %d total loss: %.4f\trecon:%.4f\tface:%.4f\teye:%.4f\tconf:%.4f'% \
                (i, sum_loss/(1+i), sum_recon/(i+1), sum_face/(1+i), sum_eye/(1+i), sum_conf/(1+i)))
    
    return sum_conf/(i+1), pred_rec

if __name__ == '__main__':
    args = get_args()
    # dataloader
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size, test_mode=True),
                                batch_size=args.batch, shuffle=False, num_workers=1, drop_last=args.batch!=1)
    # init model
    model = ShuffleNetV2.ShuffleNetV2(n_class=9, input_size=args.in_size)
    ckpt = torch.load(os.path.join(args.ckpt, 'best_acc.pth'))
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model.to(device)

    loss_func = (Criterion(weights=weight, batch_size=args.batch, device=device),
                    ContentLoss(side_len=args.in_size, device=device))

    conf_loss, pred_rec = test(model, data, loss_func)
    sio.savemat(os.path.join(args.ckpt, args.save_mat), pred_rec)
