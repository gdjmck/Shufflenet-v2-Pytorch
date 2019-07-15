import argparse
import torch
import ShuffleNetV2
import dataset
import os

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
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # dataloader
    data = torch.utils.data.DataLoader(dataset.Faceset(args.anno, args.img_folder, args.in_size),
                                batch_size=1, shuffle=True, num_workers=4, drop_last=False)
    # init model
    model = ShuffleNetV2.ShuffleNetV2(n_class=9, input_size=args.in_size)
    ckpt = torch.load(os.path.join(args.ckpt, 'best_acc.pth'))
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model.to(device)

    loss_func = Criterion(weights=weight, batch_size=1, device=device)
    sum_loss, sum_face, sum_eye, sum_conf = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(data):
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss, loss_face, loss_eye, loss_conf = criterion(y, pred)

            sum_loss += loss.item()
            sum_face += loss_face.item()
            sum_eye += loss_eye.item()
            sum_conf += loss_conf.item()
            print('\tBatch %d total loss: %.4f\tface:%.4f\teye:%.4f\tconf:%.4f'% \
                (i, sum_loss/(1+i), sum_face/(1+i), sum_eye/(1+i), sum_conf/(1+i)))