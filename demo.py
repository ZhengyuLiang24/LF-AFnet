import importlib
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import matplotlib.image as mp
from utils.imresize import *
from utils.utils import *
import imageio
import os
import argparse


def main(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)


    '''Model Loading '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)
    net = net.to(device)
    cudnn.benchmark = True


    ''' Pretrained Model Loading '''
    try:
        checkpoint = torch.load(args.path_pre_pth, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
    except:
        net.apply(MODEL.weights_init)
        print('No existing model!')
        exit()
        pass
    pass


    print('\nStart ...')
    img_list = os.listdir(args.path_demo)
    angRes = args.angRes
    for index, _ in enumerate(img_list):
        img_dir = args.path_demo + img_list[index]
        img = mp.imread(img_dir)

        Lr_SAI_ycbcr = torch.from_numpy(rgb2ycbcr(img))
        Lr_SAI_y = Lr_SAI_ycbcr[:,:,0].to(device)
        Sr_SAI_ycbcr = imresize(Lr_SAI_ycbcr, scalar_scale=args.scale_factor)

        uh, vw = Lr_SAI_y.shape
        h0, w0 = uh // angRes, vw // angRes
        subLFin = LFdivide(Lr_SAI_y, angRes, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFout = torch.zeros(numU, numV, angRes * args.patch_size_for_test * args.scale_factor,
                               angRes * args.patch_size_for_test * args.scale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(device), torch.tensor([angRes]).int())
                    subLFout[u, v, :, :] = out.squeeze()

        Sr_4D_y = LFintegrate(subLFout, angRes, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, h0 * args.scale_factor, w0 * args.scale_factor)

        Sr_SAI_y = Sr_4D_y.permute(0, 2, 1, 3).reshape((h0 * angRes * args.scale_factor, w0 * angRes * args.scale_factor, 1))
        Sr_SAI_y = Sr_SAI_y.numpy()
        Sr_SAI_ycbcr[:,:,0:1] = Sr_SAI_y
        Sr_SAI_rgb = ycbcr2rgb(Sr_SAI_ycbcr).clip(0,255)
        Sr_center_rgb = get_LFcenter(Sr_SAI_rgb, angRes)

        imageio.imwrite(args.path_demo_result + 'full_x' + str(args.scale_factor) + '_'+img_list[index], Sr_SAI_rgb)
        imageio.imwrite(args.path_demo_result + 'center_x' + str(args.scale_factor) + '_' + img_list[index], Sr_center_rgb)
        print('%s has been super-resolved. ' % (img_list[index]))
        pass


def get_LFcenter(LF_full, angRes):
    # numpy
    [H, W, C] = LF_full.shape
    LF_center = LF_full.reshape(angRes, H // angRes, angRes, W // angRes, C)
    LF_center = LF_center[angRes // 2, :, angRes // 2, :, :]

    return LF_center


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
    parser.add_argument('--model_name', type=str, default='LF_AFnet', help="")
    parser.add_argument("--channels", type=int, default=64, help="channels")
    parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
    parser.add_argument('--path_demo', type=str, default='./demo/')
    parser.add_argument('--angRes', type=int, default='5')
    parser.add_argument('--path_demo_result', type=str, default='./demo_result/')
    parser.add_argument("--path_pre_pth", type=str, default='./LF_AFnet_x4_epoch_50_model.pth',
                        help="use pre model ckpt")


    parser.add_argument('--patch_size_for_test', default=32, type=int, help='patch size')
    parser.add_argument('--stride_for_test', default=16, type=int, help='stride')

    parser.add_argument('--num_workers', type=int, default=4, help='num workers of the Data Loader')
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

    args = parser.parse_args()

    main(args)
