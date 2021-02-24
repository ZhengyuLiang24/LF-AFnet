from torch.utils.data import DataLoader
import shutil
import importlib
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import argparse
import numpy as np

def main(args):
    def log_string(str):
        if args.local_rank==0:
            logger.info(str)
            print(str)


    ''' Create Dir for Save'''
    experiment_dir, checkpoints_dir, log_dir = create_dir(args)

    ''' Logger '''
    logger = get_logger(log_dir, args)

    ''' Print Parameters '''
    log_string('PARAMETER ...')
    log_string(args)


    ''' Distributed '''
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    try:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:18372',
            world_size=torch.cuda.device_count(),
            rank=args.local_rank,
        )
        log_string('Distributed GPUs Training!')
    except:
        log_string('Single GPU Training!')


    ''' Data Training Loading '''
    log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    log_string("The number of training data is: %d" % len(train_Dataset))
    try:
        train_Sampler = torch.utils.data.distributed.DistributedSampler(train_Dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                                   sampler=train_Sampler, pin_memory=True, drop_last=True)
    except:
        train_Sampler = None
        train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,)


    ''' Data Test Loading '''
    log_string('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    log_string("The number of test data is: %d" % length_of_tests)


    '''Model Loading '''
    log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    shutil.copy('./model/%s.py' % args.model_name, str(experiment_dir))
    net = MODEL.get_model(args)
    net = net.to(device)
    try:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        log_string('Model Distributed DataParallel')
    except:
        log_string('Not Model Distributed DataParallel')
    cudnn.benchmark = True


    ''' Pretrained Model Loading '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        log_string('Do not use pretrain model!')
    else:
        try:
            checkpoint = torch.load(args.path_pre_pth, map_location='cuda:0')
            start_epoch = checkpoint['epoch']
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            log_string('Use pretrain model!')
        except:
            net.apply(MODEL.weights_init)
            start_epoch = 0
            log_string('No existing model!')
            pass
        pass


    '''Loss Loading '''
    criterion = MODEL.get_loss(args).to(device)


    ''' Optimizer Loading'''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)


    ''' TRAINING & VALID '''
    log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        if train_Sampler is not None:
            train_Sampler.set_epoch(idx_epoch)
        log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))


        ''' train '''
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer)
        log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' % (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))


        ''' save model per epoch '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_x%d_epoch_%02d_model.pth' % (
            args.model_name, args.scale_factor, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' valid '''
        if (idx_epoch+1)%1==0:
            ''' test one epoch on every dataset'''
            with torch.no_grad():
                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]
                    psnr_epoch_test, ssim_epoch_test = valid(test_loader, device, net)
                    psnr_testset.append(psnr_epoch_test)
                    ssim_testset.append(ssim_epoch_test)
                    log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (idx_epoch + 1, test_name,
                                                                                psnr_epoch_test, ssim_epoch_test))

        ''' scheduler '''
        scheduler.step()
        pass
    pass



def train(train_loader, device, net, criterion, optimizer):
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        angRes_mixed = select_angRes(args.mixed_ratio)
        patch_size = data.size(2) // args.max_angRes
        data = data[:, :,
               (args.max_angRes - angRes_mixed) // 2 * patch_size:(args.max_angRes + angRes_mixed) // 2 * patch_size,
               (args.max_angRes - angRes_mixed) // 2 * patch_size:(args.max_angRes + angRes_mixed) // 2 * patch_size]
        patch_size = label.size(2) // args.max_angRes
        label = label[:, :,
                (args.max_angRes - angRes_mixed) // 2 * patch_size:(args.max_angRes + angRes_mixed) // 2 * patch_size,
                (args.max_angRes - angRes_mixed) // 2 * patch_size:(args.max_angRes + angRes_mixed) // 2 * patch_size]


        data = data.to(device)
        label = label.to(device)
        angRes_Info = torch.tensor([angRes_mixed]).int()
        out = net(data, angRes_Info)
        loss = criterion(out, label, angRes_Info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(label, out, angRes_Info)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train



def valid(test_loader, device, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)
        Hr_SAI_y = Hr_SAI_y.to(device)
        angRes_test = 5

        patch_h = Lr_SAI_y.size(0) // args.max_angRes
        patch_w = Lr_SAI_y.size(1) // args.max_angRes
        Lr_SAI_y = Lr_SAI_y[
               (args.max_angRes - angRes_test) // 2 * patch_h:(args.max_angRes + angRes_test) // 2 * patch_h,
               (args.max_angRes - angRes_test) // 2 * patch_w:(args.max_angRes + angRes_test) // 2 * patch_w]

        patch_h = Hr_SAI_y.size(2) // args.max_angRes
        patch_w = Hr_SAI_y.size(3) // args.max_angRes
        Hr_SAI_y = Hr_SAI_y[:, :,
                (args.max_angRes - angRes_test) // 2 * patch_h:(args.max_angRes + angRes_test) // 2 * patch_h,
                (args.max_angRes - angRes_test) // 2 * patch_w:(args.max_angRes + angRes_test) // 2 * patch_w]

        angRes_Info = torch.tensor([angRes_test]).int()

        uh, vw = Lr_SAI_y.shape
        h0, w0 = uh // angRes_test, vw // angRes_test
        subLFin = LFdivide(Lr_SAI_y, angRes_test, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFout = torch.zeros(numU, numV, angRes_test * args.patch_size_for_test * args.scale_factor, angRes_test * args.patch_size_for_test * args.scale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(device), angRes_Info)
                    subLFout[u, v, :, :] = out.squeeze()


        Sr_4D_y = LFintegrate(subLFout, angRes_test, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, h0 * args.scale_factor, w0 * args.scale_factor)
        Sr_SAI_y = Sr_4D_y.permute(0, 2, 1, 3).reshape((1, h0 * angRes_test * args.scale_factor,
                                                        w0 * angRes_test * args.scale_factor))
        psnr, ssim = cal_metrics(Hr_SAI_y, Sr_SAI_y, angRes_Info)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    return psnr_epoch_test, ssim_epoch_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
    parser.add_argument("--mixed_ratio", type=str, default='1:1:1:1:1:1:1:1', help="mixed the datasets of angRes")
    parser.add_argument("--max_angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='LF_AFnet', help="")
    parser.add_argument("--channels", type=int, default=64, help="channels")
    parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
    parser.add_argument("--path_pre_pth", type=str, default='./LF_AFnet_x4_epoch_50_model.pth',
                        help="use pre model ckpt")
    parser.add_argument('--path_for_train', type=str, default='./data_for_train/')
    parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
    parser.add_argument('--path_log', type=str, default='./log/')
    parser.add_argument('--data_name', type=str, default='ALL',
                        help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, and ALL of them')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=4, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')
    parser.add_argument('--patch_size_for_test', default=32, type=int, help='patch size')
    parser.add_argument('--stride_for_test', default=16, type=int, help='stride')

    parser.add_argument('--num_workers', type=int, default=4, help='num workers of the Data Loader')
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

    args = parser.parse_args()

    args.mixed_ratio = args.mixed_ratio.split(':')
    assert len(args.mixed_ratio) == 8
    args.mixed_ratio = list(map(float, args.mixed_ratio))
    args.mixed_ratio = np.array(args.mixed_ratio)
    args.mixed_ratio = args.mixed_ratio / np.sum(args.mixed_ratio)

    main(args)
