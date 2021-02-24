import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
parser.add_argument("--mixed_ratio", type=str, default='1:1:1:1:1:1:1:1', help="mixed the datasets of angRes")
parser.add_argument("--max_angRes", type=int, default=9, help="angular resolution")
parser.add_argument('--model_name', type=str, default='LF_AFnet', help="")
parser.add_argument("--channels", type=int, default=64, help="channels")
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument('--path_demo', type=str, default='./demo/')
parser.add_argument('--angRes_demo', type=int, default='5')
parser.add_argument('--path_demo_result', type=str, default='./demo_result/')
parser.add_argument("--path_pre_pth", type=str, default='./LF_AFnet_x4_epoch_50_model.pth', help="use pre model ckpt")
parser.add_argument('--path_for_train', type=str, default='./data_for_train/')
parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--data_name', type=str, default='ALL', help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, and ALL of them')
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


