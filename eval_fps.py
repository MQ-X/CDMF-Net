import time
import torch
import torch.backends.cudnn as cudnn

from argparse import ArgumentParser
#from builders.model_builder import build_model
# from networks.Fast_scnn_change import FastSCNN_NEW3
from networks.fast_scnn import FastSCNN
from networks.Fast_scnn_change import *

def compute_speed(model, input_size, device, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)
    print(args.model+' is training')
    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time , fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--size", type=str, default="800,800", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=19)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--model', type=str, default='CFPNet')
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    parser.add_argument('--aux', type=str, default=False, help='folder id contain images ROIs to train or validation')
    args = parser.parse_args()

    h, w = map(int, args.size.split(','))
    #model = build_model(args.model, num_classes=args.classes)
    model = FastSCNN_NEW3(num_classes=2, aux=args.aux)
    #print('111')
    #print(model)
    compute_speed(model, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)
