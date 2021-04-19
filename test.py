import sys
import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkUS as Network

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
us_25 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

parser = argparse.ArgumentParser("US")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='./best_acc_weights.pt',
                    help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='us_25', help='which architecture to use')
args = parser.parse_args()

US_CLASSES = 11

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, US_CLASSES, args.layers, args.auxiliary, genotype, args.drop_path_prob)
    model = model.cuda()
    utils.load(model, args.model_path)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_us(args)
    test_data = datasets.ImageFolder(root=args.data, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_acc = infer(test_queue, model, criterion)
    print('test_acc %f', test_acc)


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(non_blocking=True)
            logits, out = model(input)
            loss = criterion(logits, target)
            prec1 = utils.accuracy(logits, target)
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1[0], n)
            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
    return top1.avg

if __name__ == '__main__':
    main()
