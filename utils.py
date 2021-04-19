import torch
import torchvision.transforms as transforms


class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target):
  batch_size = target.size(0)
  _, pred = output.topk(1, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  res = []
  correct_k = correct[:1].view(-1).float().sum(0)
  res.append(correct_k.mul_(100.0/batch_size))
  return res

def _data_transforms_us(args):
    # RBG
    US_MEAN = [0.243756, 0.256587, 0.282307]
    US_STD = [0.185635, 0.199266, 0.209196]
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(US_MEAN, US_STD),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(US_MEAN, US_STD),
    ])
    return train_transform, valid_transform

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))
  print('load model finish')
