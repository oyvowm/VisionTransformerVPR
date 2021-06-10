"""
Different classes and function being used in other scripts
"""

import torch
import numpy as np
# calculate accuracy

def accuracy(output, target, top_num):
    with torch.no_grad():
        _, prediction = torch.max(output, 1)
        acc = (prediction == target).sum().item()
        return acc

    
# from https://github.com/pytorch/examples/blob/507493d7b5fab51d55af88c5df9eadceb144fb67/imagenet/main.py#L363   
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


def to_tuple(a):
    a = [a]
    return(tuple(np.repeat(a,2)))

    
def save_model(epoch, model, optimizer, train_epoch_loss, train_loss, learning_rate_plot, scheduler, training_time, save_path, val_loss):
    print(f'Saving model at {save_path}', flush=True)
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_epoch_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate_plot': learning_rate_plot,
        'scheduler_state_dict': scheduler.state_dict(),
        'training_time': training_time
        }, save_path)
