import os

import torch
from torch import nn
import torch.nn.functional as F

from utils.util import get_lr
from tqdm import tqdm

def fit_one_epoch(model, loss_fn, gen, gen_val, device, optimizer, Batch_size, epoch, total_epoch, epoch_step, epoch_step_val, save_dir, save_period):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0

    print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3)

    model.train()
    for iteration, batch in enumerate(gen):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs1, outputs2 = model(images, "train")
        _triplet_loss = loss_fn(outputs1, Batch_size)
        _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
        _loss = _triplet_loss + _CE_loss
        _loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                            'total_CE_loss': total_CE_loss / (iteration + 1),
                            'accuracy': total_accuracy / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)


    print('Finish Train')
    print('Start Validation')

    model.eval()
    for iteration, batch in enumerate(gen_val):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs1, outputs2 = model(images, "train")

        _triplet_loss = loss_fn(outputs1, Batch_size)
        _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
        _loss = _triplet_loss + _CE_loss

        accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        val_total_triple_loss += _triplet_loss.item()
        val_total_CE_loss += _CE_loss.item()
        val_total_accuracy += accuracy.item()

        pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                            'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                            'val_accuracy': val_total_accuracy / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    print('Finish Validation')

    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))
    if (epoch + 1) % save_period == 0 or epoch + 1 == total_CE_loss:
        torch.save(model.state_dict(),
                   os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1),
                              (total_triple_loss + total_CE_loss) / epoch_step, (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)))
