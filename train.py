import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from pathlib import Path

import models
import dataloader


def train_one_epoch(tr_loader, model, loss_fn, optimizer, writer, epoch, logger,
                    device='cuda', report_interval=100):
    """train one epoch
    
    Args:
        tr_loader (torch.utils.data.DataLoader): training data loader
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        writer (torch.utils.tensorboard.SummaryWriter): tensorboard writer
        epoch (int): epoch index
        device (str): device
        report_interval (int): report interval
        
    Returns:
        float: last reported loss

    """
    running_loss = 0.

    for i, data in enumerate(tr_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % report_interval == report_interval - 1 or i == len(tr_loader) - 1:
            last_loss = running_loss / report_interval
            logger.info('  batch %d loss: %.4f' % (i + 1, last_loss))
            tb_x = epoch * len(tr_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(model, loss_fn, optimizer, scheduler, n_epochs,
          tr_loader, vl_loader, device='cuda', work_dir='runs/model/'):
    """train model
    
    Args:
        model (torch.nn.Module): model
        loss_fn (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler
        n_epochs (int): number of epochs
        tr_loader (torch.utils.data.DataLoader): training data loader
        vl_loader (torch.utils.data.DataLoader): validation data loader
        device (str): device
        save_dir (str): directory to save model

    Returns:
        None

    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path('%s/%s' % (work_dir, timestamp)).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename='%s/%s/log' % (work_dir, timestamp), filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('train')
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    writer = SummaryWriter('%s/%s/tb/' % (work_dir, timestamp))

    best_vloss = 1_000_000.

    model.to(device)

    for epoch in range(n_epochs):
        logger.info('EPOCH %d:' % (epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch(
            tr_loader, model, loss_fn, optimizer, writer, epoch, logger,
            device, report_interval=500
        )
        scheduler.step()

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(vl_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        logger.info('LOSS train %.4f valid %.4f' % (avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '%s/%s/cp_%d.pth' % (work_dir, timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    writer.close()
    model.to('cpu')

    return


if __name__ == '__main__':

    model_name = 'convnext_tiny'
    n_classes = 3

    tr_dir = '/mnt/d/MyFiles/Research/ImageClassification/data/formatted/busi/tr/'
    vl_dir = '/mnt/d/MyFiles/Research/ImageClassification/data/formatted/busi/vl/'
    batch_size = 32

    tr_loader = DataLoader(
        dataloader.BUSI_Dataset(tr_dir, transform=True, mask=False),
        batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn= lambda x : (torch.stack([i[0] for i in x]),
                                torch.stack([torch.tensor(i[1]['label']) for i in x]))
    )
    vl_loader = DataLoader(
        dataloader.BUSI_Dataset(vl_dir, transform=False, mask=False),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn= lambda x : (torch.stack([i[0] for i in x]),
                                torch.stack([torch.tensor(i[1]['label']) for i in x]))
    )

    model = models.get_convnext(model_name, n_classes, new_classifier=False)
    models.freeze_model(model, pattern='^(?!classifier).*$')

    loss_fn = torch.nn.CrossEntropyLoss()

    lr = 1e-3
    optimizer = torch.optim.AdamW(
        filter(lambda p : p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-9, last_epoch=-1)

    n_epochs = 100
    device = 'cuda'

    work_dir = '/mnt/d/MyFiles/Research/ImageClassification/runs/convnext_tiny_busi/'

    train(
        model, loss_fn,
        optimizer, scheduler, n_epochs,
        tr_loader, vl_loader,
        device, work_dir=work_dir
    )

    print('Done!')
