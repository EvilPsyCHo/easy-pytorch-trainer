import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class AverageMeter(object):
    val = 0
    avg = 0
    sum = 0
    count = 0
    best = None

    """Computes and stores the average and current value"""
    def __init__(self, larger_is_better=False):
        self.larger_is_better = larger_is_better
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.larger_is_better:
            self.best = -np.inf
        else:
            self.best = np.inf

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.larger_is_better and (val > self.best):
            self.best = val
            return True
        elif (not self.larger_is_better) and (val < self.best):
            self.best = val
            return True
        else:
            return False


def to_device(d, device):
    if isinstance(d, dict):
        return {k: to_device(v, device) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    elif isinstance(d, list):
        return [to_device(i, device) for i in d]
    else:
        raise ValueError


class TrainerConfig:
    max_epochs = 8
    batch_size = 32
    grad_norm_clip = None
    # Data Loader
    num_workers = 8
    collate_fn = None
    # Device
    device = "cuda:0"
    gpu_ids = None

    def __init__(self, save_path, **kwargs):
        self.save_path = save_path
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, train_config, model, optimizer, train_dataset,
                 test_dataset=None, metric_fn=None, metric_larger_better=True, lr_scheduler=None, collate_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = train_config
        self.metric_fn = metric_fn
        self.metric_larger_better = metric_larger_better
        self.collate_fn = collate_fn
        # take over whatever gpus are on the system
        self.device = self.config.device
        if self.config.gpu_ids is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_ids).to(self.device)
        else:
            self.model.to(self.device)

    def save_checkpoint(self, oof=None):
        # DataParallel wrappers keep raw model object in .module attribute
        Path(self.config.save_path).parent.mkdir(parents=True, exist_ok=True)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        save_path = self.config.save_path
        print(f"saved in {save_path}")
        torch.save({'checkpoint': raw_model.state_dict(), 'oof': oof}, save_path)

    @torch.no_grad()
    def valid_epoch(self, loader):
        model, config = self.model, self.config
        model.eval()
        losses = AverageMeter()
        pbar = enumerate(loader)
        score = None
        preds, labels = [], []
        for it, (x, y) in pbar:
            x = to_device(x, self.device)
            y = to_device(y, self.device)
            # forward the model
            logit, loss = model(**x, label=y)
            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            losses.update(loss.item())
            preds.append(logit.cpu().numpy())
            labels.append(y.cpu().numpy())
            del x; del y; gc.collect()
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        if self.metric_fn is not None:
            score = self.metric_fn(labels, preds)
        return losses.avg, preds, score

    def train_epoch(self, loader, num_epoch):
        model, optimizer, lr_scheduler, config = self.model, self.optimizer, self.lr_scheduler, self.config
        model.train()
        score = None
        losses = AverageMeter()
        pbar = tqdm(enumerate(loader), total=len(loader))
        preds = []
        labels = []

        for it, (x, y) in pbar:
            # place data on the correct device
            x = to_device(x, self.device)
            y = to_device(y, self.device)

            # forward the model
            logit, loss = model(**x, label=y)
            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

            # backprop and update the parameters
            model.zero_grad()
            loss.backward()
            if self.config.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()

            # record
            loss = loss.item()
            losses.update(loss)

            preds.append(logit.detach().cpu().numpy())
            labels.append(y.cpu().numpy())

            # decay the learning rate based on our progress
            if lr_scheduler is not None:
                lr_scheduler.step(num_epoch+it/len(loader))
            lr = []
            for param_group in optimizer.param_groups:
                lr.append(param_group['lr'])
            lr = np.mean(lr)

            # report progress
            pbar.set_description(f"train loss {loss:.4f} lr {lr:.6f}")

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        if self.metric_fn is not None:
            score = self.metric_fn(labels, preds)
        return losses.avg, score

    def fit(self):
        config = self.config
        scores = AverageMeter(self.metric_larger_better)

        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=self.collate_fn
        )

        if self.test_dataset is not None:
            test_loader = DataLoader(
                self.test_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                collate_fn=self.collate_fn
            )

        for epoch in range(config.max_epochs):
            trn_loss, trn_score = self.train_epoch(train_loader, epoch)
            info = f"Epoch {epoch + 1} train loss {trn_loss:.4f}"
            if self.metric_fn is not None:
                info += f" score {trn_score:.4f}"
            if self.test_dataset is not None:
                val_loss, val_preds, val_score = self.valid_epoch(test_loader)
                info += f" val loss {val_loss:.4f}"
                if self.metric_fn is not None:
                    info += f" score {val_score: .4f}"
                good_model = scores.update(val_score)
                if good_model:
                    self.save_checkpoint(val_preds)
            else:
                self.save_checkpoint()
            print(info)
            # supports early stopping based on the test loss, or just save always if no test set is provided

    @torch.no_grad()
    def predict(self, dataset):
        self.model.eval()
        loader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )
        preds = []
        for x in loader:
            to_device(x, self.device)
            logit = self.model(**x)
            preds.append(logit.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return preds
