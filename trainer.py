from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader


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
            self.best = -np.float('inf')
        else:
            self.best = np.float('inf')

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


class TrainerConfig:
    max_epochs = 8
    batch_size = 32
    grad_norm_clip = 1.0
    ckpt_path = "model.bin"
    num_workers = 8
    collate_fn=None
    device="cuda:0"
    gpu_ids=None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, train_config, model, optimizer, lr_scheduler, train_dataset,
                 test_dataset, metric_fn, metric_larger_better=True, collate_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = train_config
        self.metric_fn = metric_fn
        self.metric_name = metric_fn.__name__
        self.metric_larger_better = metric_larger_better
        self.collate_fn = collate_fn
        # take over whatever gpus are on the system
        self.device = self.config.device
        if self.config.gpu_ids is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_ids).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        Path(self.config.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    @torch.no_grad()
    def valid_epoch(self, loader):
        model, config = self.model, self.config
        model.eval()
        scores = AverageMeter(self.metric_larger_better)
        losses = AverageMeter()
        pbar = enumerate(loader)

        for it, (x, y) in pbar:
            for k in x.keys():
                x[k] = x[k].to(self.device)

            # forward the model
            logit, loss = model(x, label=y.to(self.device))
            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            losses.update(loss.item())
            scores.update(self.metric_fn(y, logit.cpu().numpy()))
        return losses.avg, scores.avg

    def train_epoch(self, loader, num_epoch):
        model, optimizer, lr_scheduler, config = self.model, self.optimizer, self.lr_scheduler, self.config
        model.train()
        scores = AverageMeter(self.metric_larger_better)
        losses = AverageMeter()
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            # place data on the correct device
            for k in x.keys():
                x[k] = x[k].to(self.device)

            # forward the model
            logit, loss = model(x, label=y.to(self.device))
            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

            # backprop and update the parameters
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()

            # record
            loss = loss.item()
            losses.update(loss)
            score = self.metric_fn(y, logit.detach().cpu().numpy())
            scores.update(score)

            # decay the learning rate based on our progress
            lr_scheduler.step(num_epoch+it/len(loader))
            lr = []
            for param_group in optimizer.param_groups:
                lr.append(param_group['lr'])
            lr = np.mean(lr)

            # report progress
            pbar.set_description(f"train loss {loss:.4f}, {self.metric_name} {score:.4f}, lr {lr:.6f}")
        return losses.avg, scores.avg

    def train(self):
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
            val_loss, val_score = self.valid_epoch(test_loader)
            print(f"Epoch {epoch + 1} "
                  f"train loss {trn_loss:.4f} {self.metric_name} {trn_score:.4f}, "
                  f"val loss {val_loss:.4f} {self.metric_name} {val_score: .4f}")
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = scores.update(val_score)
            if self.config.ckpt_path is not None and good_model:
                self.save_checkpoint()
        return scores.best

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
            for k in x.keys():
                x[k] = x[k].to(self.device)
            logit = self.model(x)
            preds.append(logit.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return preds
