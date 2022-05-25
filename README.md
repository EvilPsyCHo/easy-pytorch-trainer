# easy-pytorch-trainer

Easy for use and modified,  pure pytorch.  My deeplearning default trainer.

## features

- Multiple GPUs
- Log loss/metric/learning_rate when training
- Auto save when valid score improved

## Examples

```python
train_config = TrainerConfig(
    gpu_ids=[0, 1],
    device=0,
    batch_size=128,
    max_epochs=2,
)

trainer = Trainer(train_config, model, optimizer, lr_sheduler, trn_ds, val_ds, metric_function)
trainer.train()
```

## Next

- fp16 support
- detail example notebook
