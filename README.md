# easy-pytorch-trainer

Easy for use and modified,  pure pytorch.  My deeplearning default trainer.

## features

- Multiple GPUs
- Grad Norm

## Examples

```python
train_config = TrainerConfig(
    save_path="./model.bin"
    gpu_ids=[0, 1],
    device=0,
    batch_size=128,
    max_epochs=2,
)

trainer = Trainer(train_config, model, optimizer, lr_sheduler, trn_ds, val_ds, metric_function)
trainer.fit()
```

## Next

- fp16 support
- detail example notebook
