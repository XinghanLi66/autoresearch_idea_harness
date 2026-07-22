# MNIST Classification: Learning Rate Scheduler Optimization

## Objective
Improve MNIST classification accuracy by designing a better learning rate scheduler. You can modify the scheduler import (line 7), custom scheduler definition (lines 9-11), and scheduler creation (line 131) in `main_custom.py`.

## Background
The optimizer is fixed as `Adadelta(lr=args.lr)`. The scheduler's `.step()` is called once per epoch. Training runs for 14 epochs. Default StepLR (gamma=0.7) achieves ~99.2% accuracy.
