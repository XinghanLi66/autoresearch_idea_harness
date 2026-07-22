#!/bin/bash
cd mnist
python main_custom.py --seed ${SEED:-42} --epochs 14 --save-model
