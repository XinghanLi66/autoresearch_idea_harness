#!/bin/bash
cd mnist
python main.py --seed ${SEED:-42} --epochs 14 --save-model
