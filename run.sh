#!/bin/bash
echo "Starting processing data by running process.py"
python process.py
echo "Done"
sh ./scripts/different_quantity.sh
sh ./scripts/different_model.sh
sh ./scripts/classification.sh
sh ./scripts/ablation.sh
sh ./scripts/robustness.sh
sh ./scripts/plot.sh