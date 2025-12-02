#!/bin/bash
echo "Starting Robustness Experiment Sequence..."
echo -e "\n========================================================"
echo "Training denoised model"
echo "========================================================"
python main.py --model_name Unet_denoise --mode train

echo "Select random acoustic data, add noise and save. Using denoised model to get and save denoised acoustic data"
echo "========================================================"
python main.py --model_name Unet_denoise --mode val
echo -e "\n========================================================"
echo "Inference on robustness data"
python main.py --model_name CoLANet --mode val