#!/bin/bash
echo "Starting Ablation Experiment Sequence..."

# 1. Complete Model
echo -e "\n========================================================"
echo "Running: complet model, complet dataset"
echo "========================================================"
python main.py

# 2. Without Mass
echo -e "\n========================================================"
echo "Running: without mass model, without mass dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_mass

# 3. Without Attention
echo -e "\n========================================================"
echo "Running: without attention model, complet dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_attn

# 4. Without LSTM
echo -e "\n========================================================"
echo "Running: without lstm model, complet dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_lstm

# 5. Without Global Time Stretching (GTS)
echo -e "\n========================================================"
echo "Running: complet model, without global time stretching dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_GTS

# 6. Without Partial Time Stretching (TPS)
echo -e "\n========================================================"
echo "Running: complet model, without partial time stretching dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_TPS

# 7. Without Noise Injection (NI)
echo -e "\n========================================================"
echo "Running: complet model, without noise injection dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_NI

# 8. Without All Augmentation
echo -e "\n========================================================"
echo "Running: complet model, without all augmentation dataset"
echo "========================================================"
python main.py --model_name CoLANet_wo_aug

# 9. Change Duration 40
echo -e "\n========================================================"
echo "Running: change duration to 40"
echo "========================================================"
python main.py --model_name CoLANet_chg_dura --duration 40

# 10. Change Duration 60
echo -e "\n========================================================"
echo "Running: change duration to 60"
echo "========================================================"
python main.py --model_name CoLANet_chg_dura --duration 60

# 11. Change Duration 80
echo -e "\n========================================================"
echo "Running: change duration to 80"
echo "========================================================"
python main.py --model_name CoLANet_chg_dura --duration 80

# 12. Change Bandwidth 375
echo -e "\n========================================================"
echo "Running: change bandwidth to 375"
echo "========================================================"
python main.py --model_name CoLANet_chg_band --bandwidth 375

# 13. Change Bandwidth 750
echo -e "\n========================================================"
echo "Running: change bandwidth to 750"
echo "========================================================"
python main.py --model_name CoLANet_chg_band --bandwidth 750

# 14. Change Bandwidth 1125
echo -e "\n========================================================"
echo "Running: change bandwidth to 1125"
echo "========================================================"
python main.py --model_name CoLANet_chg_band --bandwidth 1125

echo -e "\nAll ablation experiments completed successfully."