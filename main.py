import argparse
from trainer import LiquidTrainer
from trainer_cla import SoluteCla, ConcentrationCla
from trainer_noise import DenoisedTrainer

def get_args():
    parser = argparse.ArgumentParser(description="Liquid Sugar Content Estimation Project")
    parser.add_argument('--model_name', type=str, default='CoLANet',
                        choices=[
                            'CoLANet', 'CoLANet_wo_aug', 'CoLANet_wo_GTS', 
                            'CoLANet_wo_TPS', 'CoLANet_wo_NI', 'CoLANet_wo_mass', 
                            'CoLANet_wo_lstm', 'CoLANet_wo_attn', 'CoLANet_chg_dura', 
                            'CoLANet_chg_band', 'Solute_cla', 'Concentration_cla', 
                            'Unet_denoise', "ResNet18_pt", "ResNet18",
                            "TCN", "static_channel", "MVUE"
                        ],
                        help='Choose the model variant to use')
    parser.add_argument('--duration', type=int, default=100, help='acoustic duration')
    parser.add_argument('--bandwidth', type=int, default=1500, help='acoustic bandwidth')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'],
                        help='Execution mode: train for training, val for inference')
    parser.add_argument('--root_dir', type=str, default=r"./dataset/merged_dataset_final",
                        help='Root directory for the training dataset')
    parser.add_argument('--robustness_path', type=str, 
                        default='dataset/robustness/',
                        help='Root directory for validation/robustness dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--solute_quantity', type=int, default=None, help='Initial learning rate')
    return parser.parse_args()

def main():
    args = get_args()
    config = vars(args)
    if args.model_name == 'Solute_cla':
        trainer = SoluteCla(config)
    elif args.model_name == 'Concentration_cla':
        trainer = ConcentrationCla(config)
    elif args.model_name == 'Unet_denoise':
        trainer = DenoisedTrainer(config)
    else:
        trainer = LiquidTrainer(config)

    if args.mode == 'train':
        trainer.trainMain()
    elif args.mode == 'val':
        trainer.valMain()
    else:
        print("Unsupported mode")

if __name__ == '__main__':
    main()