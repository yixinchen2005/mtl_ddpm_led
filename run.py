import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
import csv
from processor.dataset import LEDProcessor, LEDDataset
from models.mtl_ddpm_model import DiffusionModel
from modules.ddpm_train import PreTrainer

# Configure logging for training and evaluation
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define dataset paths for Twitter15 and Twitter17
DATA_PATH = {
    'twitter15': {
        'pretrain': 'data/NER_data/twitter2015/unlabeled.txt',
        'finetune': 'data/NER_data/twitter2015/labeled.txt',
        'auximgs': 'data/NER_data/twitter2015/twitter2015_aux_dict.pth',
        'img2crop': 'data/NER_data/twitter15_detect/twitter15_img2crop.pth'
    },
    'twitter17': {
        'pretrain': 'data/NER_data/twitter2017/unlabeled.txt',
        'finetune': 'data/NER_data/twitter2017/labeled.txt',
        'auximgs': 'data/NER_data/twitter2017/twitter2017_aux_dict.pth',
        'img2crop': 'data/NER_data/twitter17_detect/twitter17_img2crop.pth'
    }
}

# Define image paths for visual prompts
IMG_PATH = {
    'twitter15': 'data/NER_data/twitter2015_images',
    'twitter17': 'data/NER_data/twitter2017_images'
}

AUX_PATH = {
    'twitter15': 'data/NER_data/twitter2015_aux_images/crops',
    'twitter17': 'data/NER_data/twitter2017_aux_images/crops'
}

RCNN_PATH = {
    'twitter15': 'data/NER_data/',
    'twitter17': 'data/NER_data/'
}

CLSTM_PATH = {
    'twitter15': 'char_lstm/twitter2015',
    'twitter17': 'char_lstm/twitter2017'
}

def set_seed(seed):
    """Set random seed for reproducibility across PyTorch, NumPy, and Python."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    """Main function to orchestrate diffusion model pretraining or fine-tuning for error detection."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Script for diffusion model error detection tasks.")
    parser.add_argument("--dataset_name", default="twitter15", type=str, choices=['twitter15', 'twitter17'], help="Dataset name.")
    parser.add_argument("--ner_model_name", default="hvpnet", type=str, help="NER model (hvpnet or mkgformer).")
    parser.add_argument('--vit_name', default='openai/clip-vit-base-patch32', type=str, help="Vision transformer name.")
    parser.add_argument('--num_epochs', default=15, type=int, help="Number of training epochs.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device: cuda or cpu.")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--finetune_lr', default=5e-6, type=float, help="Fine-tuning learning rate.")
    parser.add_argument('--warmup_ratio', default=0.01, type=float, help="Warmup ratio for scheduler.")
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="Epoch to start evaluation.")
    parser.add_argument('--seed', default=2021, type=int, help="Random seed.")
    parser.add_argument("--local_cache_path", default="./cache", type=str, help="Local HuggingFace model cache path.")
    parser.add_argument("--lm_name", default="bert-base-uncased", type=str, help="Pretrained language model.")
    parser.add_argument("--char_hidden_dim", default=512, type=int, help="Character-level LSTM hidden dimension.")
    parser.add_argument('--label_hidden_dim', default=256, type=int, help="Label feature hidden dimension.")
    parser.add_argument('--time_hidden_dim', default=256, type=int, help="Time embedding hidden dimension.")
    parser.add_argument('--prompt_len', default=10, type=int, help="Prompt length.")
    parser.add_argument('--prompt_dim', default=800, type=int, help="Prompt projection layer dimension.")
    parser.add_argument('--noise_dim', default=128, type=int, help="Noise dimension.")
    parser.add_argument('--load_path', default=None, type=str, help="Path to load pretrained model.")
    parser.add_argument('--save_path', default="./models", type=str, help="Path to save models.")
    parser.add_argument('--notes', default="", type=str, help="Notes for save path directory.")
    parser.add_argument('--do_error_pretrain', action='store_true', help="Run diffusion pretraining for error detection on unlabeled data.")
    parser.add_argument('--do_error_fine_tune', action='store_true', help="Run diffusion fine-tuning for error detection on labeled data.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="Maximum sequence length.")
    parser.add_argument("--max_char_len", default=128, type=int, help="Maximum character length.")
    parser.add_argument('--use_prompt', action='store_true', help="Use visual prompts (images).")
    parser.add_argument('--crf_lr', default=5e-2, type=float, help="CRF learning rate.")
    parser.add_argument('--prompt_lr', default=3e-4, type=float, help="Prompt learning rate.")
    parser.add_argument('--aux_size', default=128, type=int, help="Auxiliary image size.")
    parser.add_argument('--rcnn_size', default=128, type=int, help="RCNN image size.")
    parser.add_argument('--train_steps', default=1000, type=int, help="Diffusion training timesteps.")
    parser.add_argument('--reverse_steps', default=20, type=int, help="Diffusion inference timesteps.")
    parser.add_argument('--patience', default=5, type=int, help="Early stopping patience.")
    parser.add_argument('--lambda_id', default=1.0, type=float, help="Identity loss weight.")
    parser.add_argument('--lambda_edit', default=1.0, type=float, help="Edit loss weight.")
    parser.add_argument('--lambda_cycle', default=1.0, type=float, help="Cycle loss weight.")
    parser.add_argument('--lambda_contrast', default=1.0, type=float, help="Contrastive loss weight.")
    parser.add_argument('--noise_rate', default=0.1, type=float, help="Fraction of labels to corrupt during pretraining.")

    args = parser.parse_args()

    # Validate exactly one mode is selected
    modes = [args.do_error_pretrain, args.do_error_fine_tune]
    if sum(modes) != 1:
        raise ValueError("Exactly one mode must be set: --do_error_pretrain or --do_error_fine_tune.")

    # Validate arguments
    if args.dataset_name not in DATA_PATH:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")
    if args.num_epochs < 1:
        raise ValueError("Number of epochs must be positive.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.load_path and not os.path.exists(args.load_path):
        raise ValueError(f"Load path {args.load_path} does not exist.")
    if args.noise_rate <= 0 or args.noise_rate >= 1:
        raise ValueError("Noise rate must be between 0 and 1.")

    # Configure image paths for visual prompts
    if args.use_prompt:
        imgs_path = IMG_PATH[args.dataset_name]
        aux_imgs_path = AUX_PATH[args.dataset_name]
        rcnn_imgs_path = RCNN_PATH[args.dataset_name]
        logger.info("Using visual prompts: images enabled.")
    else:
        imgs_path = aux_imgs_path = rcnn_imgs_path = None
        logger.info("No visual prompts: using text-only encoding.")

    data_path = DATA_PATH[args.dataset_name]
    clstm_path = CLSTM_PATH[args.dataset_name]

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if args.use_prompt else None

    # Set random seed
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logdir = os.path.join("logs", f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr}{args.notes}")
    os.makedirs(logdir, exist_ok=True)

    # Initialize LEDProcessor
    logger.info("Initializing LEDProcessor...")
    processor = LEDProcessor(data_path, clstm_path, args)
    label_mapping = processor.get_label_mapping()
    label_embeddings = processor.get_label_embedding().to(args.device)
    num_labels = len(label_mapping)
    logger.info(f"Loaded {num_labels} labels from processor.")

    # Initialize datasets
    unlabeled_dataset = None
    labeled_dataset = None

    # Load unlabeled dataset for pretraining
    if args.do_error_pretrain:
        logger.info("Loading unlabeled dataset...")
        unlabeled_dataset = LEDDataset(
            processor=processor,
            transform=transform,
            imgs_path=imgs_path,
            aux_imgs_path=aux_imgs_path,
            max_seq_len=args.max_seq_len,
            max_char_len=args.max_char_len,
            mode="pretrain",
            aux_size=args.aux_size,
            rcnn_imgs_path=rcnn_imgs_path,
            rcnn_size=args.rcnn_size
        )
        if len(unlabeled_dataset) == 0:
            raise ValueError("Unlabeled dataset is empty.")
        logger.info(f"Unlabeled dataset size: {len(unlabeled_dataset)}")

    # Load labeled dataset for fine-tuning
    if args.do_error_fine_tune:
        logger.info("Loading labeled dataset...")
        labeled_dataset = LEDDataset(
            processor=processor,
            transform=transform,
            imgs_path=imgs_path,
            aux_imgs_path=aux_imgs_path,
            max_seq_len=args.max_seq_len,
            max_char_len=args.max_char_len,
            mode="finetune",
            aux_size=args.aux_size,
            rcnn_imgs_path=rcnn_imgs_path,
            rcnn_size=args.rcnn_size
        )
        if len(labeled_dataset) == 0:
            raise ValueError("Labeled dataset is empty; cannot proceed with fine-tuning.")
        logger.info(f"Labeled dataset size: {len(labeled_dataset)}")

    # Split datasets into train/val/test
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    if args.do_error_pretrain and unlabeled_dataset:
        # Split unlabeled dataset (80/10/10)
        train_size = int(0.8 * len(unlabeled_dataset))
        val_size = int(0.1 * len(unlabeled_dataset))
        test_size = len(unlabeled_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            unlabeled_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        logger.info(f"Unlabeled dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    elif args.do_error_fine_tune and labeled_dataset:
        # Split labeled dataset (70/15/15)
        train_size = int(0.7 * len(labeled_dataset))
        val_size = int(0.15 * len(labeled_dataset))
        test_size = len(labeled_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            labeled_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        logger.info(f"Labeled dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Initialize metrics logging
    metrics_file = os.path.join(logdir, "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'task', 'stage', 'loss', 'error_precision', 'error_recall', 'error_f1'])
    logger.info(f"Logging metrics to {metrics_file}")

    # Initialize diffusion model
    model = DiffusionModel(
        args=args,
        num_labels=num_labels,
        label_embedding_table=label_embeddings,
        clstm_path=clstm_path,
        ner_model_name=args.ner_model_name
    ).to(args.device)

    # Load pretrained model if specified
    if args.load_path:
        logger.info(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path))
        logger.info("Model loaded successfully.")

    # Initialize trainer
    trainer = PreTrainer(
        train_data=train_dataloader,
        val_data=val_dataloader,
        test_data=test_dataloader,
        model=model,
        label_map=label_mapping,
        args=args,
        logger=logger,
        metrics_file=metrics_file
    )

    # Error Pretraining
    if args.do_error_pretrain:
        if not train_dataloader:
            raise ValueError("Unlabeled dataset not loaded; cannot perform error pretraining.")
        logger.info("Starting error pretraining with task='error_pretrain'...")
        trainer.train(task="error_pretrain")
        logger.info(f"Error pretraining completed: Best val error_f1={trainer.best_error_f1:.4f}")
        error_pretrain_model_path = os.path.join(args.save_path, "error_pretrain.pth")
        torch.save(model.state_dict(), error_pretrain_model_path)
        logger.info(f"Saved error pretrained model to {error_pretrain_model_path}")
        test_error_f1 = trainer.test(task="error_pretrain")
        logger.info(f"Error pretraining test error_f1: {test_error_f1:.4f}")

    # Error Fine-tuning
    if args.do_error_fine_tune:
        if not train_dataloader:
            raise ValueError("Labeled dataset not loaded; cannot perform error fine-tuning.")
        if args.load_path or os.path.exists(os.path.join(args.save_path, "error_pretrain.pth")):
            load_path = args.load_path or os.path.join(args.save_path, "error_pretrain.pth")
            logger.info(f"Loading pretrained error model from {load_path}")
            model.load_state_dict(torch.load(load_path))
            logger.info("Model loaded successfully.")
        logger.info("Starting error fine-tuning with task='error_finetune'...")
        trainer.train(task="error_finetune")
        logger.info(f"Error fine-tuning completed: Best val error_f1={trainer.best_error_f1:.4f}")
        error_finetune_model_path = os.path.join(args.save_path, "error_finetune.pth")
        torch.save(model.state_dict(), error_finetune_model_path)
        logger.info(f"Saved error fine-tuned model to {error_finetune_model_path}")
        test_error_f1 = trainer.test(task="error_finetune")
        logger.info(f"Error fine-tuning test error_f1: {test_error_f1:.4f}")

if __name__ == "__main__":
    main()