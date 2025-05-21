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
from models.bert_model import HMNeTNERModel
from models.unimo_model import UnimoCRFModel
from modules.ddpm_train import PreTrainer, NERTrainer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define dataset paths
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

# Define image paths
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

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# Main function to orchestrate training, fine-tuning, or prediction
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Main script for NER/diffusion pretraining, fine-tuning, or prediction with mutually exclusive modes.")
    parser.add_argument("--dataset_name", default="twitter15", type=str, choices=['twitter15', 'twitter17'], help="The name of dataset.")
    parser.add_argument("--ner_model_name", default="hvpnet", type=str, help="The name of supporting NER model (hvpnet or mkgformer).")
    parser.add_argument('--vit_name', default='openai/clip-vit-base-patch32', type=str, help="The name of vision transformer.")
    parser.add_argument('--num_epochs', default=15, type=int, help="Number of training epochs.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device: cuda or cpu.")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--finetune_lr', default=5e-6, type=float, help="Learning rate for fine-tuning.")
    parser.add_argument('--warmup_ratio', default=0.01, type=float, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="Epoch to start evaluation.")
    parser.add_argument('--seed', default=2021, type=int, help="Random seed.")
    parser.add_argument("--local_cache_path", default="./cache", type=str, help="Path to local HuggingFace model cache.")
    parser.add_argument("--lm_name", default="bert-base-uncased", type=str, help="Pretrained language model.")
    parser.add_argument("--char_hidden_dim", default=512, type=int, help="Dimension of Character-level LSTM hidden embeddings.")
    parser.add_argument('--label_hidden_dim', default=256, type=int, help="Hidden dimensions of label features.")
    parser.add_argument('--time_hidden_dim', default=256, type=int, help="Hidden dimensions of time.")
    parser.add_argument('--prompt_len', default=10, type=int, help="Prompt length.")
    parser.add_argument('--prompt_dim', default=800, type=int, help="Mid dimension of prompt project layer.")
    parser.add_argument('--noise_dim', default=128, type=int, help="Dimensions of noises.")
    parser.add_argument('--load_path', default=None, type=str, help="Path to load model for fine-tuning or NER training.")
    parser.add_argument('--save_path', default="./models", type=str, help="Path to save model.")
    parser.add_argument('--notes', default="", type=str, help="Remarks for save path directory.")
    parser.add_argument("--do_ner_pretrain", action="store_true", help="Run NER pretraining on unlabeled dataset.")
    parser.add_argument("--do_ner_fine_tune", action="store_true", help="Run NER fine-tuning on labeled dataset.")
    parser.add_argument('--do_diffusion_pretrain', action='store_true', help="Run diffusion pretraining on unlabeled dataset.")
    parser.add_argument('--do_diffusion_fine_tune', action='store_true', help="Run diffusion fine-tuning on labeled dataset.")
    parser.add_argument('--predict', action='store_true', help="Run prediction with 70/15/15 cross-validation.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="Maximum sequence length.")
    parser.add_argument("--max_char_len", default=128, type=int, help="Maximum character length.")
    parser.add_argument('--use_prompt', action='store_true', help="Use visual prompts (images) for hvpnet/mkgformer.")
    parser.add_argument('--crf_lr', default=5e-2, type=float, help="CRF learning rate.")
    parser.add_argument('--prompt_lr', default=3e-4, type=float, help="Prompt learning rate.")
    parser.add_argument('--aux_size', default=128, type=int, help="Auxiliary image size.")
    parser.add_argument('--rcnn_size', default=128, type=int, help="RCNN image size.")
    parser.add_argument('--train_steps', default=1000, type=int, help="Timesteps for diffusion training.")
    parser.add_argument('--reverse_steps', default=50, type=int, help="Timesteps for diffusion inference.")
    parser.add_argument('--patience', default=5, type=int, help="Threshold for early stopping.")
    parser.add_argument('--lambda_id', default=1.0, type=float, help="Identity loss weight.")
    parser.add_argument('--lambda_edit', default=1.0, type=float, help="Edit loss weight.")
    parser.add_argument('--lambda_cycle', default=1.0, type=float, help="Cycle loss weight.")
    parser.add_argument('--lambda_contrast', default=1.0, type=float, help="Contrastive loss weight.")
    parser.add_argument('--noise_rate', default=0.3, type=float, help="Synthetic noise rate for low error rate.")

    args = parser.parse_args()

    # Validate that exactly one mode is selected
    modes = [args.predict, args.do_ner_pretrain, args.do_ner_fine_tune, args.do_diffusion_pretrain, args.do_diffusion_fine_tune]
    if sum(modes) != 1:
        raise ValueError("Exactly one of --predict, --do_ner_pretrain, --do_ner_fine_tune, --do_diffusion_pretrain, or --do_diffusion_fine_tune must be set.")

    # Validate dataset and arguments
    if args.dataset_name not in DATA_PATH:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")
    if args.num_epochs < 1:
        raise ValueError("Number of epochs must be positive.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.load_path and not os.path.exists(args.load_path):
        raise ValueError(f"Load path {args.load_path} does not exist.")

    # Configure image paths if prompts are used
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

    # Define image transformations if prompts are used
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

    # Initialize LEDProcessor for data processing
    logger.info("Initializing LEDProcessor...")
    processor = LEDProcessor(data_path, clstm_path, args)
    label_mapping = processor.get_label_mapping()
    label_embeddings = processor.get_label_embedding().to(args.device)
    num_labels = len(label_mapping)
    logger.info(f"Loaded {num_labels} labels from processor.")

    unlabeled_dataset = None
    labeled_dataset = None

    # Load unlabeled dataset for pretraining or prediction
    if args.do_ner_pretrain or args.do_diffusion_pretrain or args.predict:
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

    # Load labeled dataset for fine-tuning or prediction
    if args.do_ner_fine_tune or args.do_diffusion_fine_tune or args.predict:
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

    # Split labeled dataset into train/val/test (70/15/15)
    train_dataloader_labeled = None
    val_dataloader_labeled = None
    test_dataloader_labeled = None
    if labeled_dataset:
        train_sz = int(0.7 * len(labeled_dataset))
        val_sz = int(0.15 * len(labeled_dataset))
        test_sz = len(labeled_dataset) - train_sz - val_sz
        train_dataset_labeled, val_dataset_labeled, test_dataset_labeled = random_split(
            labeled_dataset, [train_sz, val_sz, test_sz], generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataloader_labeled = DataLoader(
            train_dataset_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_dataloader_labeled = DataLoader(
            val_dataset_labeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        test_dataloader_labeled = DataLoader(
            test_dataset_labeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        logger.info(f"Labeled dataset: train={len(train_dataset_labeled)}, val={len(val_dataset_labeled)}, test={len(test_dataset_labeled)}")

    # Training and fine-tuning logic
    if not args.predict:
        metrics_file = os.path.join(logdir, "metrics.csv")
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'task', 'stage', 'loss', 'ner_f1', 'diffusion_f1', 'error_f1'])
        logger.info(f"Logging metrics to {metrics_file}")

        # Split unlabeled dataset into train/val/test (80/10/10)
        train_dataloader_unlabeled = None
        val_dataloader_unlabeled = None
        test_dataloader_unlabeled = None
        if unlabeled_dataset and (args.do_ner_pretrain or args.do_diffusion_pretrain):
            train_size_unlabeled = int(0.8 * len(unlabeled_dataset))
            val_size_unlabeled = int(0.1 * len(unlabeled_dataset))
            test_size_unlabeled = len(unlabeled_dataset) - train_size_unlabeled - val_size_unlabeled
            train_dataset_unlabeled, val_dataset_unlabeled, test_dataset_unlabeled = random_split(
                unlabeled_dataset, [train_size_unlabeled, val_size_unlabeled, test_size_unlabeled], 
                generator=torch.Generator().manual_seed(args.seed)
            )
            train_dataloader_unlabeled = DataLoader(
                train_dataset_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
            val_dataloader_unlabeled = DataLoader(
                val_dataset_unlabeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            test_dataloader_unlabeled = DataLoader(
                test_dataset_unlabeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            logger.info(f"Unlabeled dataset: train={len(train_dataset_unlabeled)}, val={len(val_dataset_unlabeled)}, test={len(test_dataset_unlabeled)}")

        # Initialize model based on task
        model = None
        if args.do_diffusion_pretrain or args.do_diffusion_fine_tune:
            model = DiffusionModel(
                args=args,
                num_labels=num_labels,
                label_embedding_table=label_embeddings,
                clstm_path=clstm_path,
                ner_model_name=args.ner_model_name
            ).to(args.device)
        elif args.do_ner_pretrain or args.do_ner_fine_tune:
            if args.ner_model_name == "hvpnet":
                model = HMNeTNERModel(num_labels=num_labels, args=args).to(args.device)
            elif args.ner_model_name == "mkgformer":
                model = UnimoCRFModel(num_labels=num_labels, args=args).to(args.device)
            else:
                raise ValueError(f"Unsupported NER model: {args.ner_model_name}")

        # Load pretrained model if specified
        if model and args.load_path:
            logger.info(f"Loading model from {args.load_path}")
            model.load_state_dict(torch.load(args.load_path))
            logger.info("Model loaded successfully.")

        # NER Pretraining
        if args.do_ner_pretrain:
            if not train_dataloader_unlabeled:
                raise ValueError("Unlabeled dataset not loaded; cannot perform NER pretraining.")
            if not model:
                raise ValueError("Model not initialized for NER pretraining.")
            trainer = NERTrainer(
                train_data=train_dataloader_unlabeled,
                val_data=val_dataloader_unlabeled,
                test_data=test_dataloader_unlabeled,
                model=model,
                label_map=label_mapping,
                args=args,
                logger=logger,
                metrics_file=os.path.join(args.save_path, f"metrics_ner_pretrain_{args.ner_model_name}.csv")
            )
            logger.info(f"Starting NER pretraining with {args.ner_model_name}...")
            trainer.train(task="ner_pretrain")
            logger.info(f"NER pretraining completed: Best val ner_f1={trainer.best_dev:.4f}")
            test_ner_f1 = trainer.test(task="ner_pretrain")
            logger.info(f"NER pretraining test ner_f1: {test_ner_f1:.4f}")
            ner_pretrain_model_path = os.path.join(args.save_path, "ner_pretrain.pth")
            torch.save(model.state_dict(), ner_pretrain_model_path)
            logger.info(f"Saved NER pretrained model to {ner_pretrain_model_path}")

        # NER Fine-tuning
        if args.do_ner_fine_tune:
            if not train_dataloader_labeled:
                raise ValueError("Labeled dataset not loaded; cannot perform NER fine-tuning.")
            if not model:
                raise ValueError("Model not initialized for NER fine-tuning.")
            if args.load_path or os.path.exists(os.path.join(args.save_path, "ner_pretrain.pth")):
                load_path = args.load_path or os.path.join(args.save_path, "ner_pretrain.pth")
                logger.info(f"Loading pretrained NER model from {load_path}")
                model.load_state_dict(torch.load(load_path))
            trainer = NERTrainer(
                train_data=train_dataloader_labeled,
                val_data=val_dataloader_labeled,
                test_data=test_dataloader_labeled,
                model=model,
                label_map=label_mapping,
                args=args,
                logger=logger,
                metrics_file=os.path.join(args.save_path, f"metrics_ner_finetune_{args.ner_model_name}.csv")
            )
            logger.info(f"Starting NER fine-tuning with {args.ner_model_name}...")
            trainer.train(task="ner_finetune")
            error_f1 = trainer.test(task="ner_finetune")
            logger.info(f"NER fine-tuning test Error Detection F1: {error_f1:.4f}")
            ner_finetune_model_path = os.path.join(args.save_path, "ner_finetune.pth")
            torch.save(model.state_dict(), ner_finetune_model_path)
            logger.info(f"Saved NER fine-tuned model to {ner_finetune_model_path}")

        # Diffusion Pretraining
        if args.do_diffusion_pretrain:
            if not train_dataloader_unlabeled:
                raise ValueError("Unlabeled dataset not loaded; cannot perform diffusion pretraining.")
            if not model:
                raise ValueError("Model not initialized for diffusion pretraining.")
            trainer = PreTrainer(
                train_data=train_dataloader_unlabeled,
                val_data=val_dataloader_unlabeled,
                test_data=test_dataloader_unlabeled,
                model=model,
                label_map=label_mapping,
                args=args,
                logger=logger,
                metrics_file=metrics_file
            )
            logger.info("Starting diffusion pretraining with task='diffusion_pretrain'...")
            trainer.train(task="diffusion_pretrain")
            logger.info(f"Diffusion pretraining completed: Best val ner_f1={trainer.best_ner_f1:.4f}, diffusion_f1={trainer.best_diffusion_f1:.4f}")
            diffusion_pretrain_model_path = os.path.join(args.save_path, "diffusion_pretrain.pth")
            torch.save(model.state_dict(), diffusion_pretrain_model_path)
            logger.info(f"Saved diffusion pretrained model to {diffusion_pretrain_model_path}")
            test_ner_f1, test_diffusion_f1 = trainer.test(task="diffusion_pretrain")
            logger.info(f"Diffusion pretraining test ner_f1: {test_ner_f1:.4f}, diffusion_f1: {test_diffusion_f1:.4f}")

        # Diffusion Fine-tuning
        if args.do_diffusion_fine_tune:
            if not train_dataloader_labeled:
                raise ValueError("Labeled dataset not loaded; cannot fine-tune.")
            if not model:
                raise ValueError("Model not initialized for diffusion fine-tuning.")
            trainer = PreTrainer(
                train_data=train_dataloader_labeled,
                val_data=val_dataloader_labeled,
                test_data=test_dataloader_labeled,
                model=model,
                label_map=label_mapping,
                args=args,
                logger=logger,
                metrics_file=metrics_file
            )
            logger.info("Starting diffusion fine-tuning with task='diffusion_finetune'...")
            trainer.train(task="diffusion_finetune")
            error_f1 = trainer.test(task="diffusion_finetune")
            logger.info(f"Diffusion fine-tuning test Error Detection F1: {error_f1:.4f}")
            diffusion_finetune_model_path = os.path.join(args.save_path, "diffusion_finetune.pth")
            torch.save(model.state_dict(), diffusion_finetune_model_path)
            logger.info(f"Saved diffusion fine-tuned model to {diffusion_finetune_model_path}")

    # Prediction mode with cross-validation
    else:
        num_rounds = 5
        test_error_f1_scores = []

        # Define split sizes for cross-validation
        total_samples = len(unlabeled_dataset)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        test_size = total_samples - train_size - val_size
        logger.info(f"Prediction cross-validation split sizes: train={train_size}, val={val_size}, test={test_size}")

        indices = list(range(total_samples))
        random.shuffle(indices, random=lambda: 0.5)

        # Perform cross-validation rounds
        for round_idx in range(num_rounds):
            logger.info(f"Starting prediction round {round_idx + 1}/{num_rounds}")

            round_save_path = os.path.join(args.save_path, f"round_{round_idx}")
            os.makedirs(round_save_path, exist_ok=True)
            round_metrics_file = os.path.join(logdir, f"metrics_round_{round_idx}.csv")
            with open(round_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'task', 'stage', 'loss', 'ner_f1', 'diffusion_f1', 'error_f1'])
            logger.info(f"Logging metrics for round {round_idx + 1} to {round_metrics_file}")

            # Define indices for train/val/test splits
            test_start = round_idx * test_size
            test_end = min(test_start + test_size, total_samples)
            test_indices = indices[test_start:test_end]

            val_start = (test_end) % total_samples
            val_end = min(val_start + val_size, total_samples)
            val_indices = indices[val_start:val_end]
            if val_end < val_start + val_size:
                val_indices.extend(indices[:val_size - (val_end - val_start)])

            train_indices = [i for i in indices if i not in test_indices and i not in val_indices]
            train_indices = train_indices[:train_size]

            # Create datasets for the round
            train_dataset_unlabeled = torch.utils.data.Subset(unlabeled_dataset, train_indices)
            val_dataset_unlabeled = torch.utils.data.Subset(unlabeled_dataset, val_indices)
            test_dataset_unlabeled = torch.utils.data.Subset(unlabeled_dataset, test_indices)

            train_dataloader_unlabeled = DataLoader(
                train_dataset_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
            val_dataloader_unlabeled = DataLoader(
                val_dataset_unlabeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            test_dataloader_unlabeled = DataLoader(
                test_dataset_unlabeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            logger.info(f"Round {round_idx + 1} dataset: train={len(train_dataset_unlabeled)}, val={len(val_dataset_unlabeled)}, test={len(test_dataset_unlabeled)}")

            # Initialize diffusion model
            model = DiffusionModel(
                args=args,
                num_labels=num_labels,
                label_embedding_table=label_embeddings,
                clstm_path=clstm_path,
                ner_model_name=args.ner_model_name
            ).to(args.device)

            # Train and evaluate for the round
            trainer = PreTrainer(
                train_data=train_dataloader_unlabeled,
                val_data=val_dataloader_unlabeled,
                test_data=None,
                model=model,
                label_map=label_mapping,
                args=args,
                logger=logger,
                metrics_file=round_metrics_file
            )
            logger.info(f"Round {round_idx + 1}: Starting diffusion pretraining with task='diffusion_pretrain'...")
            trainer.train(task="diffusion_pretrain")
            logger.info(f"Round {round_idx + 1}: Diffusion pretraining completed: Best val ner_f1={trainer.best_ner_f1:.4f}, diffusion_f1={trainer.best_diffusion_f1:.4f}")
            pretrain_model_path = os.path.join(round_save_path, f"diffusion_pretrain_round_{round_idx}.pth")
            torch.save(model.state_dict(), pretrain_model_path)
            logger.info(f"Saved diffusion pretrained model for round {round_idx + 1} to {pretrain_model_path}")

if __name__ == "__main__":
    main()