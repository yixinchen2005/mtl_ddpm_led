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

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Main script for NER pretraining and fine-tuning.")
    parser.add_argument("--dataset_name", default="twitter15", type=str, choices=['twitter15', 'twitter17'], help="The name of dataset.")
    parser.add_argument("--ner_model_name", default="hvpnet", type=str, help="The name of supporting NER model (hvpnet or mkgformer).")
    parser.add_argument('--vit_name', default='openai/clip-vit-base-patch32', type=str, help="The name of vision transformer.")
    parser.add_argument('--num_epochs', default=30, type=int, help="Number of training epochs.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device: cuda or cpu.")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
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
    parser.add_argument('--load_path', default=None, type=str, help="Path to load model for fine-tuning or prediction.")
    parser.add_argument('--save_path', default="./models", type=str, help="Path to save model.")
    parser.add_argument('--notes', default="", type=str, help="Remarks for save path directory.")
    parser.add_argument("--do_ner_train", action="store_true", help="Run NER training on unlabeled dataset with noisy targets_unk.")
    parser.add_argument('--do_pretrain', action='store_true', help="Run pre-training on unlabeled dataset.")
    parser.add_argument('--do_fine_tune', action='store_true', help="Run fine-tuning on labeled dataset.")
    parser.add_argument('--predict', action='store_true', help="Run prediction on labeled test dataset.")
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

    args = parser.parse_args()

    # Validate arguments
    if args.dataset_name not in DATA_PATH:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")
    if args.num_epochs < 1:
        raise ValueError("Number of epochs must be positive.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.load_path and not os.path.exists(args.load_path):
        raise ValueError(f"Load path {args.load_path} does not exist.")

    # Set image paths based on use_prompt
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

    # Initialize transform for hvpnet (used only when use_prompt=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if args.use_prompt else None

    # Set random seed
    set_seed(args.seed)

    # Create directories for saving models and logs
    os.makedirs(args.save_path, exist_ok=True)
    logdir = os.path.join("logs", f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr}{args.notes}")
    os.makedirs(logdir, exist_ok=True)
    metrics_file = os.path.join(logdir, "metrics.csv")

    # Initialize CSV file with updated header
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'mode', 'loss', 'ner_f1', 'diffusion_f1', 'error_f1'])
    logger.info(f"Logging metrics to {metrics_file}")

    # Initialize processor
    logger.info("Initializing LEDProcessor...")
    processor = LEDProcessor(data_path, clstm_path, args)
    label_mapping = processor.get_label_mapping()
    label_embeddings = processor.get_label_embedding().to(args.device)
    num_labels = len(label_mapping)
    logger.info(f"Loaded {num_labels} labels from processor.")

    # Initialize datasets and dataloaders
    train_dataloader_unlabeled = val_dataloader_unlabeled = test_dataloader_unlabeled = None
    train_dataloader_labeled = val_dataloader_labeled = test_dataloader_labeled = None

    # Load unlabeled dataset for pretraining and NER training
    if args.do_pretrain or args.do_ner_train:
        logger.info("Loading unlabeled dataset for pretraining/NER training...")
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

        # Split unlabeled dataset (80% train, 10% val, 10% test)
        train_sz = int(0.8 * len(unlabeled_dataset))
        val_sz = int(0.1 * len(unlabeled_dataset))
        test_sz = len(unlabeled_dataset) - train_sz - val_sz
        train_dataset_unlabeled, val_dataset_unlabeled, test_dataset_unlabeled = random_split(
            unlabeled_dataset, [train_sz, val_sz, test_sz], generator=torch.Generator().manual_seed(args.seed)
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

    # Load labeled dataset for fine-tuning and prediction
    if args.do_fine_tune or args.predict:
        logger.info("Loading labeled dataset for fine-tuning/prediction...")
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
            raise ValueError("Labeled dataset is empty; cannot proceed with fine-tuning or prediction.")
        logger.info(f"Labeled dataset size: {len(labeled_dataset)}")

        # Split labeled dataset (70% train, 15% val, 15% test)
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

    # Initialize model
    logger.info("Initializing model...")
    if args.do_pretrain or args.do_fine_tune:
        model = DiffusionModel(
            args=args,
            num_labels=num_labels,
            label_embedding_table=label_embeddings,
            clstm_path=clstm_path,
            ner_model_name=args.ner_model_name
        ).to(args.device)
    elif args.do_ner_train:
        if args.ner_model_name == "hvpnet":
            model = HMNeTNERModel(num_labels=num_labels, args=args).to(args.device)
        elif args.ner_model_name == "mkgformer":
            model = UnimoCRFModel(num_labels=num_labels, args=args).to(args.device)
        else:
            raise ValueError(f"Unsupported NER model: {args.ner_model_name}")
    else:
        logger.warning("No training mode specified; skipping model initialization.")
        model = None

    # Load pretrained model if specified
    if model and args.load_path:
        logger.info(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path))
        logger.info("Model loaded successfully.")

    # NER Training (using unlabeled dataset with noisy targets_unk)
    if args.do_ner_train:
        if not train_dataloader_unlabeled:
            raise ValueError("Unlabeled dataset not loaded; cannot perform NER training.")
        if not model:
            raise ValueError("Model not initialized for NER training.")
        trainer = NERTrainer(
            train_data=train_dataloader_unlabeled,
            val_data=val_dataloader_unlabeled,
            test_data=test_dataloader_unlabeled,
            model=model,
            label_map=label_mapping,
            args=args,
            logger=logger,
            metrics_file=os.path.join(args.save_path, f"metrics_ner_{args.ner_model_name}.csv")
        )
        logger.info(f"Starting NER training with {args.ner_model_name} on unlabeled dataset (noisy targets_unk)...")
        trainer.train()
        ner_f1 = trainer.test()
        logger.info(f"NER test F1: {ner_f1:.4f}")

    # Pre-training (using unlabeled dataset)
    if args.do_pretrain:
        if not train_dataloader_unlabeled:
            raise ValueError("Unlabeled dataset not loaded; cannot perform pretraining.")
        if not model:
            raise ValueError("Model not initialized for pretraining.")
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
        logger.info("Starting pre-training with mode='pretrain'...")
        trainer.train(mode="pretrain")
        diffusion_f1 = trainer.test(mode="pretrain")
        logger.info(f"Pre-training test Diffusion F1: {diffusion_f1:.4f}")

    # Fine-tuning (using labeled dataset with targets_unk and targets_new)
    if args.do_fine_tune:
        if not train_dataloader_labeled:
            raise ValueError("Labeled dataset not loaded; cannot fine-tune.")
        if not model:
            raise ValueError("Model not initialized for fine-tuning.")
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
        logger.info("Starting fine-tuning with mode='finetune_forward'...")
        trainer.train(mode="finetune_forward")
        error_f1 = trainer.test(mode="finetune_forward")
        logger.info(f"Fine-tuning test Error Detection F1: {error_f1:.4f}")

    # # Prediction (using labeled test dataset for error detection)
    # if args.predict:
    #     if not test_dataloader_labeled:
    #         raise ValueError("Labeled test dataset not loaded; cannot predict.")
    #     if not model:
    #         model = DiffusionModel(
    #             args=args,
    #             num_labels=num_labels,
    #             label_embedding_table=label_embeddings,
    #             clstm_path=clstm_path,
    #             ner_model_name=args.ner_model_name
    #         ).to(args.device)
    #         if args.load_path:
    #             logger.info(f"Loading model from {args.load_path}")
    #             model.load_state_dict(torch.load(args.load_path))
    #             logger.info("Model loaded successfully.")
    #     trainer = PreTrainer(
    #         train_data=None,
    #         val_data=None,
    #         test_data=test_dataloader_labeled,
    #         model=model,
    #         label_map=label_mapping,
    #         args=args,
    #         logger=logger,
    #         metrics_file=metrics_file
    #     )
    #     logger.info("Generating predictions with mode='finetune_forward'...")
    #     # Run test and collect predictions
    #     with torch.no_grad():
    #         predictions = []
    #         for batch in test_dataloader_labeled:
    #             batch = [tup.to(args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
    #             loss, ner_logits, diffusion_logits, targets_batch, attention_mask, words, img_names = trainer._step(
    #                 batch, mode="test", epoch=0
    #             )
    #             targets_unk = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
    #             targets_new = targets_batch[1] if isinstance(targets_batch, tuple) else None
    #             _, pred_labels = trainer._gen_labels(diffusion_logits, targets_new, attention_mask)
    #             _, true_labels = trainer._gen_labels(targets_new, targets_new, attention_mask) if targets_new is not None else ([], [])
    #             for i in range(len(words)):
    #                 predictions.append((img_names[i], words[i], pred_labels[i], true_labels[i] if true_labels else []))
        
    #     # Save predictions to CSV
    #     output_file = os.path.join(args.save_path, f"{args.dataset_name}_predictions.csv")
    #     with open(output_file, 'w', encoding='utf-8', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['img_name', 'word', 'pred_label', 'true_label'])
    #         for img_name, word_list, pred_label_list, true_label_list in predictions:
    #             for word, pred_label, true_label in zip(word_list, pred_label_list, true_label_list or [''] * len(word_list)):
    #                 if word not in ['[CLS]', '[SEP]', '[PAD]'] and pred_label != label_mapping.get('[PAD]', 'O'):
    #                     writer.writerow([img_name, word, pred_label, true_label])
    #     logger.info(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()