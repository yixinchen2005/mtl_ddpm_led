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

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define dataset and image paths
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
GNN_PATH = {
    'twitter15': 'gnn/twitter2015',
    'twitter17': 'gnn/twitter2017'
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def validate_paths(data_path, imgs_path, aux_imgs_path, rcnn_imgs_path, clstm_path, gnn_path, use_prompt):
    """Validate dataset, image, and model paths."""
    for key, path in data_path.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path {path} for {key} does not exist")
    if use_prompt:
        for path in [imgs_path, aux_imgs_path, rcnn_imgs_path]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"Image path {path} does not exist")
    for file in ["char_vocab.pkl", "char_lstm.pth"]:
        if not os.path.exists(os.path.join(clstm_path, file)):
            raise FileNotFoundError(f"{file} not found at {clstm_path}/{file}")
    if not os.path.exists(os.path.join(gnn_path, "gnn_hetero_best_decoder.pth")):
        raise FileNotFoundError(f"GNN weights not found at {gnn_path}/gnn_hetero_best_decoder.pth")

def main():
    """Main function for NER diffusion model pretraining or finetuning."""
    parser = argparse.ArgumentParser(description="Diffusion model NER pretraining/finetuning.")
    parser.add_argument("--dataset_name", default="twitter15", type=str, choices=['twitter15', 'twitter17'], help="Dataset name.")
    parser.add_argument("--ner_model_name", default="hvpnet", type=str, choices=['hvpnet', 'mkgformer'], help="NER model.")
    parser.add_argument('--vit_name', default='openai/clip-vit-base-patch32', type=str, help="Vision transformer name.")
    parser.add_argument('--num_epochs', default=15, type=int, help="Number of training epochs.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device: cuda or cpu.")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size.")
    parser.add_argument('--grad_accum_steps', default=2, type=int, help="Gradient accumulation steps.")
    parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--warmup_ratio', default=0.01, type=float, help="Warmup ratio for scheduler.")
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="Epoch to start evaluation.")
    parser.add_argument('--seed', default=2021, type=int, help="Random seed.")
    parser.add_argument("--local_cache_path", default="./cache", type=str, help="Local HuggingFace model cache path.")
    parser.add_argument("--lm_name", default="vinai/bertweet-base", type=str, help="Pretrained language model.")
    parser.add_argument("--char_hidden_dim", default=64, type=int, help="Character-level LSTM hidden dimension.")
    parser.add_argument('--label_hidden_dim', default=768, type=int, help="Label feature input dimension for GNN.")
    parser.add_argument('--time_hidden_dim', default=32, type=int, help="Time embedding hidden dimension.")
    parser.add_argument('--embed_dim', default=128, type=int, help="Dimension for projected features.")
    parser.add_argument('--max_seq_len', default=80, type=int, help="Max sequence length.")
    parser.add_argument('--max_char_len', default=50, type=int, help="Max character length.")
    parser.add_argument('--prompt_len', default=10, type=int, help="Prompt length for hvpnet.")
    parser.add_argument('--prompt_dim', default=800, type=int, help="Prompt projection layer dimension for hvpnet.")
    parser.add_argument('--load_path', default=None, type=str, help="Path to load pretrained model.")
    parser.add_argument('--save_path', default="./models", type=str, help="Path to save models.")
    parser.add_argument('--notes', default="", type=str, help="Notes for save path directory.")
    parser.add_argument('--use_prompt', action='store_true', help="Use visual prompts (images).")
    parser.add_argument('--aux_size', default=128, type=int, help="Auxiliary image size.")
    parser.add_argument('--rcnn_size', default=128, type=int, help="RCNN image size.")
    parser.add_argument('--train_steps', default=1000, type=int, help="Diffusion training timesteps.")
    parser.add_argument('--reverse_steps', default=10, type=int, help="Diffusion inference timesteps.")
    parser.add_argument('--patience', default=5, type=int, help="Early stopping patience.")
    parser.add_argument('--noise_rate', default=0.1, type=float, help="Fraction of labels to corrupt during pretraining.")
    parser.add_argument("--mode", default="pretrain", type=str, choices=["pretrain", "finetune"], help="Training mode.")
    parser.add_argument('--compute_eval_loss', action='store_true', help="Compute loss during validation and testing.")

    args = parser.parse_args()

    # Validate arguments
    if args.dataset_name not in DATA_PATH:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")
    if args.num_epochs < 1:
        raise ValueError("Number of epochs must be positive.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")
    if args.load_path and not os.path.exists(args.load_path):
        raise ValueError(f"Load path {args.load_path} does not exist.")
    if args.noise_rate <= 0 or args.noise_rate >= 1:
        raise ValueError("Noise rate must be between 0 and 1.")
    if args.embed_dim < 1:
        raise ValueError("Embedding dimension must be positive.")

    # Configure image and model paths
    imgs_path = IMG_PATH[args.dataset_name] if args.use_prompt else None
    aux_imgs_path = AUX_PATH[args.dataset_name] if args.use_prompt else None
    rcnn_imgs_path = RCNN_PATH[args.dataset_name] if args.use_prompt else None
    data_path = DATA_PATH[args.dataset_name]
    clstm_path = CLSTM_PATH[args.dataset_name]
    gnn_path = GNN_PATH[args.dataset_name]
    logger.info("Using visual prompts: images enabled." if args.use_prompt else "No visual prompts: text-only encoding.")

    # Validate paths
    validate_paths(data_path, imgs_path, aux_imgs_path, rcnn_imgs_path, clstm_path, gnn_path, args.use_prompt)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if args.use_prompt else None

    # Set random seed
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logdir = os.path.join("logs", f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr}_ed{args.embed_dim}{args.notes}")
    os.makedirs(logdir, exist_ok=True)

    # Initialize LEDProcessor
    logger.info("Initializing LEDProcessor...")
    processor = LEDProcessor(args, data_path=data_path, clstm_path=clstm_path)
    label_mapping = processor.get_label_mapping()
    label_embeddings = processor.get_label_embedding().to(args.device)
    num_labels = len(label_mapping)
    logger.info(f"Loaded {num_labels} labels from processor, embeddings shape: {label_embeddings.shape}")

    # Load dataset
    logger.info(f"Loading {args.mode} dataset...")
    dataset = LEDDataset(
        processor=processor,
        transform=transform,
        imgs_path=imgs_path,
        aux_imgs_path=aux_imgs_path,
        max_seq_len=args.max_seq_len,
        max_char_len=args.max_char_len,
        mode=args.mode,
        aux_size=args.aux_size,
        rcnn_imgs_path=rcnn_imgs_path,
        rcnn_size=args.rcnn_size
    )
    if len(dataset) == 0:
        raise ValueError(f"{args.mode.capitalize()} dataset is empty.")
    logger.info(f"{args.mode.capitalize()} dataset size: {len(dataset)}")

    # Check dataset sequence and token lengths
    max_seq = max(len(sent) for sent in dataset.data_dict["words"])
    max_char = max(len(token) for sent in dataset.data_dict["words"] for token in sent)
    logger.info(f"Max sequence length: {max_seq}, Max token length: {max_char}")
    if max_seq > args.max_seq_len - 2:
        logger.warning(f"Some sequences will be truncated: max_seq={max_seq} > max_seq_len-2={args.max_seq_len-2}")
    if max_char > args.max_char_len:
        logger.warning(f"Some tokens will be truncated: max_char={max_char} > max_char_len={args.max_char_len}")

    # Split dataset (80/10/10)
    train_size = int(0.8 * len(dataset))
    val_size = max(1, int(0.1 * len(dataset)))
    test_size = max(1, len(dataset) - train_size - val_size)
    logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed)
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
    logger.info(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Initialize metrics logging
    metrics_file = os.path.join(logdir, "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'stage', 'batch', 'loss', 'mse_loss', 'crf_loss', 'ner_f1'])
    logger.info(f"Logging metrics to {metrics_file}")

    # Initialize diffusion model
    model = DiffusionModel(
        args=args,
        num_labels=num_labels,
        label_embeddings=label_embeddings,
        gnn_path=gnn_path,
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

    # Training
    logger.info(f"Starting {args.mode} with task='ner_{args.mode}'...")
    trainer.train(task=f"ner_{args.mode}")
    logger.info(f"NER {args.mode} completed: Best val NER F1={trainer.best_dev_f1:.4f}")
    ner_model_path = os.path.join(args.save_path, f"ner_{args.mode}_ed{args.embed_dim}.pth")
    torch.save(model.state_dict(), ner_model_path)
    logger.info(f"Saved NER {args.mode} model to {ner_model_path}")
    test_ner_f1 = trainer.test(task=f"ner_{args.mode}")
    logger.info(f"NER {args.mode} test NER F1: {test_ner_f1:.4f}")

if __name__ == "__main__":
    main()