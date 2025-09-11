import torch
import os
import csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from ddim_train import BaseTrainer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
from processor.dataset import LEDProcessor, LEDDataset
from models.mtl_ddim_model import DiffusionModel
from models.gnn_model import HeteroLabelEmbeddingGNN
from models.bert_model import HMNeTNERModel
from models.unimo_model import UnimoCRFModel

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

class VTTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        super().__init__(label_map, args, logger, metrics_file)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_num_steps = len(self.train_data) * args.num_epochs if train_data else 0
        self.best_dev_f1 = 0.0
        self.best_dev_epoch = None
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_ner_{args.mode}_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_ner_{args.mode}_final.pth")
        if self.metrics_file:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'stage', 'batch', 'ner_f1', 'crf_loss'])

    def training_settings_init(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape, param.requires_grad)
        """Configure optimizer and scheduler for NER pre-training."""
        # Text parameters
        parameters = []
        params = {'lr':3e-5, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name or 'text' in name:
                params['params'].append(param)
        parameters.append(params)

        # Vision parameters
        params = {'lr':3e-5, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name or 'vision' in name:
                params['params'].append(param)
        parameters.append(params)

        # crf
        params = {'lr':5e-2, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        # Freeze image_model for hvpnet
        for name, par in self.model.named_parameters(): # freeze resnet
            if 'image_model' in name:   par.requires_grad = False

        # Verify no parameter overlap
        param_ids = []
        for group in parameters:
            for param in group['params']:
                param_id = id(param)
                if param_id in param_ids:
                    raise ValueError(f"Parameter {param_id} appears in multiple groups")
                param_ids.append(param_id)

        self.logger.info(f"All model parameters: {[name for name, _ in self.model.named_parameters()]}")
        for i, group in enumerate(parameters):
            self.logger.info(f"Parameter group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, params={len(group['params'])}")

        self.optimizer = AdamW(parameters)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)

        trainable = [name for name, param in self.model.named_parameters() if param.requires_grad]
        frozen = [name for name, param in self.model.named_parameters() if not param.requires_grad]
        self.logger.info(f"Trainable parameters: {len(trainable)}, Frozen parameters: {len(frozen)}")
        self.logger.info(f"Trainable parameter names: {trainable}")

    def train(self, stage="train"):
        """Train the diffusion model for NER pre-training."""
        self.training_settings_init()
        self.model.train()
        self.logger.info(f"***** Running NER *****")
        self.logger.info(f"  Num instances = {len(self.train_data) * self.args.batch_size}")
        self.logger.info(f"  Num epochs = {self.args.num_epochs}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")
        self.logger.info(f"  Gradient accumulation steps = {self.args.grad_accum_steps}")
        self.logger.info(f"  Evaluate begin = {self.args.eval_begin_epoch}")

        self.step = 0
        self.no_improve = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True) as pbar:
            avg_loss = 0
            loss_count = 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                epoch_loss = 0.0
                batch_count = 0
                all_true_labels, all_pred_labels = [], []
                self._batch_idx = 0
                self.optimizer.zero_grad()

                for batch in self.train_data:
                    self.step += 1
                    self._batch_idx += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    loss, true_labels, pred_labels, attention_mask = self._step(batch)
                    if loss is not None:
                        loss = loss / self.args.grad_accum_steps
                        loss.backward()
                        if self.step % self.args.grad_accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()

                        batch_loss = loss.detach().cpu().item() * self.args.grad_accum_steps
                        epoch_loss += batch_loss
                        batch_count += 1
                        avg_loss += batch_loss
                        loss_count += 1
                        if pred_labels is not None:
                            true_labels_batch, pred_labels_batch = self._gen_labels(
                                pred_labels, 
                                true_labels, 
                                attention_mask
                            )
                            all_true_labels.extend(true_labels_batch)
                            all_pred_labels.extend(pred_labels_batch)

                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(f"loss: {avg_loss_display:<6.5f}")
                        avg_loss, loss_count = 0, 0

                if batch_count > 0:
                    micro_f1 = 0.0
                    if all_true_labels and all_pred_labels:
                        results_dict = classification_report(all_true_labels, all_pred_labels, digits=4, zero_division=0, output_dict=True)
                        results_str = classification_report(all_true_labels, all_pred_labels, digits=4, zero_division=0)
                        micro_f1 = results_dict.get('micro avg', {}).get('f1-score', 0.0)
                        self.logger.info(f"***** Epoch {epoch + 1} Train Eval Results *****")
                        self.logger.info("\n%s", results_str)
                    self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Loss: {epoch_loss/batch_count:.4f}, NER Micro F1: {micro_f1:.4f}")

                    if self.metrics_file:
                        with open(self.metrics_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch + 1,
                                stage,
                                batch_count,
                                micro_f1,
                                epoch_loss/batch_count
                            ])

                if epoch >= self.args.eval_begin_epoch:
                    if self.evaluate(stage="val", epoch=epoch):
                        break

            torch.cuda.empty_cache()
            pbar.close()

        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    def evaluate(self, stage="val", epoch=0):
        """Evaluate the diffusion model on NER task."""
        self.model.eval()
        self.logger.info(f"***** Running {stage} evaluation for NER *****")
        self.logger.info(f"  Num instances = {len(self.val_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        if len(self.val_data) == 0:
            self.logger.warning(f"Validation data loader is empty. Skipping evaluation.")
            if self.metrics_file:
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, stage, 0, 0.0, 0.0, 0.0])
            self.model.train()
            return False

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                micro_f1 = self._eval_labels(pbar, self.val_data, stage, epoch)
                self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, NER Micro F1: {micro_f1:.4f}")

                if micro_f1 > self.best_dev_f1:
                    self.best_dev_f1 = micro_f1
                    self.best_dev_epoch = epoch + 1
                    self.no_improve = 0
                    if self.args.save_path:
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.logger.info(f"Saved best model (Micro F1: {micro_f1:.4f}) to {self.best_model_path}")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.args.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        return True

        self.model.train()
        return False

    def test(self, stage="test", epoch=0):
        """Test the diffusion model on NER task."""
        self.model.eval()
        self.logger.info(f"***** Running {stage} testing for NER *****")
        self.logger.info(f"  Num instances = {len(self.test_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        if len(self.test_data) == 0:
            self.logger.warning(f"Test data loader is empty. Skipping testing.")
            if self.metrics_file:
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, stage, 0, 0.0, 0.0, 0.0])
            self.model.train()
            return 0.0

        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                micro_f1 = self._eval_labels(pbar, self.test_data, stage, epoch)
                self.logger.info(f"Test NER Micro F1: {micro_f1:.4f}")

        self.model.train()
        return micro_f1

    def _step(self, batch):
        """Perform a single training or evaluation step."""
        if not hasattr(self, '_batch_idx'):
            self._batch_idx = 0
        self._batch_idx += 1

        expected_len = 10
        if len(batch) != expected_len:
            self.logger.error(f"Expected {expected_len} batch elements, got {len(batch)}")
            raise ValueError(f"Expected {expected_len} batch elements, got {len(batch)}")

        labels, input_ids, token_type_ids, attention_mask, hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words = batch
        words = list(map(list, zip(*words)))
        images, aux_imgs = self._select_images(hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs)

        if self.args.ner_model_name == "hvpnet":
            loss, pred_labels, _ = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                labels=labels, 
                images=images, 
                aux_imgs=aux_imgs
            )
        elif self.args.ner_model_name == "mkgformer":
            loss, pred_labels, _ = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                labels=labels,
                images=images,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs
            )
        else:
            raise ValueError("Invalid ner_model_name")

        return loss, labels, pred_labels, attention_mask

    def _eval_labels(self, pbar, data, stage="val", epoch=0):
        """Evaluate NER labels for validation or test set."""
        self._batch_idx = 0
        all_true_labels, all_pred_labels = [], []
        batch_count = 0
        pbar.set_description_str("Validation" if stage == "val" else "Testing")

        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            _, true_labels, pred_labels, attention_mask = self._step(batch)

            if pred_labels is not None:
                true_labels_batch, pred_labels_batch = self._gen_labels(pred_labels, true_labels, attention_mask)
                all_true_labels.extend(true_labels_batch)
                all_pred_labels.extend(pred_labels_batch)

            batch_count += 1
            pbar.update()

        pbar.close()
        micro_f1 = 0.0
        if all_true_labels and all_pred_labels:
            results_dict = classification_report(all_true_labels, all_pred_labels, digits=4, zero_division=0, output_dict=True)
            results_str = classification_report(all_true_labels, all_pred_labels, digits=4, zero_division=0)
            micro_f1 = results_dict.get('micro avg', {}).get('f1-score', 0.0)
            self.logger.info(f"***** {stage.capitalize()} Eval Results *****")
            self.logger.info("\n%s", results_str)
        else:
            self.logger.warning(f"No labels collected in {stage} phase. Setting micro_f1 to 0.0")

        if self.metrics_file:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, stage, batch_count, micro_f1, 0.0, 0.0])

        return micro_f1

    def _select_images(self, hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs):
        """Select images based on ner_model_name."""
        if self.args.ner_model_name == "hvpnet":
            return hvp_img, hvp_aux_imgs
        elif self.args.ner_model_name == "mkgformer":
            return mkg_img, mkg_aux_imgs
        else:
            raise ValueError(f"Unsupported ner_model_name: {self.args.ner_model_name}")

    def _gen_labels(self, pred_labels, true_labels, token_attention_mask, return_indices=False):
        """Generate NER labels from pred_labels and true_labels, applying attention mask."""
        if isinstance(true_labels, torch.Tensor):
            label_ids = true_labels.detach().cpu().numpy()
        elif isinstance(true_labels, list):
            label_ids = np.array(true_labels)
        else:
            label_ids = true_labels

        if isinstance(token_attention_mask, torch.Tensor):
            token_attention_mask = token_attention_mask.detach().cpu().numpy()
        elif isinstance(token_attention_mask, list):
            token_attention_mask = np.array(token_attention_mask)

        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.detach().cpu().numpy()
        elif not isinstance(pred_labels, list):
            pred_labels = np.array(pred_labels)

        label_map = {idx: label for label, idx in self.label_map.items()}
        special_tokens = []
        for label in ['[CLS]', '[SEP]', 'X']:
            if label in self.label_map:
                special_tokens.append(self.label_map[label])
        special_tokens = set(special_tokens) or {-100}

        given_label_batch, pred_label_batch = [], []
        for row in range(token_attention_mask.shape[0]):
            mask = token_attention_mask[row].astype(bool)
            label_row_masked = label_ids[row][mask] if true_labels is not None else []
            pred_row = pred_labels[row] if isinstance(pred_labels, list) else pred_labels[row][mask]
            given_label_sent, pred_label_sent = [], []

            valid_length = min(len(pred_row), len(label_row_masked))
            for column in range(valid_length):
                true_label = label_row_masked[column]
                if true_label in special_tokens:
                    continue
                if return_indices:
                    given_label_sent.append(int(true_label))
                    pred_label_sent.append(int(pred_row[column]))
                else:
                    given_label_sent.append(label_map.get(true_label, 'O'))
                    pred_label_sent.append(label_map.get(pred_row[column], 'O'))
            given_label_batch.append(given_label_sent)
            pred_label_batch.append(pred_label_sent)

        return given_label_batch, pred_label_batch
    
def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def validate_paths(data_path, imgs_path, aux_imgs_path, rcnn_imgs_path, use_prompt):
    """Validate dataset, image, and model paths."""
    for key, path in data_path.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path {path} for {key} does not exist")
    if use_prompt:
        for path in [imgs_path, aux_imgs_path, rcnn_imgs_path]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"Image path {path} does not exist")
            
if __name__ == "__main__":
    """Main function for NER diffusion model pretraining or finetuning."""
    parser = argparse.ArgumentParser(description="Diffusion model NER pretraining/finetuning.")
    parser.add_argument("--dataset_name", default="twitter15", type=str, choices=['twitter15', 'twitter17'], help="Dataset name.")
    parser.add_argument("--ner_model_name", default="hvpnet", type=str, choices=['hvpnet', 'mkgformer'], help="NER model.")
    parser.add_argument('--vit_name', default='openai/clip-vit-base-patch32', type=str, help="Vision transformer name.")
    parser.add_argument('--num_epochs', default=30, type=int, help="Number of training epochs.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device: cuda or cpu.")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size.")
    parser.add_argument('--grad_accum_steps', default=2, type=int, help="Gradient accumulation steps.")
    parser.add_argument('--warmup_ratio', default=0.01, type=float, help="Warmup ratio for scheduler.")
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="Epoch to start evaluation.")
    parser.add_argument('--seed', default=1234, type=int, help="Random seed.")
    parser.add_argument("--local_cache_path", default="/home/yixin/workspace/huggingface/", type=str, help="Local HuggingFace model cache path.")
    parser.add_argument("--lm_name", default="bert-base-uncased", type=str, help="Pretrained language model.")
    parser.add_argument('--max_seq_len', default=80, type=int, help="Max sequence length.")
    parser.add_argument('--use_prompt', action='store_true', default=True, help="Use visual prompts for HVPNet.")
    parser.add_argument('--prompt_len', default=4, type=int, help="Prompt length for HVPNet.")
    parser.add_argument('--prompt_dim', default=800, type=int, help="Prompt projection layer dimension for HVPNet.")
    parser.add_argument('--load_path', default=None, type=str, help="Path to load pretrained model.")
    parser.add_argument('--save_path', default=".", type=str, help="Path to save models.")
    parser.add_argument('--notes', default="", type=str, help="Notes for save path directory.")
    parser.add_argument('--aux_size', default=128, type=int, help="Auxiliary image size.")
    parser.add_argument('--rcnn_size', default=64, type=int, help="RCNN image size.")
    parser.add_argument('--patience', default=5, type=int, help="Early stopping patience.")
    parser.add_argument("--mode", default="pretrain", type=str, choices=["pretrain", "finetune"], help="Training mode.")

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

    # Configure image and model paths
    imgs_path = IMG_PATH[args.dataset_name] if args.use_prompt else None
    aux_imgs_path = AUX_PATH[args.dataset_name] if args.use_prompt else None
    rcnn_imgs_path = RCNN_PATH[args.dataset_name] if args.use_prompt else None
    data_path = DATA_PATH[args.dataset_name]
    logger.info("Using visual prompts: images enabled." if args.use_prompt else "No visual prompts: text-only encoding.")

    # Validate paths
    validate_paths(data_path, imgs_path, aux_imgs_path, rcnn_imgs_path, args.use_prompt)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if args.use_prompt else None

    # Set random seed
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logdir = os.path.join("logs", f"{args.dataset_name}_bs{args.batch_size}_lr3e-5_{args.notes}")
    os.makedirs(logdir, exist_ok=True)

    # Initialize LEDProcessor
    logger.info("Initializing LEDProcessor...")
    processor = LEDProcessor(args, data_path=data_path)
    label_mapping = processor.get_label_mapping()
    label_embeddings = processor.get_label_embedding().to(args.device)
    num_labels = len(label_mapping)
    logger.info(f"Loaded {num_labels} labels from processor, embeddings shape: {label_embeddings.shape}")

    # Load dataset
    logger.info(f"Loading {args.dataset_name} dataset...")
    dataset = LEDDataset(
        processor=processor,
        transform=transform,
        imgs_path=imgs_path,
        aux_imgs_path=aux_imgs_path,
        rcnn_imgs_path=rcnn_imgs_path,
        max_seq_len=args.max_seq_len,
        mode=args.mode,
        aux_size=args.aux_size,
        rcnn_size=args.rcnn_size
    )
    if len(dataset) == 0:
        raise ValueError(f"{args.mode.capitalize()} dataset is empty.")
    logger.info(f"{args.mode.capitalize()} dataset size: {len(dataset)}")

    # Check dataset sequence length
    max_seq = max(len(sent) for sent in dataset.data_dict["words"])
    logger.info(f"Max sequence length: {max_seq}")
    if max_seq > args.max_seq_len - 2:
        logger.warning(f"Some sequences will be truncated: max_seq={max_seq} > max_seq_len-2={args.max_seq_len-2}")

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
        writer.writerow(['epoch', 'stage', 'batch', 'ner_f1', 'crf_loss'])
    logger.info(f"Logging metrics to {metrics_file}")

    # Visual-textual encoder
    if args.ner_model_name == "hvpnet":
        ner_model = HMNeTNERModel(num_labels, args)
    elif args.ner_model_name == "mkgformer":
        ner_model = UnimoCRFModel(num_labels, args)
    else:
        raise ValueError("Invalid ner_model_name")
    ner_model = ner_model.to(args.device)

    # Initialize trainer
    trainer = VTTrainer(
        train_data=train_dataloader,
        val_data=val_dataloader,
        test_data=test_dataloader,
        model=ner_model,
        label_map=label_mapping,
        args=args,
        logger=logger,
        metrics_file=metrics_file
    )

    # Training
    logger.info(f"Starting the pre-training of '{args.ner_model_name}'...")
    trainer.train()
    logger.info(f"Model {args.ner_model_name} pre-training completed: Best val NER F1={trainer.best_dev_f1:.4f}")
    test_ner_f1 = trainer.test()
    logger.info(f"NER {args.ner_model_name} test NER F1: {test_ner_f1:.4f}")