import torch
import os
import csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from .metrics import comp_f1_score

# Base class for training models
class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, metrics_file=None):
        """Initialize base trainer with label map, arguments, logger, and metrics file."""
        self.label_map = label_map
        self.args = args
        self.logger = logger
        self.metrics_file = metrics_file
        self.refresh_step = 2
        self.no_improve = 0
        self.step = 0
        self.max_grad_norm = 1.0

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()

# Trainer for pretraining and fine-tuning the diffusion model
class PreTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        """Initialize trainer with data, model, and configuration."""
        super().__init__(label_map, args, logger, metrics_file)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_num_steps = len(self.train_data) * args.num_epochs if train_data else 0
        self.best_dev = 0
        self.best_dev_epoch = None
        self.best_error_f1 = 0.0
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_{'error_pretrain' if args.do_diffusion_pretrain else 'error_finetune'}_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_{'error_pretrain' if args.do_diffusion_pretrain else 'error_finetune'}_final.pth")
        # Initialize metrics CSV file
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'task', 'stage', 'loss', 'error_precision', 'error_recall', 'error_f1'])

    def train(self, task="error_pretrain", stage="train", epoch=0):
        """Train the model for pretraining or fine-tuning."""
        # Set up optimizer and scheduler
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model.train()
        self.logger.info(f"***** Running {'error pre-training' if task == 'error_pretrain' else 'error fine-tuning'} *****")
        self.logger.info(f"  Num instances = {len(self.train_data) * self.args.batch_size}")
        self.logger.info(f"  Num epochs = {self.args.num_epochs}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")
        self.logger.info(f"  Learning rate = {self.args.lr}")
        self.logger.info(f"  Evaluate begin = {self.args.eval_begin_epoch}")

        self.step = 0
        self.no_improve = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True) as pbar:
            avg_loss, loss_count = 0, 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                epoch_loss = 0.0
                error_labels, error_preds, error_masks = [], [], []

                for batch in self.train_data:
                    self.step += 1
                    # Move batch to device
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    loss, error_logits, error_mask_pred, targets_batch, attention_mask, words, img_names = self._step(
                        batch, task, stage, epoch
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                    # Update loss tracking
                    batch_loss = loss.detach().cpu().item()
                    avg_loss += batch_loss
                    loss_count += 1
                    epoch_loss += batch_loss

                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(f"loss: {avg_loss_display:<6.5f}")
                        avg_loss, loss_count = 0, 0

                    # Collect labels and predictions for metrics
                    targets = targets_batch[1] if isinstance(targets_batch, tuple) else targets_batch
                    error_labels_batch, error_pred_batch = self._gen_labels(error_logits, targets, attention_mask, return_indices=True)
                    error_mask_pred_batch = (torch.sigmoid(error_mask_pred) > 0.5).long().detach().cpu().numpy()
                    error_mask_true_batch = error_mask_pred_batch if task == "error_pretrain" else (targets_batch[0] != targets).float().detach().cpu().numpy()
                    error_labels.extend(error_labels_batch)
                    error_preds.extend(error_pred_batch)
                    error_masks.extend([error_mask_true_batch[i][attention_mask[i].cpu().numpy().astype(bool)] for i in range(len(error_mask_true_batch))])

                # Compute and log error detection metrics
                error_precision, error_recall, error_f1 = comp_f1_score(error_labels, [e for e in error_labels], error_preds, error_masks)
                self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Error Detection Precision: {error_precision:.4f}, Recall: {error_recall:.4f}, F1: {error_f1:.4f}")

                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, task, "train", epoch_loss / len(self.train_data), error_precision, error_recall, error_f1])

                # Evaluate on validation set
                if epoch >= self.args.eval_begin_epoch:
                    if self.evaluate(task, stage="val", epoch=epoch):
                        break

            torch.cuda.empty_cache()
            pbar.close()

        # Save final model
        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    def evaluate(self, task="error_pretrain", stage="val", epoch=0):
        """Evaluate the model on validation or test set."""
        self.model.eval()
        self.logger.info(f"***** Running {stage} evaluation for {task} *****")
        self.logger.info(f"  Num instances = {len(self.val_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, val_loss = self._eval_labels(pbar, self.val_data, epoch, task, stage)
                error_precision, error_recall, error_f1 = metrics
                self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Error Detection Precision: {error_precision:<6.5f}, Recall: {error_recall:<6.5f}, F1: {error_f1:<6.5f}")
                metric_for_saving = error_f1

                # Save best model based on error F1
                if metric_for_saving > self.best_error_f1:
                    self.best_error_f1 = metric_for_saving
                    self.best_dev = metric_for_saving
                    self.best_dev_epoch = epoch + 1
                    self.no_improve = 0
                    if self.args.save_path:
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.logger.info(f"Saved best model (F1: {metric_for_saving:.4f}) to {self.best_model_path}")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.args.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        return True

                # Log metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, task, stage, val_loss, error_precision, error_recall, error_f1])

        self.model.train()
        return False

    def test(self, task="error_pretrain", stage="test", epoch=0):
        """Test the model on the test set."""
        self.model.eval()
        self.logger.info(f"***** Running {stage} testing for {task} *****")
        self.logger.info(f"  Num instances = {len(self.test_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        # Load best model if available
        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, test_loss = self._eval_labels(pbar, self.test_data, epoch, task, stage)
                error_precision, error_recall, error_f1 = metrics
                self.logger.info(f"Test Error Detection Precision: {error_precision:<6.5f}, Recall: {error_recall:<6.5f}, F1: {error_f1:<6.5f}")

                # Log test metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0, task, stage, test_loss, error_precision, error_recall, error_f1])

        self.model.train()
        return error_f1

    def _step(self, batch, task="error_pretrain", stage="train", epoch=0):
        """Process a single training or evaluation step."""
        # Unpack batch based on task and prompt usage
        if self.args.use_prompt:
            if task == "error_finetune":
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
            else:
                (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = targets_unk
        else:
            if task == "error_finetune":
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
            else:
                (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = targets_unk
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))

        # Select images based on model type
        imgs, aux_imgs = self._select_images(hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs)

        if stage == "train":
            if task == "error_pretrain":
                # Generate targets_noise for context
                targets_noise, error_mask_true = self.model.inject_random_noise(targets_unk, attention_mask)
                loss, recon_emissions, error_logits = self.model(
                    labels=targets_unk,
                    targets_unk=targets_noise,  # Use targets_noise as context
                    char_input_ids=char_input_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    images=imgs,
                    aux_imgs=aux_imgs,
                    rcnn_imgs=rcnn_imgs,
                    mode="pretrain",
                    epoch=epoch
                )
                error_mask_pred = error_logits
            elif task == "error_finetune":
                loss, recon_emissions, error_logits = self.model(
                    labels=None,  # Not used in fine-tuning
                    targets_unk=targets_unk,
                    targets_new=targets_batch[1],
                    char_input_ids=char_input_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    images=imgs,
                    aux_imgs=aux_imgs,
                    rcnn_imgs=rcnn_imgs,
                    mode="finetune",
                    epoch=epoch
                )
                error_mask_pred = error_logits
        elif stage in ["val", "test"]:
            # Use reverse diffusion for inference
            context_labels = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
            error_logits, error_mask_pred = self.model.reverse_diffusion(
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                context_labels=context_labels,
                steps=getattr(self.args, 'reverse_steps', 20),
                temperature=0.8
            )
            targets = targets_batch[1] if isinstance(targets_batch, tuple) else targets_batch
            loss, recon_emissions, error_logits_inner = self.model(
                labels=None,
                targets_unk=targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch,
                targets_new=targets_batch[1] if isinstance(targets_batch, tuple) else None,
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                mode="pretrain" if task == "error_pretrain" else "finetune",
                epoch=epoch
            )
            error_mask_pred = error_mask_pred.float()

        self.logger.debug(f"Epoch {epoch}, Task {task}, Stage {stage}, Total Loss: {loss:.4f}")
        return loss, error_logits, error_mask_pred, targets_batch, attention_mask, words, img_names
    
    def _select_images(self, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs):
        """Select appropriate images based on model type."""
        if self.model.ner_model_name == "hvpnet":
            return hvp_imgs, hvp_aux_imgs
        elif self.model.ner_model_name == "mkgformer":
            return mkg_imgs, mkg_aux_imgs
        return None, None

    def _gen_labels(self, logits, targets, token_attention_mask, return_indices=False):
        """Generate predicted and true labels, filtering out padding and special tokens."""
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)
        if targets is None:
            label_ids = np.zeros_like(logits)
        elif isinstance(targets, torch.Tensor):
            label_ids = targets.detach().cpu().numpy()
        elif isinstance(targets, list):
            label_ids = np.array(targets)
        if isinstance(token_attention_mask, torch.Tensor):
            token_attention_mask = token_attention_mask.detach().cpu().numpy()
        elif isinstance(token_attention_mask, list):
            token_attention_mask = np.array(token_attention_mask)
        label_map = {idx: label for label, idx in self.label_map.items()}
        given_label_batch, pred_label_batch = [], []

        for row in range(token_attention_mask.shape[0]):
            mask = token_attention_mask[row].astype(bool)
            label_row_masked = label_ids[row][mask] if targets is not None else np.zeros(sum(mask), dtype=np.int64)
            pred_row = logits[row][mask]
            given_label_sent, pred_label_sent = [], []
            for column in range(len(label_row_masked)):
                if column == 0 or (label_map.get(label_row_masked[column], '') in ["X", "[SEP]"]):
                    continue
                if return_indices:
                    given_label_sent.append(int(label_row_masked[column]))
                    pred_label_sent.append(int(pred_row[column]))
                else:
                    given_label_sent.append(label_map.get(label_row_masked[column], 'O'))
                    pred_label_sent.append(label_map.get(pred_row[column], 'O'))
            given_label_batch.append(given_label_sent)
            pred_label_batch.append(pred_label_sent)

        return given_label_batch, pred_label_batch

    def _eval_labels(self, pbar, data, epoch, task="error_pretrain", stage="val"):
        """Evaluate labels and compute error detection metrics."""
        given_labels, error_pred_labels, error_masks = [], [], []
        total_loss = 0.0
        batch_count = 0
        pbar.set_description_str("Validation" if stage == "val" else "Testing")

        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, error_logits, error_mask_pred, targets_batch, attention_mask, _, _ = self._step(
                batch, task, stage, epoch
            )
            total_loss += loss.detach().cpu().item() if loss is not None else 0.0
            batch_count += 1

            targets = targets_batch[1] if isinstance(targets_batch, tuple) else targets_batch
            given_labels_batch, error_pred_labels_batch = self._gen_labels(error_logits, targets, attention_mask, return_indices=True)
            error_mask_pred_batch = error_mask_pred.long().detach().cpu().numpy()
            error_mask_true_batch = (targets_batch[0] != targets).float().detach().cpu().numpy() if isinstance(targets_batch, tuple) else error_mask_pred_batch
            given_labels.extend(given_labels_batch)
            error_pred_labels.extend(error_pred_labels_batch)
            error_masks.extend([error_mask_true_batch[i][attention_mask[i].cpu().numpy().astype(bool)] for i in range(len(error_mask_true_batch))])

            pbar.update()
        pbar.close()

        error_precision, error_recall, error_f1 = comp_f1_score(given_labels, [e for e in given_labels], error_pred_labels, error_masks) if given_labels else (0.0, 0.0, 0.0)
        return (error_precision, error_recall, error_f1), total_loss / batch_count

    def training_settings_text_only(self):
        """Configure optimizer and scheduler for text-only training."""
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name:
                param.requires_grad = False
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-2)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)

    def training_settings_with_prompt(self):
        """Configure optimizer and scheduler for training with prompts."""
        parameters = []
        params = {'lr': 5e-2, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc') or 'noise_pred' in name or 'error_pred' in name:
                params['params'].append(param)
        parameters.append(params)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if ('ner_model' in name and 'image_model' not in name and 'crf' not in name and not name.startswith('fc')) or \
               'vt_encoder' in name or 'time_mlp' in name or 'norm_' in name or '_attn' in name:
                params['params'].append(param)
        parameters.append(params)
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name or 'ner_model.image_model' in name:
                param.requires_grad = False
        self.optimizer = AdamW(parameters, weight_decay=1e-2)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)