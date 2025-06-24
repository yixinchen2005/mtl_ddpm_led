import torch
import os
import csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, metrics_file=None):
        """Initialize base trainer with label_map, arguments, logger, and metrics file.
        
        Args:
            label_map (dict): Mapping of NER label indices to names.
            args: Training arguments (e.g., batch_size, lr).
            logger: Logger for training progress.
            metrics_file (str): File path to save metrics.
        """
        self.label_map = label_map
        self.args = args
        self.logger = logger
        self.metrics_file = metrics_file
        self.refresh_step = 2  # Frequency to update progress bar
        self.no_improve = 0  # Counter for early stopping
        self.step = 0  # Global training step counter
        self.max_grad_norm = 1.0  # Gradient clipping threshold

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()

class PreTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        """Initialize trainer with data, model, and configuration for error detection and NER.
        
        Args:
            train_data: Training dataset loader.
            val_data: Validation dataset loader.
            test_data: Test dataset loader.
            model: Diffusion model for error detection and NER.
            label_map (dict): Mapping of NER label indices to names.
            args: Training arguments.
            logger: Logger for training progress.
            metrics_file (str): File path to save metrics.
        """
        super().__init__(label_map, args, logger, metrics_file)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_num_steps = len(self.train_data) * args.num_epochs if train_data else 0
        self.best_dev = 0  # Best validation metric (error_f1)
        self.best_dev_epoch = None
        self.best_error_f1 = 0.0
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_error_{'pretrain' if args.do_error_pretrain else 'finetune'}_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_error_{'pretrain' if args.do_error_pretrain else 'finetune'}_final.pth")
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'task', 'stage', 'batch', 'loss', 'alignment_loss', 'mse_loss', 
                                'denoise_ce_loss', 'error_focal_loss', 'contrastive_loss', 
                                'error_precision', 'error_recall', 'error_f1'])

    def train(self, task="error_pretrain", stage="train", epoch=0):
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
            avg_alignment_loss = 0.0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                epoch_loss = 0.0
                epoch_alignment_loss = 0.0
                epoch_mse_loss = 0.0
                epoch_denoise_ce_loss = 0.0
                epoch_error_focal_loss = 0.0
                epoch_contrastive_loss = 0.0
                batch_count = 0
                all_true_masks, all_pred_masks, all_attn_masks = [], [], []
                self._batch_idx = 0

                for batch in self.train_data:
                    self.step += 1
                    self._batch_idx += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    loss, error_pred_mask, error_true_mask, targets_batch, attention_mask, words, img_names = self._step(
                        batch, task, stage, epoch
                    )
                    if loss is not None:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()

                        batch_loss = loss.detach().cpu().item()
                        batch_alignment_loss = self.model.alignment_loss.detach().cpu().item() if self.model.alignment_loss is not None else 0.0
                        batch_mse_loss = self.model.mse_loss.item() if self.model.mse_loss is not None else 0.0
                        batch_denoise_ce_loss = self.model.denoise_ce_loss.item() if self.model.denoise_ce_loss is not None else 0.0
                        batch_error_focal_loss = self.model.error_focal_loss.item() if self.model.error_focal_loss is not None else 0.0
                        batch_contrastive_loss = self.model.contrastive_loss.item() if self.model.contrastive_loss is not None else 0.0
                        avg_loss += batch_loss
                        avg_alignment_loss += batch_alignment_loss
                        epoch_loss += batch_loss
                        epoch_alignment_loss += batch_alignment_loss
                        epoch_mse_loss += batch_mse_loss
                        epoch_denoise_ce_loss += batch_denoise_ce_loss
                        epoch_error_focal_loss += batch_error_focal_loss
                        epoch_contrastive_loss += batch_contrastive_loss
                        batch_count += 1

                        pred_mask_batch = (torch.sigmoid(error_pred_mask) > 0.5).long().detach().cpu().numpy()
                        true_mask_batch = error_true_mask.detach().cpu().numpy()
                        attn_mask_batch = attention_mask.detach().cpu().numpy()
                        all_true_masks.extend(true_mask_batch)
                        all_pred_masks.extend(pred_mask_batch)
                        all_attn_masks.extend(attn_mask_batch)

                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(f"loss: {avg_loss_display:<6.5f}")
                        avg_loss, loss_count = 0, 0

                if batch_count > 0:
                    error_precision, error_recall, error_f1 = self.compute_f1_score(all_true_masks, all_pred_masks, all_attn_masks)
                    avg_loss_epoch = epoch_loss / batch_count
                    avg_alignment_loss_epoch = epoch_alignment_loss / batch_count
                    avg_mse_loss_epoch = epoch_mse_loss / batch_count
                    avg_denoise_ce_loss_epoch = epoch_denoise_ce_loss / batch_count
                    avg_error_focal_loss_epoch = epoch_error_focal_loss / batch_count
                    avg_contrastive_loss_epoch = epoch_contrastive_loss / batch_count
                    self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Error Detection Precision: {error_precision:.4f}, Recall: {error_recall:.4f}, F1: {error_f1:.4f}, Alignment Loss: {avg_alignment_loss_epoch:.4f}")

                    if self.metrics_file:
                        with open(self.metrics_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch + 1,
                                task,
                                "train",
                                0,
                                avg_loss_epoch,
                                avg_alignment_loss_epoch,
                                avg_mse_loss_epoch,
                                avg_denoise_ce_loss_epoch,
                                avg_error_focal_loss_epoch,
                                avg_contrastive_loss_epoch,
                                error_precision,
                                error_recall,
                                error_f1
                            ])

                if epoch >= self.args.eval_begin_epoch:
                    if self.evaluate(task, stage="val", epoch=epoch):
                        break

            torch.cuda.empty_cache()
            pbar.close()

        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    def evaluate(self, task="error_pretrain", stage="val", epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running {stage} evaluation for {task} *****")
        self.logger.info(f"  Num instances = {len(self.val_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, val_loss, val_alignment_loss = self._eval_labels(pbar, self.val_data, epoch, task, stage)
                error_precision, error_recall, error_f1 = metrics
                self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Error Detection Precision: {error_precision:<6.5f}, Recall: {error_recall:<6.5f}, F1: {error_f1:<6.5f}, Alignment Loss: {val_alignment_loss:<6.5f}")
                metric_for_saving = error_f1

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

        self.model.train()
        return False

    def test(self, task="error_pretrain", stage="test", epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running {stage} testing for {task} *****")
        self.logger.info(f"  Num instances = {len(self.test_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, test_loss, test_alignment_loss = self._eval_labels(pbar, self.test_data, epoch, task, stage)
                error_precision, error_recall, error_f1 = metrics
                self.logger.info(f"Test Error Detection Precision: {error_precision:<6.5f}, Recall: {error_recall:<6.5f}, F1: {error_f1:<6.5f}, Alignment Loss: {test_alignment_loss:<6.5f}")

                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch + 1,
                            task,
                            stage,
                            0,
                            test_loss,
                            test_alignment_loss,
                            self.model.mse_loss.item() if self.model.mse_loss is not None else 0.0,
                            self.model.denoise_ce_loss.item() if self.model.denoise_ce_loss is not None else 0.0,
                            self.model.error_focal_loss.item() if self.model.error_focal_loss is not None else 0.0,
                            self.model.contrastive_loss.item() if self.model.contrastive_loss is not None else 0.0,
                            error_precision,
                            error_recall,
                            error_f1
                        ])

        self.model.train()
        return error_f1

    def _step(self, batch, task="error_pretrain", stage="train", epoch=0):
        if not hasattr(self, '_batch_idx'):
            self._batch_idx = 0
        self._batch_idx += 1

        if self.args.use_prompt:
            if task == "error_finetune":
                assert len(batch) == 13, f"Expected 13 batch elements, got {len(batch)}"
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
            else:
                assert len(batch) == 13, f"Expected 13 batch elements, got {len(batch)}"
                (targets_unk, targets_noise, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = (targets_unk, targets_noise)
        else:
            if task == "error_finetune":
                assert len(batch) == 8, f"Expected 8 batch elements, got {len(batch)}"
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
            else:
                assert len(batch) == 8, f"Expected 8 batch elements, got {len(batch)}"  # Fixed assertion
                (targets_unk, targets_noise, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = (targets_unk, targets_noise)
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))

        imgs, aux_imgs = self._select_images(hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs)

        if task == "error_pretrain":
            labels = targets_unk
            targets_noise = targets_noise
        else:
            labels = targets_new
            targets_noise = targets_unk

        if stage == "train":
            loss, recon_emissions, error_pred_logits, error_true_mask = self.model(
                labels=labels,
                targets_noise=targets_noise,
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
        elif stage in ["val", "test"]:
            context_labels = targets_noise if task == "error_pretrain" else targets_unk
            pred_labels, error_pred_logits = self.model.reverse_diffusion(
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
            loss, recon_emissions, error_pred_logits_model, error_true_mask = self.model(
                labels=labels,
                targets_noise=targets_noise,
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

        return loss, error_pred_logits, error_true_mask, targets_batch, attention_mask, words, img_names

    def _eval_labels(self, pbar, data, epoch, task="error_pretrain", stage="val"):
        self._batch_idx = 0
        all_true_masks, all_pred_masks = [], []
        total_loss = 0.0
        total_alignment_loss = 0.0
        total_mse_loss = 0.0
        total_denoise_ce_loss = 0.0
        total_error_focal_loss = 0.0
        total_contrastive_loss = 0.0
        batch_count = 0
        pbar.set_description_str("Validation" if stage == "val" else "Testing")

        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, error_pred_logits, error_true_mask, targets_batch, attention_mask, _, _ = self._step(
                batch, task, stage, epoch
            )
            if loss is not None:
                total_loss += loss.detach().cpu().item()
                total_alignment_loss += self.model.alignment_loss.detach().cpu().item() if self.model.alignment_loss is not None else 0.0
                total_mse_loss += self.model.mse_loss.item() if self.model.mse_loss is not None else 0.0
                total_denoise_ce_loss = self.model.denoise_ce_loss.item() if self.model.denoise_ce_loss is not None else 0.0
                total_error_focal_loss += self.model.error_focal_loss.item() if self.model.error_focal_loss is not None else 0.0
                total_contrastive_loss += self.model.contrastive_loss.item() if self.model.contrastive_loss is not None else 0.0
                batch_count += 1

            pred_mask_batch = (torch.sigmoid(error_pred_logits) > 0.5).long().detach().cpu().numpy()
            true_mask_batch = error_true_mask.detach().cpu().numpy()
            attn_mask_batch = attention_mask.detach().cpu().numpy()
            all_true_masks.extend(true_mask_batch)
            all_pred_masks.extend(pred_mask_batch)

            pbar.update()
        pbar.close()

        error_precision, error_recall, error_f1 = self.compute_f1_score(all_true_masks, all_pred_masks)
        
        if self.metrics_file and batch_count > 0:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    task,
                    stage,
                    0,
                    total_loss / batch_count,
                    total_alignment_loss / batch_count,
                    total_mse_loss / batch_count,
                    total_denoise_ce_loss / batch_count,
                    total_error_focal_loss / batch_count,
                    total_contrastive_loss / batch_count,
                    error_precision,
                    error_recall,
                    error_f1
                ])

        return (error_precision, error_recall, error_f1), total_loss / batch_count, total_alignment_loss / batch_count
    
    def _select_images(self, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs):
        if self.model.ner_model_name == "hvpnet":
            return hvp_imgs, hvp_aux_imgs
        elif self.model.ner_model_name == "mkgformer":
            return mkg_imgs, mkg_aux_imgs
        return None, None

    def _gen_labels(self, logits, targets, token_attention_mask, return_indices=False):
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
            label_row_masked = label_ids[row][mask] if targets is not None else []
            pred_row = logits[row][mask]
            given_label_sent, pred_label_sent = [], []
            for column in range(len(label_row_masked)):
                if column == 0 or label_map.get(label_row_masked[column], '') in ["X", "[SEP]"]:
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

    def compute_f1_score(self, true_labels, pred_labels, attention_masks=None):
        """Compute precision, recall, and F1 score for binary error detection using masks."""
        true_flat, pred_flat = [], []
        if attention_masks:
            for true_mask, pred_mask, attn_mask in zip(true_labels, pred_labels, attention_masks):
                mask = attn_mask.astype(bool)  # Use full attention mask
                true_label_masked = true_mask[mask]
                pred_label_masked = pred_mask[mask]
                true_flat.extend(true_label_masked)
                pred_flat.extend(pred_label_masked)
        else:
            for true_mask, pred_mask in zip(true_labels, pred_labels):
                true_flat.extend(true_mask)
                pred_flat.extend(pred_mask)
        if not true_flat or all(t == 0 for t in true_flat):
            return 0.0, 0.0, 0.0
        precision = precision_score(true_flat, pred_flat, average='binary', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='binary', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='binary', zero_division=0)
        return precision, recall, f1

    def training_settings_text_only(self):
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name.lower():
                param.requires_grad = False
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-2)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)

    def training_settings_with_prompt(self):
        parameters = []
        params = {'lr': 5e-2, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'crf' in name.lower() or name.lower().startswith('fc') or 'noise_pred' in name.lower() or 'error_pred' in name.lower():
                params['params'].append(param)
        parameters.append(params)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if ('ner_model' in name.lower() and 'image_model' not in name.lower() and 'crf' not in name.lower() and not name.lower().startswith('fc')) or \
               'vt_encoder' in name.lower() or 'time_mlp' in name.lower() or 'norm_' in name.lower() or '_attn' in name.lower():
                params['params'].append(param)
        parameters.append(params)
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name.lower() or 'ner_model.image_model' in name.lower():
                param.requires_grad = False
        self.optimizer = AdamW(parameters)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)