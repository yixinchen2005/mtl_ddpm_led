import torch
import os
import csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from .metrics import comp_f1_score

# Base class for trainers, providing common attributes and abstract methods
class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, metrics_file=None) -> None:
        # Initialize label mapping, arguments, logger, and metrics file
        self.label_map = label_map
        self.args = args
        self.logger = logger
        self.metrics_file = metrics_file
        self.refresh_step = 2  # Frequency for updating progress bar
        self.no_improve = 0  # Counter for early stopping
        self.step = 0  # Training step counter

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()

# Trainer for NER tasks, supporting pretraining and fine-tuning
class NERTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        super().__init__(label_map, args, logger, metrics_file)
        # Initialize datasets, model, and training parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_num_steps = len(self.train_data) * args.num_epochs if train_data else 0
        self.best_dev = 0  # Best validation F1 score
        self.best_dev_epoch = None  # Epoch with best validation F1
        self.model = model
        self.optimizer = None
        self.scheduler = None
        # Define paths for saving best and final models
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_ner_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_ner_final.pth")
        self.checkpoint_dir = os.path.join(args.save_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Initialize metrics CSV file with headers
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'task', 'stage', 'loss', 'ner_f1', 'error_f1'])

    # Train the NER model for either pretraining or fine-tuning
    def train(self, task="ner_pretrain"):
        # Configure training settings based on whether prompts are used
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model.train()
        self.logger.info(f"***** Running NER {'pre-training' if task == 'ner_pretrain' else 'fine-tuning'} *****")
        self.logger.info("  Num instances = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epochs = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = %f", self.args.lr)
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        self.step = 0
        self.no_improve = 0
        # Initialize progress bar for training
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            avg_loss, loss_count = 0, 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch + 1, self.args.num_epochs))
                epoch_loss = 0.0
                given_labels, pred_labels, targets_unk = [], [], []

                for batch in self.train_data:
                    self.step += 1
                    # Move batch tensors to device
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    # Perform forward pass and compute loss
                    loss, ner_logits_batch, targets_batch, attention_mask, words, img_names = self._step(
                        batch, task, "train", epoch
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    batch_loss = loss.detach().cpu().item()
                    avg_loss += batch_loss
                    loss_count += 1
                    epoch_loss += batch_loss

                    # Update progress bar with average loss
                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss_display))
                        avg_loss, loss_count = 0, 0

                    # Select targets based on task
                    targets = targets_batch[1] if task == "ner_finetune" else targets_batch
                    given_labels_batch, pred_labels_batch = self._gen_labels(
                        ner_logits_batch, targets, attention_mask, return_indices=(task == "ner_finetune")
                    )
                    given_labels.extend(given_labels_batch)
                    pred_labels.extend(pred_labels_batch)
                    if task == "ner_finetune":
                        targets_unk_batch, _ = self._gen_labels(targets_batch[0], targets_batch[0], attention_mask, return_indices=True)
                        targets_unk.extend(targets_unk_batch)

                ner_f1_score = error_f1_score = 0.0
                # Compute metrics for pretraining (ner_f1) or fine-tuning (error_f1)
                if task == "ner_pretrain" and given_labels and pred_labels:
                    ner_report = classification_report(given_labels, pred_labels, digits=4, output_dict=True)
                    ner_f1_score = ner_report['micro avg']['f1-score']
                    self.logger.info("***** NER Pretrain results *****")
                    self.logger.info("\n%s", classification_report(given_labels, pred_labels, digits=4))
                    self.logger.info("Epoch {}/{}, NER train F1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, ner_f1_score))
                elif task == "ner_finetune" and given_labels and pred_labels and targets_unk:
                    # targets_unk = [batch[0] for batch in self.train_data]
                    error_precision, error_recall, error_f1_score = comp_f1_score(given_labels, targets_unk, pred_labels)
                    self.logger.info("***** NER Finetune results *****")
                    self.logger.info("Epoch {}/{}, Error Detection Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, error_precision, error_recall, error_f1_score))

                # Log metrics to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, task, "train", epoch_loss / len(self.train_data), ner_f1_score, error_f1_score])

                # Save checkpoint after each epoch
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Evaluate model if past eval_begin_epoch
                if epoch >= self.args.eval_begin_epoch:
                    metric = self.evaluate(task, epoch)
                    if metric < self.best_dev:
                        self.no_improve += 1
                    else:
                        self.no_improve = 0
                        self.best_dev = metric
                    if self.no_improve >= self.args.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            torch.cuda.empty_cache()
            pbar.close()

        # Save final model
        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    # Evaluate the model on validation set
    def evaluate(self, task="ner_pretrain", epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running evaluation for {task} *****")
        self.logger.info("  Num instances = %d", len(self.val_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, val_loss = self._eval_labels(pbar, self.val_data, epoch, task, "val")
                if task == "ner_pretrain":
                    ner_precision, ner_recall, ner_f1 = metrics
                    self.logger.info("Epoch {}/{}, NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                        epoch + 1, self.args.num_epochs, ner_precision, ner_recall, ner_f1))
                    metric_for_saving = ner_f1
                    error_f1 = 0.0
                else:
                    error_precision, error_recall, error_f1 = metrics
                    self.logger.info("Epoch {}/{}, Error Detection Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                        epoch + 1, self.args.num_epochs, error_precision, error_recall, error_f1))
                    metric_for_saving = error_f1
                    ner_f1 = 0.0

                self.logger.info("Best F1: {:<6.5f}, Best Epoch: {}".format(self.best_dev, self.best_dev_epoch))
                # Save best model if performance improves
                if metric_for_saving >= self.best_dev:
                    self.logger.info("Get better performance at epoch {}".format(epoch + 1))
                    self.best_dev_epoch = epoch + 1
                    self.best_dev = metric_for_saving
                    if self.args.save_path:
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.logger.info(f"Saved best model (F1: {metric_for_saving:.4f}) to {self.best_model_path}")
                # Log validation metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, task, "val", val_loss, ner_f1, error_f1])

        self.model.train()
        return metric_for_saving

    # Test the model on test set using the best model
    def test(self, task="ner_pretrain"):
        self.model.eval()
        self.logger.info(f"***** Running testing for {task} *****")
        self.logger.info("  Num instances = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        # Load best model if available
        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, test_loss = self._eval_labels(pbar, self.test_data, epoch=0, task=task, stage="test")
                if task == "ner_pretrain":
                    ner_precision, ner_recall, ner_f1 = metrics
                    self.logger.info("Test NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                        ner_precision, ner_recall, ner_f1))
                    error_f1 = 0.0
                else:
                    error_precision, error_recall, error_f1 = metrics
                    self.logger.info("Test Error Detection Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                        error_precision, error_recall, error_f1))
                    ner_f1 = 0.0
                # Log test metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0, task, "test", test_loss, ner_f1, error_f1])

        self.model.train()
        return ner_f1 if task == "ner_pretrain" else error_f1

    # Perform a single training step
    def _step(self, batch, task="ner_pretrain", stage="train", epoch=0):
        # Unpack batch based on task and prompt usage
        if self.args.use_prompt:
            if task == "ner_finetune":
                (targets_unk, targets_new, _, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
            else:
                (targets_unk, _, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = targets_unk
        else:
            if task == "ner_finetune":
                (targets_unk, targets_new, _, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
            else:
                (targets_unk, _, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = targets_unk
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))

        # Select images for hvpnet or mkgformer
        imgs, aux_imgs = self._select_images(hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs)

        # Forward pass through the model
        if self.args.ner_model_name == "hvpnet":
            loss, ner_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=targets_batch[1] if task == "ner_finetune" else targets_batch,
                images=imgs,
                aux_imgs=aux_imgs
            )
        elif self.args.ner_model_name == "mkgformer":
            loss, ner_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=targets_batch[1] if task == "ner_finetune" else targets_batch,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs
            )
        ner_logits = torch.tensor(ner_logits, device=self.args.device)

        return loss, ner_logits, targets_batch, attention_mask, words, img_names
    
    # Select images based on NER model type
    def _select_images(self, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs):
        if self.args.ner_model_name == "hvpnet":
            return hvp_imgs, hvp_aux_imgs
        elif self.args.ner_model_name == "mkgformer":
            return mkg_imgs, mkg_aux_imgs
        return None, None

    # Generate labels from logits for evaluation
    def _gen_labels(self, logits, targets, token_attention_mask, return_indices=False):
        # Convert tensors to numpy arrays
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)
        label_ids = targets.detach().cpu().numpy() if targets is not None else np.zeros_like(logits)
        token_attention_mask = token_attention_mask.detach().cpu().numpy()
        label_map = {idx: label for label, idx in self.label_map.items()}
        given_label_batch, pred_label_batch = [], []

        # Process each sequence in the batch
        for row in range(token_attention_mask.shape[0]):
            mask = token_attention_mask[row].astype(bool)
            label_row_masked = label_ids[row][mask]
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

    # Evaluate labels for validation or testing
    def _eval_labels(self, pbar, data, epoch, task="ner_pretrain", stage="val"):
        given_labels, pred_labels, targets_unk = [], [], []
        total_loss = 0.0
        batch_count = 0
        pbar.set_description_str("Validation" if stage == "val" else "Testing")
        
        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, ner_logits, targets_batch, attention_mask, _, _ = self._step(
                batch, task, stage, epoch
            )
            total_loss += loss.detach().cpu().item()
            batch_count += 1
            targets = targets_batch[1] if task == "ner_finetune" else targets_batch
            given_labels_batch, pred_labels_batch = self._gen_labels(
                ner_logits, targets, attention_mask, return_indices=(task == "ner_finetune")
            )
            given_labels.extend(given_labels_batch)
            pred_labels.extend(pred_labels_batch)
            if task == "ner_finetune":
                targets_unk_batch, _ = self._gen_labels(targets_batch[0], targets_batch[0], attention_mask, return_indices=True)
                targets_unk.extend(targets_unk_batch)
            pbar.update()
        pbar.close()

        # Compute metrics based on task
        if task == "ner_pretrain":
            ner_report = classification_report(given_labels, pred_labels, digits=4, output_dict=True) if given_labels else {'micro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
            ner_precision = ner_report['micro avg']['precision']
            ner_recall = ner_report['micro avg']['recall']
            ner_f1 = ner_report['micro avg']['f1-score']
            return (ner_precision, ner_recall, ner_f1), total_loss / batch_count
        else:
            error_precision = error_recall = error_f1 = 0.0
            if given_labels and pred_labels and targets_unk:
                error_precision, error_recall, error_f1 = comp_f1_score(given_labels, targets_unk, pred_labels)
            return (error_precision, error_recall, error_f1), total_loss / batch_count

    # Configure optimizer and scheduler for text-only training
    def training_settings_text_only(self):        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.1
        )
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)

    # Configure optimizer and scheduler for training with visual prompts
    def training_settings_with_prompt(self):
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name or 'text' in name:
                params['params'].append(param)
        parameters.append(params)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name or 'vision' in name:
                params['params'].append(param)
        parameters.append(params)
        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)
        for name, par in self.model.named_parameters():
            if 'image_model' in name:
                par.requires_grad = False
        self.optimizer = AdamW(parameters, weight_decay=0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        self.model.to(self.args.device)

# Trainer for diffusion tasks, supporting pretraining and fine-tuning
class PreTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        super().__init__(label_map, args, logger, metrics_file)
        # Initialize datasets, model, and training parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_num_steps = len(self.train_data) * args.num_epochs if train_data else 0
        self.best_dev = 0
        self.best_dev_epoch = None
        self.best_ner_f1 = 0.0
        self.best_diffusion_f1 = 0.0
        self.model = model
        self.optimizer = None
        self.scheduler = None
        # Define paths for saving best and final models
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_{'diffusion_pretrain' if args.do_diffusion_pretrain else 'diffusion_finetune'}_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_{'diffusion_pretrain' if args.do_diffusion_pretrain else 'diffusion_finetune'}_final.pth")
        # Initialize metrics CSV file with headers
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'task', 'stage', 'loss', 'ner_f1', 'diffusion_f1', 'error_f1'])

    # Train the diffusion model for either pretraining or fine-tuning
    def train(self, task="diffusion_pretrain", stage="train", epoch=0):
        # Configure training settings
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model.train()
        self.logger.info(f"***** Running {'diffusion pre-training' if task == 'diffusion_pretrain' else 'diffusion fine-tuning'} *****")
        self.logger.info(f"  Num instances = {len(self.train_data) * self.args.batch_size}")
        self.logger.info(f"  Num epochs = {self.args.num_epochs}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")
        self.logger.info(f"  Learning rate = {self.args.lr}")
        self.logger.info(f"  Evaluate begin = {self.args.eval_begin_epoch}")

        self.step = 0
        self.no_improve = 0
        # Initialize progress bar
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True) as pbar:
            avg_loss, loss_count = 0, 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                epoch_loss = 0.0
                if task == "diffusion_pretrain":
                    ner_labels, ner_preds, diffusion_labels, diffusion_preds = [], [], [], []
                else:
                    targets_old, error_labels, error_preds = [], [], []

                for batch in self.train_data:
                    self.step += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    # Perform forward pass
                    loss, ner_logits_batch, diffusion_logits_batch, targets_batch, attention_mask, words, img_names = self._step(
                        batch, task, stage, epoch
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    batch_loss = loss.detach().cpu().item()
                    avg_loss += batch_loss
                    loss_count += 1
                    epoch_loss += batch_loss

                    # Update progress bar
                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(f"loss: {avg_loss_display:<6.5f}")
                        avg_loss, loss_count = 0, 0

                    # Generate labels based on task
                    if task == "diffusion_pretrain":
                        targets = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
                        ner_labels_batch, ner_pred_batch = self._gen_labels(ner_logits_batch, targets, attention_mask, return_indices=False)
                        diffusion_labels_batch, diffusion_pred_batch = self._gen_labels(diffusion_logits_batch, targets, attention_mask, return_indices=False)
                        ner_labels.extend(ner_labels_batch)
                        ner_preds.extend(ner_pred_batch)
                        diffusion_labels.extend(diffusion_labels_batch)
                        diffusion_preds.extend(diffusion_pred_batch)
                    else:
                        old_labels_batch, _ = self._gen_labels(targets_batch[0], targets_batch[0], attention_mask, return_indices=True)
                        error_labels_batch, error_pred_batch = self._gen_labels(diffusion_logits_batch, targets_batch[1], attention_mask, return_indices=True)
                        targets_old.extend(old_labels_batch)
                        error_labels.extend(error_labels_batch)
                        error_preds.extend(error_pred_batch)

                ner_f1_score = diffusion_f1_score = error_f1_score = error_precision = error_recall = 0.0
                # Compute metrics for diffusion tasks
                if task == "diffusion_pretrain":
                    if ner_labels and ner_preds:
                        ner_report = classification_report(ner_labels, ner_preds, digits=4, output_dict=True)
                        ner_f1_score = ner_report['micro avg']['f1-score']
                        self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, NER Train F1: {ner_f1_score:.4f}")
                    if diffusion_labels and diffusion_preds:
                        diffusion_report = classification_report(diffusion_labels, diffusion_preds, digits=4, output_dict=True)
                        diffusion_f1_score = diffusion_report['micro avg']['f1-score']
                        self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Diffusion Train F1: {diffusion_f1_score:.4f}")
                else:
                    if targets_old and error_labels and error_preds:
                        error_precision, error_recall, error_f1_score = comp_f1_score(error_labels, targets_old, error_preds)
                        self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Error Detection Precision: {error_precision:.4f}, Recall: {error_recall:.4f}, F1: {error_f1_score:.4f}")

                # Log metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, task, "train", epoch_loss / len(self.train_data), ner_f1_score, diffusion_f1_score, error_f1_score])

                # Evaluate model
                if epoch >= self.args.eval_begin_epoch:
                    if self.evaluate(task, stage="val", epoch=epoch):
                        break

            torch.cuda.empty_cache()
            pbar.close()

        # Save final model
        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    # Evaluate the diffusion model
    def evaluate(self, task="diffusion_pretrain", stage="val", epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running {stage} evaluation for {task} *****")
        self.logger.info(f"  Num instances = {len(self.val_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, val_loss = self._eval_labels(pbar, self.val_data, epoch, task, stage)
                if task == "diffusion_pretrain":
                    (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1) = metrics
                    self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, NER Precision: {ner_precision:<6.5f}, Recall: {ner_recall:<6.5f}, F1: {ner_f1:<6.5f}")
                    self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Diffusion Precision: {diffusion_precision:<6.5f}, Recall: {diffusion_recall:<6.5f}, F1: {diffusion_f1:<6.5f}")
                    metric_for_saving = diffusion_f1
                    error_f1 = 0.0
                    if ner_f1 > self.best_ner_f1:
                        self.best_ner_f1 = ner_f1
                    if diffusion_f1 > self.best_diffusion_f1:
                        self.best_diffusion_f1 = diffusion_f1
                else:
                    (error_precision, error_recall, error_f1) = metrics
                    self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Error Detection Precision: {error_precision:<6.5f}, Recall: {error_recall:<6.5f}, F1: {error_f1:<6.5f}")
                    metric_for_saving = error_f1
                    ner_f1 = diffusion_f1 = 0.0

                save_model = False
                # Save best model if performance improves
                if metric_for_saving > self.best_dev:
                    self.best_dev = metric_for_saving
                    self.best_dev_epoch = epoch + 1
                    self.no_improve = 0
                    save_model = True
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.args.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        return True

                if save_model and self.args.save_path:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.logger.info(f"Saved best model (F1: {metric_for_saving:.4f}) to {self.best_model_path}")

                # Log metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, task, stage, val_loss, ner_f1, diffusion_f1, error_f1])

        self.model.train()
        return False

    # Test the diffusion model
    def test(self, task="diffusion_pretrain", stage="test", epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running {stage} testing for {task} *****")
        self.logger.info(f"  Num instances = {len(self.test_data) * self.args.batch_size}")
        self.logger.info(f"  Batch size = %d", self.args.batch_size)

        # Load best model
        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                metrics, test_loss = self._eval_labels(pbar, self.test_data, epoch, task, stage)
                if task == "diffusion_pretrain":
                    (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1) = metrics
                    self.logger.info(f"Test NER Precision: {ner_precision:<6.5f}, Recall: {ner_recall:<6.5f}, F1: {ner_f1:<6.5f}")
                    self.logger.info(f"Test Diffusion Precision: {diffusion_precision:<6.5f}, Recall: {diffusion_recall:<6.5f}, F1: {diffusion_f1:<6.5f}")
                    error_f1 = 0.0
                else:
                    (error_precision, error_recall, error_f1) = metrics
                    self.logger.info(f"Test Error Detection Precision: {error_precision:<6.5f}, Recall: {error_recall:<6.5f}, F1: {error_f1:<6.5f}")
                    ner_f1 = diffusion_f1 = 0.0

                # Log test metrics
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0, task, stage, test_loss, ner_f1, diffusion_f1, error_f1])

        self.model.train()
        return error_f1 if task == "diffusion_finetune" else (ner_f1, diffusion_f1)

    # Perform a single training step for diffusion tasks
    def _step(self, batch, task="diffusion_pretrain", stage="train", epoch=0):
        # Unpack batch based on task and prompt usage
        if self.args.use_prompt:
            if task == "diffusion_finetune":
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
            else:
                (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = targets_unk
        else:
            if task == "diffusion_finetune":
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
            else:
                (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
                targets_batch = targets_unk
                hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))

        # Select images
        imgs, aux_imgs = self._select_images(hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs)

        # Forward pass through the diffusion model
        if stage == "train":
            if task == "diffusion_pretrain":
                loss, recon_emissions, ner_logits = self.model(
                    labels=targets_unk,
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
                diffusion_logits = recon_emissions.argmax(dim=-1)
            elif task == "diffusion_finetune":
                loss, recon_emissions, ner_logits = self.model(
                    labels=targets_unk,
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
                diffusion_logits = recon_emissions.argmax(dim=-1)
        elif stage in ["val", "test"]:
            diffusion_logits = self.model.reverse_diffusion(
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                steps=getattr(self.args, 'reverse_steps', 20)
            )
            targets = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
            loss, recon_emissions, ner_logits = self.model(
                labels=targets,
                targets_new=targets_batch[1] if isinstance(targets_batch, tuple) else None,
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                mode="pretrain" if task == "diffusion_pretrain" else "finetune",
                epoch=epoch
            )
            if task == "diffusion_pretrain":
                diffusion_logits = recon_emissions.argmax(dim=-1)

        self.logger.debug(f"Epoch {epoch}, Task {task}, Stage {stage}, Total Loss: {loss:.4f}")
        return loss, ner_logits, diffusion_logits, targets_batch, attention_mask, words, img_names
    
    # Select images based on NER model type
    def _select_images(self, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs):
        if self.model.ner_model_name == "hvpnet":
            return hvp_imgs, hvp_aux_imgs
        elif self.model.ner_model_name == "mkgformer":
            return mkg_imgs, mkg_aux_imgs
        return None, None

    # Generate labels for diffusion tasks
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

    # Evaluate labels for diffusion tasks
    def _eval_labels(self, pbar, data, epoch, task="diffusion_pretrain", stage="val"):
        given_labels, ner_pred_labels, diffusion_pred_labels, targets_old, error_labels, error_preds = [], [], [], [], [], []
        total_loss = 0.0
        batch_count = 0
        pbar.set_description_str("Validation" if stage == "val" else "Testing")

        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, ner_logits, diffusion_logits, targets_batch, attention_mask, _, _ = self._step(
                batch, task, stage, epoch
            )
            total_loss += loss.detach().cpu().item() if loss is not None else 0.0
            batch_count += 1

            if task == "diffusion_pretrain":
                targets = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
                given_labels_batch, ner_pred_labels_batch = self._gen_labels(ner_logits, targets, attention_mask, return_indices=False)
                given_labels_batch, diffusion_pred_labels_batch = self._gen_labels(diffusion_logits, targets, attention_mask, return_indices=False)
                given_labels.extend(given_labels_batch)
                ner_pred_labels.extend(ner_pred_labels_batch)
                diffusion_pred_labels.extend(diffusion_pred_labels_batch)
            else:
                old_labels_batch, _ = self._gen_labels(targets_batch[0], targets_batch[0], attention_mask, return_indices=True)
                error_labels_batch, error_pred_batch = self._gen_labels(diffusion_logits, targets_batch[1], attention_mask, return_indices=True)
                targets_old.extend(old_labels_batch)
                error_labels.extend(error_labels_batch)
                error_preds.extend(error_pred_batch)

            pbar.update()
        pbar.close()

        if task == "diffusion_pretrain":
            ner_report = classification_report(given_labels, ner_pred_labels, digits=4, output_dict=True) if given_labels else {'micro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
            diffusion_report = classification_report(given_labels, diffusion_pred_labels, digits=4, output_dict=True) if given_labels else {'micro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
            ner_precision = ner_report['micro avg']['precision']
            ner_recall = ner_report['micro avg']['recall']
            ner_f1 = ner_report['micro avg']['f1-score']
            diffusion_precision = diffusion_report['micro avg']['precision']
            diffusion_recall = diffusion_report['micro avg']['recall']
            diffusion_f1 = diffusion_report['micro avg']['f1-score']
            return ((ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1)), total_loss / batch_count
        else:
            error_precision = error_recall = error_f1 = 0.0
            if targets_old and error_labels and error_preds:
                error_precision, error_recall, error_f1 = comp_f1_score(error_labels, targets_old, error_preds)
            return (error_precision, error_recall, error_f1), total_loss / batch_count

    # Configure optimizer and scheduler for text-only diffusion training
    def training_settings_text_only(self):
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

    # Configure optimizer and scheduler for diffusion training with prompts
    def training_settings_with_prompt(self):
        parameters = []
        params = {'lr': 5e-2, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc') or 'noise_pred' in name:
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