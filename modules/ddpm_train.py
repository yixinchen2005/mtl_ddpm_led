import torch
import os, csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from .metrics import comp_f1_score

class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, metrics_file=None) -> None:
        # Initialize base trainer with label map, arguments, logger, and metrics file
        self.label_map = label_map
        self.args = args
        self.logger = logger
        self.metrics_file = metrics_file
        self.refresh_step = 2  # Frequency for updating progress bar
        self.no_improve = 0    # Counter for early stopping
        self.step = 0          # Training step counter

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()

class NERTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        super().__init__(label_map, args, logger, metrics_file)
        # Data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        # Training Steps
        self.train_num_steps = len(self.train_data) * args.num_epochs
        # Metrics
        self.best_dev = 0
        self.best_dev_epoch = None
        # Model and Optimizer
        self.model = model
        self.optimizer = None
        self.scheduler = None
        # Model Paths
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_ner_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_ner_final.pth")
        self.checkpoint_dir = os.path.join(args.save_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Initialize CSV
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'mode', 'loss', 'ner_f1'])

    def train(self):
        # Setup training settings based on whether prompt is used
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model.train()
        self.logger.info("***** Running NER pre-training *****")
        self.logger.info("  Num instances = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epochs = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = %f", self.args.lr)
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        self.step = 0
        self.no_improve = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            avg_loss, loss_count = 0, 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch + 1, self.args.num_epochs))
                given_labels, ner_labels = [], []
                epoch_loss = 0.0

                for batch in self.train_data:
                    self.step += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    loss, ner_logits_batch, targets_batch, attention_mask, words, img_names = self._step(
                        batch, "train", epoch
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    # Track loss
                    batch_loss = loss.detach().cpu().item()
                    avg_loss += batch_loss
                    loss_count += 1
                    epoch_loss += batch_loss

                    # Log loss periodically
                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss_display))
                        avg_loss, loss_count = 0, 0

                    # Collect labels
                    given_labels_batch, ner_labels_batch = self._gen_labels(
                        ner_logits_batch, targets_batch, attention_mask
                    )
                    given_labels.extend(given_labels_batch)
                    ner_labels.extend(ner_labels_batch)

                # Compute F1 score
                ner_f1_score = 0.0
                if given_labels and ner_labels:
                    ner_report = classification_report(given_labels, ner_labels, digits=4, output_dict=True)
                    ner_f1_score = ner_report['micro avg']['f1-score']
                    self.logger.info("***** NER Train results *****")
                    self.logger.info("\n%s", classification_report(given_labels, ner_labels, digits=4))
                    self.logger.info("Epoch {}/{}, NER train F1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, ner_f1_score))
                else:
                    self.logger.info("***** NER Train results *****")
                    self.logger.info("No labels collected for epoch %d", epoch + 1)

                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, "train", epoch_loss / len(self.train_data), ner_f1_score])

                # Save checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Evaluate
                if epoch >= self.args.eval_begin_epoch:
                    ner_f1 = self.evaluate(epoch)
                    if ner_f1 < self.best_dev:
                        self.no_improve += 1
                    else:
                        self.no_improve = 0
                        self.best_dev = ner_f1
                    if self.no_improve >= self.args.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            torch.cuda.empty_cache()
            pbar.close()

        # Save final model
        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instances = %d", len(self.val_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                (ner_precision, ner_recall, ner_f1), val_loss = self._eval_labels(
                    pbar, self.val_data, epoch, "dev"
                )
                self.logger.info("Epoch {}/{}, NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    epoch + 1, self.args.num_epochs, ner_precision, ner_recall, ner_f1))
                self.logger.info("Best NER F1: {:<6.5f}, Best Epoch: {}".format(
                    self.best_dev, self.best_dev_epoch))
                if ner_f1 >= self.best_dev:
                    self.logger.info("Get better performance at epoch {}".format(epoch + 1))
                    self.best_dev_epoch = epoch + 1
                    self.best_dev = ner_f1
                    if self.args.save_path:
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.logger.info(f"Saved best model (NER F1: {ner_f1:.4f}) to {self.best_model_path}")
                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, "val", val_loss, ner_f1])

        self.model.train()
        return ner_f1

    def test(self):
        self.model.eval()
        self.logger.info("***** Running testing *****")
        self.logger.info("  Num instances = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        # Load best model
        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                (ner_precision, ner_recall, ner_f1), test_loss = self._eval_labels(
                    pbar, self.test_data, epoch=0, mode="test"
                )
                self.logger.info("Test NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    ner_precision, ner_recall, ner_f1))
                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0, "test", test_loss, ner_f1])

        self.model.train()
        return ner_f1

    def _step(self, batch, mode="train", epoch=0):
        # Extract data
        if self.args.use_prompt:
            (targets_unk, _, input_ids, token_type_ids, attention_mask,
             hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
        else:
            (targets_unk, _, input_ids, token_type_ids, attention_mask, words, img_names) = batch
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))

        # Select images based on model
        imgs, aux_imgs = self._select_images(hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs)

        # Forward pass
        if mode in ["train", "dev", "test"]:
            if self.args.ner_model_name == "hvpnet":
                loss, ner_logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=targets_unk,
                    images=imgs,
                    aux_imgs=aux_imgs
                )
            elif self.args.ner_model_name == "mkgformer":
                loss, ner_logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=targets_unk,
                    images=imgs,
                    aux_imgs=aux_imgs,
                    rcnn_imgs=rcnn_imgs
                )
            ner_logits = torch.tensor(ner_logits, device=self.args.device)
        else:
            raise ValueError("Invalid mode")

        return loss, ner_logits, targets_unk, attention_mask, words, img_names

    def _gen_labels(self, logits, targets, token_attention_mask):
        # Convert logits, targets, and attention mask to numpy for processing
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)
        label_ids = targets.detach().cpu().numpy()
        token_attention_mask = token_attention_mask.detach().cpu().numpy()
        label_map = {idx: label for label, idx in self.label_map.items()}
        given_label_batch, pred_label_batch = [], []

        # Process each sequence in the batch
        for row in range(token_attention_mask.shape[0]):
            mask = token_attention_mask[row].astype(bool)
            label_row_masked = label_ids[row][mask]
            pred_row = logits[row]
            given_label_sent, pred_label_sent = [], []
            for column in range(len(label_row_masked)):
                # Skip padding and special tokens
                if column == 0 or label_map.get(label_row_masked[column], '') in ["X", "[SEP]"]:
                    continue
                given_label_sent.append(label_map.get(label_row_masked[column], 'O'))
                pred_label_sent.append(label_map.get(pred_row[column], 'O'))
            given_label_batch.append(given_label_sent)
            pred_label_batch.append(pred_label_sent)
        
        return given_label_batch, pred_label_batch

    def _eval_labels(self, pbar, data, epoch, mode="dev"):
        # Initialize lists for labels and loss tracking
        given_labels, ner_pred_labels = [], []
        total_loss = 0.0
        batch_count = 0
        pbar.set_description_str(desc="Dev" if mode == "dev" else "Testing")
        
        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, ner_logits, targets_unk, attention_mask, words, img_names = self._step(
                batch, mode, epoch
            )
            total_loss += loss.detach().cpu().item()
            batch_count += 1
            given_labels_batch, ner_pred_labels_batch = self._gen_labels(
                ner_logits, targets_unk, attention_mask
            )
            given_labels.extend(given_labels_batch)
            ner_pred_labels.extend(ner_pred_labels_batch)
            pbar.update()
        pbar.close()

        # Compute NER metrics
        ner_report = classification_report(given_labels, ner_pred_labels, digits=4, output_dict=True)
        ner_precision = ner_report['micro avg']['precision']
        ner_recall = ner_report['micro avg']['recall']
        ner_f1 = ner_report['micro avg']['f1-score']
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return (ner_precision, ner_recall, ner_f1), avg_loss

    def training_settings_text_only(self):        
        # Initialize optimizer and scheduler for text-only training
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

    def training_settings_with_prompt(self):
        # Define parameters with different learning rates for prompt-based training
        parameters = []
        # bert lr
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name or 'text' in name:
                params['params'].append(param)
        parameters.append(params)

        # prompt lr / vit lr
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name or 'vision' in name:
                params['params'].append(param)
        parameters.append(params)

        # crf lr
        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        # Freeze image model
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

class PreTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        super().__init__(label_map, args, logger, metrics_file)
        # Data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        # Training Steps
        self.train_num_steps = len(self.train_data) * args.num_epochs
        self.eval_steps = 100
        # Metrics
        self.best_dev = 0
        self.best_dev_epoch = None
        # Data Structures for training
        self.model = model
        self.optimizer = None
        self.scheduler = None
        # History for moving average
        self.ner_f1_history = []
        self.diffusion_f1_history = []
        # Define model paths
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_{args.ner_model_name}_final.pth")
        self.checkpoint_dir = os.path.join(args.save_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Initialize CSV header
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'mode', 'loss', 'ner_f1', 'diffusion_f1', 'error_f1'])

    def train(self, mode="pretrain"):
        # Setup training settings
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model.train()
        self.logger.info(f"***** Running {'pre-training' if mode == 'pretrain' else 'fine-tuning'} *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = %s", str(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        self.step = 0
        self.no_improve = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            avg_loss, loss_count = 0, 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch + 1, self.args.num_epochs))
                given_labels, ner_labels, diffusion_labels = [], [], []
                error_labels, error_preds = [], []  # For error detection F1 in fine-tuning
                epoch_loss = 0.0

                for batch in self.train_data:
                    self.step += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    loss, ner_logits_batch, diffusion_logits_batch, targets_batch, attention_mask, words, img_names = self._step(
                        batch, mode, epoch
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    # Track loss
                    batch_loss = loss.detach().cpu().item()
                    avg_loss += batch_loss
                    loss_count += 1
                    epoch_loss += batch_loss

                    # Log loss periodically
                    if self.step % self.refresh_step == 0:
                        avg_loss_display = avg_loss / loss_count if loss_count > 0 else 0.0
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss_display))
                        avg_loss, loss_count = 0, 0

                    # Collect labels
                    targets = targets_batch[0] if mode in ["finetune_forward", "finetune_backward"] else targets_batch
                    given_labels_batch, ner_labels_batch = self._gen_labels(
                        ner_logits_batch, targets, attention_mask
                    )
                    given_labels_batch, diffusion_labels_batch = self._gen_labels(
                        diffusion_logits_batch, targets, attention_mask
                    )
                    given_labels.extend(given_labels_batch)
                    ner_labels.extend(ner_labels_batch)
                    diffusion_labels.extend(diffusion_labels_batch)

                    # Collect error detection labels for fine-tuning
                    if mode in ["finetune_forward", "finetune_backward"]:
                        _, error_pred_batch = self._gen_labels(diffusion_logits_batch, targets_batch[1], attention_mask)
                        error_labels.extend(given_labels_batch)
                        error_preds.extend(error_pred_batch)

                # Compute F1 scores and log to CSV
                ner_f1_score = diffusion_f1_score = error_f1_score = 0.0
                if given_labels and ner_labels:
                    ner_report = classification_report(given_labels, ner_labels, digits=4, output_dict=True)
                    ner_f1_score = ner_report['micro avg']['f1-score']
                    self.logger.info("***** MNER Train results *****")
                    self.logger.info("\n%s", classification_report(given_labels, ner_labels, digits=4))
                    self.logger.info("Epoch {}/{}, current MNER train f1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, ner_f1_score))
                else:
                    self.logger.info("***** MNER Train results *****")
                    self.logger.info("No labels collected for epoch %d", epoch + 1)

                if given_labels and diffusion_labels:
                    diffusion_report = classification_report(given_labels, diffusion_labels, digits=4, output_dict=True)
                    diffusion_f1_score = diffusion_report['micro avg']['f1-score']
                    self.logger.info("***** Diffusion Train results *****")
                    self.logger.info("\n%s", classification_report(given_labels, diffusion_labels, digits=4))
                    self.logger.info("Epoch {}/{}, current Diffusion train f1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, diffusion_f1_score))
                else:
                    self.logger.info("***** Diffusion Train results *****")
                    self.logger.info("No labels collected for epoch %d", epoch + 1)

                # Compute error detection F1 for fine-tuning
                if mode in ["finetune_forward", "finetune_backward"] and error_labels and error_preds:
                    error_f1_score = comp_f1_score(error_labels, error_preds)
                    self.logger.info("***** Error Detection Train results *****")
                    self.logger.info("Epoch {}/{}, Error Detection train F1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, error_f1_score))

                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, "train", epoch_loss / len(self.train_data), ner_f1_score, diffusion_f1_score, error_f1_score])

                # Save checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Evaluate if needed
                if epoch >= self.args.eval_begin_epoch:
                    if self.evaluate(epoch, mode):
                        break  # Early stopping

            torch.cuda.empty_cache()
            pbar.close()

        # Save final model
        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    def evaluate(self, epoch, mode="pretrain"):
        self.model.eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instance = %d", len(self.val_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1), error_f1, val_loss = self._eval_labels(
                    pbar, self.val_data, epoch, mode, "dev"
                )
                # Moving average for NER and Diffusion F1
                self.ner_f1_history.append(ner_f1)
                self.diffusion_f1_history.append(diffusion_f1)
                if len(self.ner_f1_history) > 3:
                    self.ner_f1_history.pop(0)
                    self.diffusion_f1_history.pop(0)
                avg_ner_f1 = sum(self.ner_f1_history) / len(self.ner_f1_history)
                avg_diffusion_f1 = sum(self.diffusion_f1_history) / len(self.diffusion_f1_history)
                
                self.logger.info("Epoch {}/{}, NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}, Avg F1: {:<6.5f}".format(
                    epoch + 1, self.args.num_epochs, ner_precision, ner_recall, ner_f1, avg_ner_f1))
                self.logger.info("Epoch {}/{}, Diffusion Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}, Avg F1: {:<6.5f}".format(
                    epoch + 1, self.args.num_epochs, diffusion_precision, diffusion_recall, diffusion_f1, avg_diffusion_f1))
                if mode in ["finetune_forward", "finetune_backward"]:
                    self.logger.info("Epoch {}/{}, Error Detection F1: {:<6.5f}".format(
                        epoch + 1, self.args.num_epochs, error_f1))
                # Update pseudo-label state
                self.model.update_pseudo_label_state(ner_f1)
                self.logger.info(f"Epoch {epoch + 1}, NER F1: {ner_f1:.4f}, Using Pseudo-Labels: {self.model.use_pseudo_labels}")
                # Early stopping based on diffusion F1 (pretrain) or error F1 (finetune), with relaxed NER F1 threshold
                save_model = False
                if mode == "pretrain":
                    if diffusion_f1 >= self.best_dev and avg_ner_f1 >= 0.70:
                        self.best_dev = diffusion_f1
                        self.best_dev_epoch = epoch + 1
                        self.no_improve = 0
                        save_model = True
                else:  # finetune_forward/backward
                    if error_f1 >= self.best_dev and avg_ner_f1 >= 0.70:
                        self.best_dev = error_f1
                        self.best_dev_epoch = epoch + 1
                        self.no_improve = 0
                        save_model = True

                if save_model and self.args.save_path:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.logger.info(f"Saved best model (Diffusion F1: {diffusion_f1:.4f}, Error F1: {error_f1:.4f}, NER F1: {ner_f1:.4f}) to {self.best_model_path}")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.args.patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        return True  # Signal early stopping
                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, "val", val_loss, ner_f1, diffusion_f1, error_f1])

        self.model.train()
        return False
    
    def test(self, mode="pretrain"):
        self.model.eval()
        self.logger.info("***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        # Load best model
        if os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info("Load model successful!")
        else:
            self.logger.warning(f"Best model not found at {self.best_model_path}. Using current model.")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1), error_f1, test_loss = self._eval_labels(
                    pbar, self.test_data, epoch=0, mode=mode, phase="test"
                )
                self.logger.info("Test NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    ner_precision, ner_recall, ner_f1))
                self.logger.info("Test Diffusion Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    diffusion_precision, diffusion_recall, diffusion_f1))
                if mode in ["finetune_forward", "finetune_backward"]:
                    self.logger.info("Test Error Detection F1: {:<6.5f}".format(error_f1))
                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0, "test", test_loss, ner_f1, diffusion_f1, error_f1])

        self.model.train()
        return diffusion_f1 if mode == "pretrain" else error_f1
    
    def _step(self, batch, mode="pretrain", epoch=0):
        # Extract data
        if self.args.use_prompt:
            if mode in ["finetune_forward", "finetune_backward"]:
                # For fine-tuning, include both targets_unk and targets_new
                (targets_unk, targets_new, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = (targets_unk, targets_new)
            else:
                # For pretraining, only targets_unk
                (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask,
                 hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
                targets_batch = targets_unk
        else:
            if mode in ["finetune_forward", "finetune_backward"]:
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

        # Forward pass
        if mode == "pretrain":
            # Pretraining mode: Compute loss and logits for NER and diffusion
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
        elif mode in ["finetune_forward", "finetune_backward"]:
            # Fine-tuning mode: Include targets_new for error detection
            loss, recon_emissions, ner_logits = self.model(
                labels=targets_unk,
                targets_new=targets_new,
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                mode=mode,
                epoch=epoch
            )
            diffusion_logits = recon_emissions.argmax(dim=-1)
        elif mode in ["dev", "test"]:
            # Validation/test mode: Compute NER logits and diffusion logits via reverse diffusion
            _, _, ner_logits = self.model(
                labels=None,  # Prevent leakage
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
            diffusion_logits = self.model.reverse_diffusion(
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                steps=getattr(self.args, 'reverse_steps', 50)  # Configurable steps
            )
            # Compute validation loss
            targets = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
            loss, recon_emissions, _ = self.model(
                labels=targets,
                targets_new=targets_batch[1] if isinstance(targets_batch, tuple) else None,
                char_input_ids=char_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=imgs,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                mode=mode if isinstance(targets_batch, tuple) else "pretrain",
                epoch=epoch
            )
        else:
            raise ValueError("Invalid mode")

        # Log loss components
        self.logger.info(f"Epoch {epoch}, Mode {mode}, Total Loss: {loss:.4f}, "
                        f"NER Loss: {getattr(self.model, 'ner_loss', 0.0):.4f}, "
                        f"MSE Loss: {getattr(self.model, 'mse_loss', 0.0):.4f}, "
                        f"Denoise CRF Loss: {getattr(self.model, 'denoise_crf_loss', 0.0):.4f}, "
                        f"KL Loss: {getattr(self.model, 'kl_loss', 0.0):.4f}")
        
        return loss, ner_logits, diffusion_logits, targets_batch, attention_mask, words, img_names
    
    def _select_images(self, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs):
        # Select appropriate images based on model type
        if self.model.ner_model_name == "hvpnet":
            return hvp_imgs, hvp_aux_imgs
        elif self.model.ner_model_name == "mkgformer":
            return mkg_imgs, mkg_aux_imgs
        return None, None

    def _gen_labels(self, logits, targets, token_attention_mask):
        # Debug input types
        self.logger.debug(f"_gen_labels inputs: logits type={type(logits)}, targets type={type(targets)}, "
                         f"token_attention_mask type={type(token_attention_mask)}")
        
        # Convert inputs to numpy arrays
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)
        else:
            raise ValueError(f"Unsupported logits type: {type(logits)}")

        if targets is None:
            # In dev/predict mode, targets may be None
            label_ids = np.zeros_like(logits)  # Dummy labels
        elif isinstance(targets, torch.Tensor):
            label_ids = targets.detach().cpu().numpy()
        elif isinstance(targets, list):
            label_ids = np.array(targets)
        else:
            raise ValueError(f"Unsupported targets type: {type(targets)}")

        if isinstance(token_attention_mask, torch.Tensor):
            token_attention_mask = token_attention_mask.detach().cpu().numpy()
        elif isinstance(token_attention_mask, list):
            token_attention_mask = np.array(token_attention_mask)
        else:
            raise ValueError(f"Unsupported token_attention_mask type: {type(token_attention_mask)}")

        label_map = {idx: label for label, idx in self.label_map.items()}
        given_label_batch, pred_label_batch = [], []
        
        # Process each sequence in the batch
        for row in range(token_attention_mask.shape[0]):
            mask = token_attention_mask[row].astype(bool)
            label_row_masked = label_ids[row][mask] if targets is not None else np.zeros(sum(mask), dtype=np.int64)
            pred_row = logits[row][mask]
            
            given_label_sent, pred_label_sent = [], []
            
            for column in range(len(label_row_masked)):
                # Skip padding and special tokens
                if column == 0 or (label_map.get(label_row_masked[column], '') in ["X", "[SEP]"]):
                    continue
                given_label_sent.append(label_map.get(label_row_masked[column], 'O'))
                pred_label_sent.append(label_map.get(pred_row[column], 'O'))
            
            given_label_batch.append(given_label_sent)
            pred_label_batch.append(pred_label_sent)
        
        return given_label_batch, pred_label_batch

    def _eval_labels(self, pbar, data, epoch, mode="pretrain", phase="dev"):
        # Initialize lists for labels and loss tracking
        given_labels, ner_pred_labels, diffusion_pred_labels, error_labels, error_preds = [], [], [], [], []
        total_loss = 0.0
        batch_count = 0
        pbar.set_description_str(desc="Dev" if phase == "dev" else "Testing")
        
        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, ner_logits, diffusion_logits, targets_batch, attention_mask, words, img_names = self._step(
                batch, phase, epoch
            )
            total_loss += loss.detach().cpu().item() if loss is not None else 0.0
            batch_count += 1
            # Select appropriate targets based on mode
            targets = targets_batch[0] if isinstance(targets_batch, tuple) else targets_batch
            given_labels_batch, ner_pred_labels_batch = self._gen_labels(
                ner_logits, targets, attention_mask
            )
            given_labels_batch, diffusion_pred_labels_batch = self._gen_labels(
                diffusion_logits, targets, attention_mask
            )
            given_labels.extend(given_labels_batch)
            ner_pred_labels.extend(ner_pred_labels_batch)
            diffusion_pred_labels.extend(diffusion_pred_labels_batch)
            # Collect error detection labels for fine-tuning
            if mode in ["finetune_forward", "finetune_backward"] and isinstance(targets_batch, tuple):
                _, error_pred_batch = self._gen_labels(diffusion_logits, targets_batch[1], attention_mask)
                error_labels.extend(given_labels_batch)
                error_preds.extend(error_pred_batch)
            pbar.update()
        pbar.close()

        # Compute NER metrics
        ner_report = classification_report(given_labels, ner_pred_labels, digits=4, output_dict=True) if given_labels and ner_pred_labels else {'micro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
        ner_precision = ner_report['micro avg']['precision']
        ner_recall = ner_report['micro avg']['recall']
        ner_f1 = ner_report['micro avg']['f1-score']
        
        # Compute diffusion metrics
        diffusion_report = classification_report(given_labels, diffusion_pred_labels, digits=4, output_dict=True) if given_labels and diffusion_pred_labels else {'micro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
        diffusion_precision = diffusion_report['micro avg']['precision']
        diffusion_recall = diffusion_report['micro avg']['recall']
        diffusion_f1 = diffusion_report['micro avg']['f1-score']

        # Compute error detection F1 for fine-tuning
        error_f1 = comp_f1_score(error_labels, error_preds) if error_labels and error_preds else 0.0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1), error_f1, avg_loss
    
    def training_settings_text_only(self):
        # Freeze char_lstm
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name:
                param.requires_grad = False
        
        # Initialize optimizer with uniform LR
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=1e-2
        )
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        
        # Move model to device
        self.model.to(self.args.device)

    def training_settings_with_prompt(self):
        # Get parameters with differential LRs
        parameters = self._get_model_parameters()
        
        # Initialize optimizer
        self.optimizer = AdamW(parameters, weight_decay=1e-2)
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        
        # Move model to device
        self.model.to(self.args.device)

    def _get_model_parameters(self):
        # Define parameters with different learning rates
        parameters = []
        
        # CRF/FC/noise_pred layers: high LR for fast adaptation
        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc') or 'noise_pred' in name:
                params['params'].append(param)
        parameters.append(params)
        
        # Other layers (ner_model, vt_encoder, time_mlp, norm, attention): standard LR
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('ner_model' in name and 'image_model' not in name and 'crf' not in name and not name.startswith('fc')) or \
               'vt_encoder' in name or 'time_mlp' in name or 'norm_' in name or '_attn' in name:
                params['params'].append(param)
        parameters.append(params)
        
        # Freeze char_lstm and image_model
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name or 'ner_model.image_model' in name:
                param.requires_grad = False
        
        return parameters