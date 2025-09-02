import torch
import os
import csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report

class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, metrics_file=None):
        """Initialize base trainer with label_map, arguments, logger, and metrics file."""
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

class PreTrainer(BaseTrainer):
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
                writer.writerow(['epoch', 'stage', 'batch', 'ner_f1', 'mse_loss', 'ce_loss'])

    def training_settings_init(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape, param.requires_grad)
        """Configure optimizer and scheduler for NER pre-training."""
        parameters = []
        # Text parameters in the vt encoder
        params = {'lr': 3e-5, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'vt_encoder.bert' in name.lower() or 'vt_encoder.text' in name.lower():
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # Vision parameters in the vt encoder
        params = {'lr': 3e-5, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if ('vt_encoder.vision' in name.lower() or 'vt_encoder.encoder_conv' in name.lower() or 
                'vt_encoder.gates' in name.lower()):
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # Attention layers
        params = {'lr': 1e-3, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'label_self_attn' in name.lower() or 'label_vt_attn' in name.lower():
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # Normalization layers
        params = {'lr': 3e-5, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if '_norm' in name.lower():
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # FiLM layers
        params = {'lr': 3e-5, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if '_film' in name.lower():
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # MLP, projection, and embedding layers
        params = {'lr': 1e-3, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if ('time_embed' in name.lower() or 'vt_proj' in name.lower() or 
                'label_proj' in name.lower() or 'embedding_pred' in name.lower() or 
                'classifier' in name.lower()):
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # LabelEncoder parameters
        params = {'lr': 1e-4, 'weight_decay': 5e-3, 'params': []}
        for name, param in self.model.named_parameters():
            if name.lower().startswith('label_encoder.') and 'label_proj' not in name.lower():
                params['params'].append(param)
        if params['params']:
            parameters.append(params)

        # Freeze image_model for hvpnet
        for name, param in self.model.named_parameters():
            if self.args.ner_model_name == 'hvpnet' and 'vt_encoder.image_model' in name.lower():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if name.lower().startswith('label_encoder'):
                param.requires_grad = False

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

    def train(self, task="ner_pretrain", stage="train", epoch=0):
        """Train the diffusion model for NER pre-training."""
        self.training_settings_init()
        self.model.train()
        self.logger.info(f"***** Running NER {task} *****")
        self.logger.info(f"  Num instances = {len(self.train_data) * self.args.batch_size}")
        self.logger.info(f"  Num epochs = {self.args.num_epochs}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")
        self.logger.info(f"  Gradient accumulation steps = {self.args.grad_accum_steps}")
        self.logger.info(f"  Evaluate begin = {self.args.eval_begin_epoch}")

        self.step = 0
        self.no_improve = 0
        t_threshold = 0.1 * self.args.train_steps
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True) as pbar:
            avg_loss = 0
            loss_count = 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                epoch_loss = 0.0
                epoch_mse_loss = 0.0
                epoch_ce_loss = 0.0
                batch_count = 0
                all_true_labels, all_pred_labels = [], []
                self._batch_idx = 0
                self.optimizer.zero_grad()

                for batch in self.train_data:
                    self.step += 1
                    self._batch_idx += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    loss, true_labels, pred_labels, attention_mask, t_random = self._step(
                        batch, task, stage, epoch
                    )
                    if loss is not None:
                        loss = loss / self.args.grad_accum_steps
                        # loss.backward()
                        # if self.step % self.args.grad_accum_steps == 0:
                        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        #     self.optimizer.step()
                        #     self.scheduler.step()
                        #     self.optimizer.zero_grad()

                        # batch_loss = loss.detach().cpu().item() * self.args.grad_accum_steps
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        batch_loss = loss.detach().cpu().item()

                        batch_mse_loss = self.model.mse_loss.item() if self.model.mse_loss is not None else 0.0
                        batch_ce_loss = self.model.ce_loss.item() if self.model.ce_loss is not None else 0.0
                        epoch_loss += batch_loss
                        epoch_mse_loss += batch_mse_loss
                        epoch_ce_loss += batch_ce_loss
                        batch_count += 1
                        avg_loss += batch_loss
                        loss_count += 1
                        if pred_labels is not None and t_random is not None and (t_random < t_threshold).any():
                            valid_mask = (t_random < t_threshold).view(-1, 1) & attention_mask.bool()
                            true_labels_batch, pred_labels_batch = self._gen_labels(
                                pred_labels[valid_mask[:, 0]], 
                                true_labels[valid_mask[:, 0]], 
                                attention_mask[valid_mask[:, 0]]
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
                    self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, Loss: {epoch_loss/batch_count:.4f}, MSE Loss: {epoch_mse_loss/batch_count:.4f}, CE Loss: {epoch_ce_loss/batch_count:.4f}, NER Micro F1: {micro_f1:.4f}")

                    if self.metrics_file:
                        with open(self.metrics_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                epoch + 1,
                                "train",
                                batch_count,
                                micro_f1,
                                epoch_mse_loss/batch_count if batch_count > 0 else 0.0,
                                epoch_ce_loss/batch_count if batch_count > 0 else 0.0
                            ])

                if epoch >= self.args.eval_begin_epoch:
                    if self.evaluate(task, stage="val", epoch=epoch):
                        break

            torch.cuda.empty_cache()
            pbar.close()

        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")

    def evaluate(self, task="ner_pretrain", stage="val", epoch=0):
        """Evaluate the diffusion model on NER task."""
        self.model.eval()
        self.logger.info(f"***** Running {stage} evaluation for NER {task} *****")
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
                micro_f1 = self._eval_labels(pbar, self.val_data, epoch, task, stage)
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

    def test(self, task="ner_pretrain", stage="test", epoch=0):
        """Test the diffusion model on NER task."""
        self.model.eval()
        self.logger.info(f"***** Running {stage} testing for NER {task} *****")
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
                micro_f1 = self._eval_labels(pbar, self.test_data, epoch, task, stage)
                self.logger.info(f"Test NER Micro F1: {micro_f1:.4f}")

        self.model.train()
        return micro_f1

    def _step(self, batch, task="ner_pretrain", stage="train", epoch=0):
        """Perform a single training or evaluation step."""
        if not hasattr(self, '_batch_idx'):
            self._batch_idx = 0
        self._batch_idx += 1

        expected_len = 10
        if len(batch) != expected_len:
            self.logger.error(f"Expected {expected_len} batch elements for task={task}, got {len(batch)}")
            raise ValueError(f"Expected {expected_len} batch elements, got {len(batch)}")

        labels, input_ids, token_type_ids, attention_mask, hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words = batch
        words = list(map(list, zip(*words)))
        images, aux_imgs = self._select_images(hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs)

        if stage == "train":
            loss, logits = self.model(
                labels=labels,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=images,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs
            )
            pred_labels = torch.argmax(logits, dim=-1) if logits is not None else None
            t_random = getattr(self.model, 't_random', None)
            return loss, labels, pred_labels, attention_mask, t_random
        else:
            pred_labels = self.model.reverse_diffusion(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=images,
                aux_imgs=aux_imgs,
                rcnn_imgs=rcnn_imgs,
                steps=getattr(self.args, 'reverse_steps', 100)
            )
            loss = None
            return loss, labels, pred_labels, attention_mask, None

    def _eval_labels(self, pbar, data, epoch, task="ner_pretrain", stage="val"):
        """Evaluate NER labels for validation or test set."""
        self._batch_idx = 0
        all_true_labels, all_pred_labels = [], []
        batch_count = 0
        pbar.set_description_str("Validation" if stage == "val" else "Testing")

        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, true_labels, pred_labels, attention_mask, _ = self._step(
                batch, task, stage, epoch
            )
            self.logger.info(f"Batch {self._batch_idx}: pred_labels shape={pred_labels.shape}, true_labels shape={true_labels.shape}")

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