import torch
import os, csv
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report

class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, metrics_file=None) -> None:
        self.label_map = label_map
        self.args = args
        self.logger = logger
        self.metrics_file = metrics_file
        self.refresh_step = 2
        self.step = 0

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()

class PreTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, model=None, label_map=None, args=None, logger=None, metrics_file=None):
        super().__init__(label_map, args, logger, metrics_file)
        # Data #
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        # Training Steps #
        self.train_num_steps = len(self.train_data) * args.num_epochs
        self.eval_steps = 100
        # Metrics #
        self.best_dev = 0
        self.best_dev_epoch = None
        # Data Structures for training #
        self.model = model
        self.optimizer = None
        self.scheduler = None
        # Define model paths
        self.best_model_path = os.path.join(args.save_path, f"{args.dataset_name}_hvpnet_best.pth")
        self.final_model_path = os.path.join(args.save_path, f"{args.dataset_name}_hvpnet_final.pth")
        self.checkpoint_dir = os.path.join(args.save_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Initialize CSV header
        if self.metrics_file:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'mode', 'loss', 'ner_f1', 'diffusion_f1'])

    def train(self):
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        self.step = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            avg_loss, loss_count = 0, 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch + 1, self.args.num_epochs))
                given_labels, ner_labels, diffusion_labels = [], [], []
                epoch_loss = 0.0

                for batch in self.train_data:
                    self.step += 1
                    batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
                    self.optimizer.zero_grad()
                    loss, ner_logits_batch, diffusion_logits_batch, targets_batch, attention_mask, words, img_names = self._step(
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
                    given_labels_batch, diffusion_labels_batch = self._gen_labels(
                        diffusion_logits_batch, targets_batch, attention_mask
                    )
                    given_labels.extend(given_labels_batch)
                    ner_labels.extend(ner_labels_batch)
                    diffusion_labels.extend(diffusion_labels_batch)

                # Compute F1 scores and log to CSV
                ner_f1_score = 0.0
                diffusion_f1_score = 0.0
                if given_labels and ner_labels:
                    ner_report = classification_report(given_labels, ner_labels, digits=4, output_dict=True)
                    ner_f1_score = ner_report['macro avg']['f1-score']
                    self.logger.info("***** MNER Train results *****")
                    self.logger.info("\n%s", classification_report(given_labels, ner_labels, digits=4))
                    self.logger.info("Epoch {}/{}, current MNER train f1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, ner_f1_score))
                else:
                    self.logger.info("***** MNER Train results *****")
                    self.logger.info("No labels collected for epoch %d", epoch + 1)

                if given_labels and diffusion_labels:
                    diffusion_report = classification_report(given_labels, diffusion_labels, digits=4, output_dict=True)
                    diffusion_f1_score = diffusion_report['macro avg']['f1-score']
                    self.logger.info("***** Diffusion Train results *****")
                    self.logger.info("\n%s", classification_report(given_labels, diffusion_labels, digits=4))
                    self.logger.info("Epoch {}/{}, current Diffusion train f1 score: {:.4f}".format(
                        epoch + 1, self.args.num_epochs, diffusion_f1_score))
                else:
                    self.logger.info("***** Diffusion Train results *****")
                    self.logger.info("No labels collected for epoch %d", epoch + 1)

                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, "train", epoch_loss / len(self.train_data), ner_f1_score, diffusion_f1_score])

                # Save checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Evaluate if needed
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)

            torch.cuda.empty_cache()
            pbar.close()

        # Save final model
        if self.args.save_path:
            torch.save(self.model.state_dict(), self.final_model_path)
            self.logger.info(f"Saved final model to {self.final_model_path}")    
    
    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instance = %d", len(self.val_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1), val_loss = self._eval_labels(
                    pbar, self.val_data, epoch, "dev"
                )
                self.logger.info("Epoch {}/{}, NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    epoch + 1, self.args.num_epochs, ner_precision, ner_recall, ner_f1))
                self.logger.info("Epoch {}/{}, Diffusion Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}, Best F1: {:<6.5f}, Best Epoch: {}".format(
                    epoch + 1, self.args.num_epochs, diffusion_precision, diffusion_recall, diffusion_f1, 
                    self.best_dev, self.best_dev_epoch))
                if diffusion_f1 >= self.best_dev:
                    self.logger.info("Get better performance at epoch {}".format(epoch + 1))
                    self.best_dev_epoch = epoch + 1
                    self.best_dev = diffusion_f1
                    if self.args.save_path:
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.logger.info(f"Saved best model (F1: {diffusion_f1:.4f}) to {self.best_model_path}")
                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, "val", val_loss, ner_f1, diffusion_f1])

        self.model.train()
        return ner_f1, diffusion_f1
    
    def test(self):
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
                (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1), test_loss = self._eval_labels(
                    pbar, self.test_data, epoch=0, mode="test"
                )
                self.logger.info("Test NER Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    ner_precision, ner_recall, ner_f1))
                self.logger.info("Test Diffusion Precision: {:<6.5f}, Recall: {:<6.5f}, F1: {:<6.5f}".format(
                    diffusion_precision, diffusion_recall, diffusion_f1))
                # Log to CSV
                if self.metrics_file:
                    with open(self.metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0, "test", test_loss, ner_f1, diffusion_f1])

        self.model.train()
        return diffusion_f1
    
    def _step(self, batch, mode="train", epoch=0):
        # Extract data from a batch #
        if self.args.use_prompt:
            (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask,
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names) = batch
        else:
            (targets_unk, char_input_ids, input_ids, token_type_ids, attention_mask, words, img_names) = batch
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))
        # Specify which images and auxiliary images are feeding into the networks #
        if self.model.ner_model_name == "hvpnet":
            imgs = hvp_imgs
            aux_imgs = hvp_aux_imgs
        elif self.model.ner_model_name == "mkgformer":
            imgs = mkg_imgs
            aux_imgs = mkg_aux_imgs
        else:
            imgs, aux_imgs = None, None

        if mode == "train":
            loss, recon_emissions, crf_logits = self.model(
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
        elif mode in ["dev", "test", "predict"]:
            loss, recon_emissions, crf_logits = self.model(
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
                steps=self.eval_steps
            )
        else:
            raise ValueError("Invalid mode")
    
        return loss, crf_logits, diffusion_logits, targets_unk, attention_mask, words, img_names
    
    def _gen_labels(self, logits, targets, token_attention_mask, words=None, img_names=None):
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
        words_batch, img_names_batch = [], []
        
        for row in range(token_attention_mask.shape[0]):
            mask = token_attention_mask[row].astype(bool)
            label_row_masked = label_ids[row][mask] if targets is not None else np.zeros(sum(mask), dtype=np.int64)
            pred_row = logits[row][mask]
            
            given_label_sent, pred_label_sent = [], []
            word_sent = words[row] if words and words[row] else []
            img_name = img_names[row] if img_names and img_names[row] else None
            
            for column in range(len(label_row_masked)):
                # Skip padding and special tokens
                if column == 0 or (label_map.get(label_row_masked[column], '') in ["X", "[SEP]"]):
                    continue
                given_label_sent.append(label_map.get(label_row_masked[column], 'O'))
                pred_label_sent.append(label_map.get(pred_row[column], 'O'))
            
            given_label_batch.append(given_label_sent)
            pred_label_batch.append(pred_label_sent)
            words_batch.append(word_sent)
            img_names_batch.append(img_name)
        
        return given_label_batch, pred_label_batch

    def _eval_labels(self, pbar, data, epoch, mode="dev"):
        given_labels, ner_pred_labels, diffusion_pred_labels = [], [], []
        total_loss = 0.0
        batch_count = 0
        pbar.set_description_str(desc="Dev" if mode == "dev" else "Testing")
        
        for batch in data:
            batch = [tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch]
            loss, crf_logits, diffusion_logits, targets_unk, attention_mask, words, img_names = self._step(
                batch, mode, epoch
            )
            total_loss += loss.detach().cpu().item() if loss is not None else 0.0
            batch_count += 1
            given_labels_batch, ner_pred_labels_batch = self._gen_labels(
                crf_logits, targets_unk, attention_mask, words, img_names
            )
            given_labels_batch, diffusion_pred_labels_batch = self._gen_labels(
                diffusion_logits, targets_unk, attention_mask, words, img_names
            )
            given_labels.extend(given_labels_batch)
            ner_pred_labels.extend(ner_pred_labels_batch)
            diffusion_pred_labels.extend(diffusion_pred_labels_batch)
            pbar.update()
        pbar.close()

        ner_report = classification_report(given_labels, ner_pred_labels, digits=4, output_dict=True) if given_labels and ner_pred_labels else {'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
        ner_precision = ner_report['macro avg']['precision']
        ner_recall = ner_report['macro avg']['recall']
        ner_f1 = ner_report['macro avg']['f1-score']
        
        diffusion_report = classification_report(given_labels, diffusion_pred_labels, digits=4, output_dict=True) if given_labels and diffusion_pred_labels else {'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}}
        diffusion_precision = diffusion_report['macro avg']['precision']
        diffusion_recall = diffusion_report['macro avg']['recall']
        diffusion_f1 = diffusion_report['macro avg']['f1-score']

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return (ner_precision, ner_recall, ner_f1), (diffusion_precision, diffusion_recall, diffusion_f1), avg_loss
    
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
        self.optimizer = AdamW(parameters)
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
            num_training_steps=self.train_num_steps
        )
        
        # Move model to device
        self.model.to(self.args.device)

    def _get_model_parameters(self):
        parameters = []
        
        # CRF/FC/MLP/LayerNorm
        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('crf' in name or name.startswith('fc') or 'noise_pred' in name or 
                'time_mlp' in name or 'norm_' in name):
                params['params'].append(param)
        parameters.append(params)
        
        # ner_model (HVPNet: bert, encoder_conv, gates; MKGFormer: vision)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('ner_model' in name and 'image_model' not in name and 
                'crf' not in name and not name.startswith('fc')):
                params['params'].append(param)
        parameters.append(params)
        
        # Vision (vt_encoder)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'vt_encoder' in name:
                params['params'].append(param)
        parameters.append(params)
        
        # Attention
        params = {'lr': 5e-6, 'weight_decay': 5e-8}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if '_attn' in name:
                params['params'].append(param)
        parameters.append(params)
        
        # Freeze char_lstm and image_model
        for name, param in self.model.named_parameters():
            if 'char_lstm' in name or 'ner_model.image_model' in name:
                param.requires_grad = False
        
        return parameters