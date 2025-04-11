import torch
from torch import optim
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from torch.nn.functional import softmax
from itertools import cycle
from .pretrain import BaseTrainer
from models.mtl_ddpm_model import MeanTeacher
from utils.loss import CrossEntropyLossWrap, ContrastiveLoss, ConsistencyLoss
from .metrics import comp_f1_score

class DDPMTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, unlabeled_data=None, forward_net=None, reverse_net=None, label_map=None, args=None, logger=None, writer=None):
        super().__init__(label_map, args, logger, writer)
        # Data #
        self.train_data_labeled = train_data
        self.val_data_labeled = val_data
        self.test_data_labeled = test_data
        self.unlabeled_data = unlabeled_data
        # Training Steps #
        self.train_num_steps = len(self.train_data_labeled) * args.num_epochs
        self.epoch_start_unlabeled = 9
        self.semi_train_num_steps = len(self.train_data_labeled) * self.epoch_start_unlabeled + len(self.unlabeled_data) * (args.num_epochs - self.epoch_start_unlabeled)
        # Metrics #
        self.best_dev_forward = 0
        self.best_dev_reverse = 0
        self.best_dev_epoch_forward = None
        self.best_dev_epoch_reverse = None
        # Data Structures for training #
        self.model_dict = {"forward": forward_net, "reverse": reverse_net}
        self.optimizer = None
        self.scheduler = None
        # Loss Functions #
        self.num_labels = len(list(self.label_map.keys()))
        self.crit_id = CrossEntropyLossWrap(self.num_labels, option="standard")
        self.crit_edit = CrossEntropyLossWrap(self.num_labels, option="standard")
        self.crit_cycle = CrossEntropyLossWrap(self.num_labels, option="standard")
        self.crit_id_per_sample = CrossEntropyLossWrap(self.num_labels, option="per_sample")
        self.crit_id_weighted = CrossEntropyLossWrap(self.num_labels, option="weighted")
        self.crit_contrast = ContrastiveLoss()
        self.crit_consistentcy = ConsistencyLoss()

    def train(self):
        if self.args.use_prompt:
            self.training_settings_with_prompt()
        else:
            self.training_settings_text_only()

        self.model_dict["forward"].train()
        self.model_dict["reverse"].train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data_labeled) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        self.step = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            avg_loss = 0
            for epoch in range(self.args.num_epochs):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch+1, self.args.num_epochs))
                true_labels_new, true_labels_old, gen_labels_new, gen_labels_old = [], [], [], []
                train_data = self.train_data_labeled
                for batch in train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    self.optimizer.zero_grad()
                    loss, fake_targets_new_batch, fake_targets_old_batch, targets_new_batch, targets_old_batch, token_attention_mask = self._step_labeled(batch, "train")
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    # Averaging losses #
                    avg_loss += loss.detach().cpu().item()
                    # Displacy the Losses of the networks #
                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss))
                        if self.writer:
                            self.writer.add_scalar(tag='loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0
                    # Collect the generated labels by the forward and reverse networks #
                    true_labels_new_batch, gen_labels_new_batch = self._gen_labels(fake_targets_new_batch, targets_new_batch, token_attention_mask)
                    true_labels_old_batch, gen_labels_old_batch = self._gen_labels(fake_targets_old_batch, targets_old_batch, token_attention_mask)
                    gen_labels_new += gen_labels_new_batch
                    gen_labels_old += gen_labels_old_batch
                    true_labels_new += true_labels_new_batch
                    true_labels_old += true_labels_old_batch
                # Obtain the f1 scores of networks during training #
                _, _, f1_score_gen_forward, _, _, f1_score_gen_reverse = comp_f1_score(true_labels_new, true_labels_old, gen_labels_new, gen_labels_old)
                # Keep a record of the f1 scores of networks during training #
                if self.writer:
                    self.writer.add_scalar(tag='train_f1_forward_net', scalar_value=f1_score_gen_forward, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='train_f1_reverse_net', scalar_value=f1_score_gen_reverse, global_step=epoch)    # tensorbordx
                self.logger.info("Epoch {}/{}, forward network f1 train score: {:<6.5f}, reverse network f1 train score {:<6.5f}.".format(epoch, self.args.num_epochs, f1_score_gen_forward, f1_score_gen_reverse))
                # Evaluate the generators on the val dataset #
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
            torch.cuda.empty_cache()
            pbar.close()
            self.pbar = None
            self.logger.info("Get best forward network dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_forward, self.best_dev_forward))
            self.logger.info("Get best reverse network dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_reverse, self.best_dev_reverse))

    # def train_flexmatch(self):
    #     if self.args.use_prompt:
    #         self.training_settings_with_prompt()
    #     else:
    #         self.training_settings_text_only()

    #     self.model_dict["forward"].train()
    #     self.model_dict["reverse"].train()

    #     self.logger.info("***** Running semi-supvervised training *****")
    #     self.logger.info("  Num instance = %d", (len(self.train_data_labeled) + len(self.unlabeled_data)) * self.args.batch_size)
    #     self.logger.info("  Num epoch = %d", self.args.num_epochs)
    #     self.logger.info("  Batch size = %d", self.args.batch_size)
    #     self.logger.info("  Learning rate = {}".format(self.args.lr))
    #     self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)
    #     # Flexmatch parameters #
    #     tau_min = 0.5
    #     tau_max = 0.95
    #     smoothing = 0.9 # Momentum factor
    #     class_thresholds_forward = torch.full((self.num_labels,), tau_min).to(self.args.device)
    #     class_progress_forward = torch.zeros(self.num_labels).to(self.args.device)
    #     class_counters_forward = torch.zeros(self.num_labels).to(self.args.device)
    #     total_counters_forward = torch.zeros(self.num_labels).to(self.args.device)
    #     class_thresholds_reverse = torch.full((self.num_labels,), tau_min).to(self.args.device)
    #     class_progress_reverse = torch.zeros(self.num_labels).to(self.args.device)
    #     class_counters_reverse = torch.zeros(self.num_labels).to(self.args.device)
    #     total_counters_reverse = torch.zeros(self.num_labels).to(self.args.device)

    #     self.step = 0
    #     with tqdm(total=self.semi_train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
    #         avg_loss = 0
    #         sample_weights = None
    #         for epoch in range(self.args.num_epochs):
    #             pbar.set_description_str(desc="Epoch {}/{}".format(epoch+1, self.args.num_epochs))
    #             true_labels_new, true_labels_old, gen_labels_new, gen_labels_old, recon_errors = [], [], [], [], []
    #             train_labeled_iter = cycle(self.train_data_labeled)
    #             print(f"Forward net class thresholds are:\n{class_thresholds_forward}")
    #             print(f"Reverse net class thresholds are:\n{class_thresholds_reverse}")
    #             iter_data = self.unlabeled_data if epoch >= self.epoch_start_unlabeled else self.train_data_labeled
    #             unlabeled_weight_idx = 0
    #             for iter_batch in iter_data:
    #                 self.step += 1
    #                 self.optimizer.zero_grad()
    #                 # Pretrain on labeled data #
    #                 if epoch < self.epoch_start_unlabeled:
    #                     labeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in iter_batch)
    #                     loss, fake_targets_new_batch, fake_targets_old_batch, targets_new_batch, targets_old_batch, token_attention_mask = self._step_labeled(labeled_batch, "train")
    #                 # # Compute reconstruction errors for sample weighting #
    #                 # elif epoch == self.epoch_start_unlabeled:
    #                 #     unlabeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in iter_batch)
    #                 #     recon_errors_batch = self._gen_sample_recon_errors(unlabeled_batch)
    #                 #     recon_errors.append(recon_errors_batch)
    #                 #     continue
    #                 # Train on labeled + unlabeled data #
    #                 else:
    #                     labeled_batch = next(train_labeled_iter)
    #                     labeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in labeled_batch)
    #                     unlabeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in iter_batch)
    #                     # sample_weights, unlabeled_weight_idx = None, None
    #                     loss, fake_targets_new_batch, fake_targets_old_batch, targets_new_batch, targets_old_batch, token_attention_mask, unlabeled_weight_idx = self._step_flexmatch(labeled_batch, unlabeled_batch, \
    #                             class_thresholds_forward, class_thresholds_reverse, class_counters_forward, class_counters_reverse, total_counters_forward, total_counters_reverse, \
    #                             unlabeled_weights=sample_weights, unlabeled_weight_idx=unlabeled_weight_idx)
    #                 loss.backward()
    #                 self.optimizer.step()
    #                 self.scheduler.step()
    #                 # Averaging losses #
    #                 avg_loss += loss.detach().cpu().item()
    #                 # Displacy the Losses of the networks #
    #                 if self.step % self.refresh_step == 0:
    #                     avg_loss = float(avg_loss) / self.refresh_step
    #                     pbar.update(self.refresh_step)
    #                     pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss))
    #                     if self.writer:
    #                         self.writer.add_scalar(tag='loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
    #                     avg_loss = 0
    #                 # Collect the generated labels by the forward and reverse networks #
    #                 true_labels_new_batch, gen_labels_new_batch = self._gen_labels(fake_targets_new_batch, targets_new_batch, token_attention_mask)
    #                 true_labels_old_batch, gen_labels_old_batch = self._gen_labels(fake_targets_old_batch, targets_old_batch, token_attention_mask)
    #                 gen_labels_new += gen_labels_new_batch
    #                 gen_labels_old += gen_labels_old_batch
    #                 true_labels_new += true_labels_new_batch
    #                 true_labels_old += true_labels_old_batch
    #             # # Compute weights for data samples #
    #             # if epoch == self.epoch_start_unlabeled:
    #             #     recon_errors = torch.cat(recon_errors, dim=0)
    #             #     sample_weights = (1 - recon_errors/recon_errors.max())
    #             #     continue
    #             # Adjust thresholds #
    #             with torch.no_grad():
    #                 class_progress_forward = torch.where(total_counters_forward > 0, class_counters_forward / total_counters_forward, torch.zeros_like(class_counters_forward))
    #                 class_thresholds_forward = (1 - smoothing) * class_progress_forward + smoothing * class_thresholds_forward
    #                 class_thresholds_forward = torch.clamp(class_thresholds_forward, tau_min, tau_max)
    #             class_counters_forward.zero_()
    #             total_counters_forward.zero_()
    #             with torch.no_grad():
    #                 class_progress_reverse = torch.where(total_counters_reverse > 0, class_counters_reverse / total_counters_reverse, torch.zeros_like(class_counters_reverse))
    #                 class_thresholds_reverse = (1 - smoothing) * class_progress_reverse + smoothing * class_thresholds_reverse
    #                 class_thresholds_reverse = torch.clamp(class_thresholds_reverse, tau_min, tau_max)
    #             class_counters_reverse.zero_()
    #             total_counters_reverse.zero_()
    #             # Obtain the f1 scores of networks during training #
    #             _, _, f1_score_gen_forward, _, _, f1_score_gen_reverse = self._comp_fr_score(true_labels_new, true_labels_old, gen_labels_new, gen_labels_old)
    #             # Keep a record of the f1 scores of networks during training #
    #             if self.writer:
    #                 self.writer.add_scalar(tag='train_f1_forward_net', scalar_value=f1_score_gen_forward, global_step=epoch)    # tensorbordx
    #                 self.writer.add_scalar(tag='train_f1_reverse_net', scalar_value=f1_score_gen_reverse, global_step=epoch)    # tensorbordx
    #             self.logger.info("Epoch {}/{}, forward network f1 train score: {:<6.5f}, reverse network f1 train score {:<6.5f}.".format(epoch, self.args.num_epochs, f1_score_gen_forward, f1_score_gen_reverse))
    #             # Evaluate the generators on the val dataset #
    #             if epoch >= self.args.eval_begin_epoch:
    #                 self.evaluate(epoch)
    #         torch.cuda.empty_cache()
    #         pbar.close()
    #         self.pbar = None
    #         self.logger.info("Get best forward network dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_forward, self.best_dev_forward))
    #         self.logger.info("Get best reverse network dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_reverse, self.best_dev_reverse))

    # def train_mean_teacher(self):
    #     if self.args.use_prompt:
    #         self.training_settings_with_prompt()
    #     else:
    #         self.training_settings_text_only()

    #     self.model_dict["forward"].train()
    #     self.model_dict["reverse"].train()

    #     self.logger.info("***** Running semi-supvervised training *****")
    #     self.logger.info("  Num instance = %d", (len(self.train_data_labeled) + len(self.unlabeled_data)) * self.args.batch_size)
    #     self.logger.info("  Num epoch = %d", self.args.num_epochs)
    #     self.logger.info("  Batch size = %d", self.args.batch_size)
    #     self.logger.info("  Learning rate = {}".format(self.args.lr))
    #     self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

    #     self.step = 0
    #     with tqdm(total=self.semi_train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
    #         avg_loss = 0
    #         for epoch in range(self.args.num_epochs):
    #             pbar.set_description_str(desc="Epoch {}/{}".format(epoch+1, self.args.num_epochs))
    #             true_labels_new, true_labels_old, gen_labels_new, gen_labels_old = [], [], [], []
    #             train_labeled_iter = cycle(self.train_data_labeled)
    #             iter_data = self.unlabeled_data if epoch >= self.epoch_start_unlabeled else self.train_data_labeled
    #             for iter_batch in iter_data:
    #                 self.step += 1
    #                 self.optimizer.zero_grad()
    #                 # Pretrain on labeled data #
    #                 if epoch < self.epoch_start_unlabeled:
    #                     labeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in iter_batch)
    #                     loss, fake_targets_new_batch, fake_targets_old_batch, targets_new_batch, targets_old_batch, token_attention_mask = self._step_labeled(labeled_batch, "train")
    #                 # Train on labeled + unlabeled data #
    #                 else:
    #                     labeled_batch = next(train_labeled_iter)
    #                     labeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in labeled_batch)
    #                     unlabeled_batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in iter_batch)
    #                     # Initialize Mean Teacher models for both forward and reverse diffusion #
    #                     if epoch == self.epoch_start_unlabeled:
    #                         teacher_forward = MeanTeacher(self.model_dict["forward"])
    #                         teacher_reverse = MeanTeacher(self.model_dict["reverse"])
    #                     loss, fake_targets_new_batch, fake_targets_old_batch, targets_new_batch, targets_old_batch, token_attention_mask = self._step_mean_teacher(teacher_forward, teacher_reverse, labeled_batch, unlabeled_batch)
    #                 loss.backward()
    #                 self.optimizer.step()
    #                 self.scheduler.step()
    #                 # Update Teacher Models #
    #                 if epoch > self.epoch_start_unlabeled:
    #                     teacher_forward.update_teacher(self.model_dict["forward"])
    #                     teacher_reverse.update_teacher(self.model_dict["reverse"])
    #                 # Averaging losses #
    #                 avg_loss += loss.detach().cpu().item()
    #                 # Displacy the Losses of the networks #
    #                 if self.step % self.refresh_step == 0:
    #                     avg_loss = float(avg_loss) / self.refresh_step
    #                     pbar.update(self.refresh_step)
    #                     pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss))
    #                     if self.writer:
    #                         self.writer.add_scalar(tag='loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
    #                     avg_loss = 0
    #                 # Collect the generated labels by the forward and reverse networks #
    #                 true_labels_new_batch, gen_labels_new_batch = self._gen_labels(fake_targets_new_batch, targets_new_batch, token_attention_mask)
    #                 true_labels_old_batch, gen_labels_old_batch = self._gen_labels(fake_targets_old_batch, targets_old_batch, token_attention_mask)
    #                 gen_labels_new += gen_labels_new_batch
    #                 gen_labels_old += gen_labels_old_batch
    #                 true_labels_new += true_labels_new_batch
    #                 true_labels_old += true_labels_old_batch
    #             # Obtain the f1 scores of networks during training #
    #             _, _, f1_score_gen_forward, _, _, f1_score_gen_reverse = self._comp_fr_score(true_labels_new, true_labels_old, gen_labels_new, gen_labels_old)
    #             # Keep a record of the f1 scores of networks during training #
    #             if self.writer:
    #                 self.writer.add_scalar(tag='train_f1_forward_net', scalar_value=f1_score_gen_forward, global_step=epoch)    # tensorbordx
    #                 self.writer.add_scalar(tag='train_f1_reverse_net', scalar_value=f1_score_gen_reverse, global_step=epoch)    # tensorbordx
    #             self.logger.info("Epoch {}/{}, forward network f1 train score: {:<6.5f}, reverse network f1 train score {:<6.5f}.".format(epoch, self.args.num_epochs, f1_score_gen_forward, f1_score_gen_reverse))
    #             # Evaluate the generators on the val dataset #
    #             if epoch >= self.args.eval_begin_epoch:
    #                 self.evaluate(epoch)
    #         torch.cuda.empty_cache()
    #         pbar.close()
    #         self.pbar = None
    #         self.logger.info("Get best forward network dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_forward, self.best_dev_forward))
    #         self.logger.info("Get best reverse network dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_reverse, self.best_dev_reverse))
    
    def evaluate(self, epoch):
        self.model_dict["forward"].eval()
        self.model_dict["reverse"].eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instance = %d", len(self.val_data_labeled)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data_labeled), leave=False, dynamic_ncols=True) as pbar:
                f1_score_forward, f1_score_reverse = self._eval_labels(pbar, self.val_data_labeled, "eval")
                # Evaluate and save the best forward network #
                self.logger.info("Epoch {}/{}, best dev f1 forward: {:<6.5f}, best epoch: {}, current dev f1 score forward: {:<6.5f}."
                                 .format(epoch, self.args.num_epochs, self.best_dev_forward, self.best_dev_epoch_forward, f1_score_forward))
                if f1_score_forward >= self.best_dev_forward:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch_forward = epoch
                    self.best_dev_forward = f1_score_forward # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model_dict["forward"].state_dict(), self.args.save_path+"/best_forward.pth")
                        self.logger.info("Save best forward network at {}".format(self.args.save_path))
                # Evaluate and save the best reverse network #
                self.logger.info("Epoch {}/{}, best dev f1 reverse: {:<6.5f}, best epoch: {}, current dev f1 score reverse: {:<6.5f}."
                                 .format(epoch, self.args.num_epochs, self.best_dev_reverse, self.best_dev_epoch_reverse, f1_score_reverse))
                if f1_score_reverse >= self.best_dev_reverse:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch_reverse = epoch
                    self.best_dev_reverse = f1_score_reverse # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model_dict["reverse"].state_dict(), self.args.save_path+"/best_reverse.pth")
                        self.logger.info("Save best reverse network at {}".format(self.args.save_path))

        self.model_dict["forward"].train()
        self.model_dict["reverse"].train()
    
    def test(self):
        self.model_dict["forward"].eval()
        self.model_dict["reverse"].eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data_labeled)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        # if self.args.load_path is not None:  # load model from load_path
        #     self.logger.info("Loading models from {}".format(self.args.load_path))
        #     self.model_dict["forward"].load_state_dict(torch.load(self.args.load_path + "forward.pth"))
        #     self.model_dict["reverse"].load_state_dict(torch.load(self.args.load_path + "reverse.pth"))
        #     self.logger.info("Load model successful!")

        with torch.no_grad():
            with tqdm(total=len(self.test_data_labeled), leave=False, dynamic_ncols=True) as pbar:
                precision_forward, recall_forward, f1_score_forward, precision_reverse, recall_reverse, f1_score_reverse = self._eval_labels(pbar, self.test_data_labeled, "test")
                self.logger.info("Forward Network Test precision: {:<6.5f}, recall: {:<6.5f}, f1 score: {:<6.5f}.".format(precision_forward, recall_forward, f1_score_forward))
                self.logger.info("Reverse Network Test precision: {:<6.5f}, recall: {:<6.5f}, f1 score: {:<6.5f}.".format(precision_reverse, recall_reverse, f1_score_reverse))

        self.model_dict["forward"].train()
        self.model_dict["reverse"].train()

        return precision_forward, recall_forward, f1_score_forward, precision_reverse, recall_reverse, f1_score_reverse
    
    def predict(self):
        self.model_dict["forward"].eval()
        self.model_dict["reverse"].eval()
        self.logger.info("\n***** Running Prediction *****")
        self.logger.info("  Num instance = %d", len(self.unlabeled_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model_dict["forward"].load_state_dict(torch.load(self.args.load_path + "forward.pth"))
            self.model_dict["reverse"].load_state_dict(torch.load(self.args.load_path + "reverse.pth"))
            self.model_dict["forward"].to(self.args.device)
            self.model_dict["reverse"].to(self.args.device)
            self.logger.info("Load model successful!")

        with torch.no_grad():
            with tqdm(total=len(self.unlabeled_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description(desc="Predicting Unlabeled Data")
                given_labels, pred_labels_new, pred_labels_old, words, img_names = [], [], [], [], []
                for batch in self.unlabeled_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    targets_being_new, probs_being_new, targets_being_old, probs_being_old, targets_unk, token_attention_mask, words_batch, img_names_batch = self._step_predict(batch)
                    probs_being_new = softmax(probs_being_new, dim=-1)
                    probs_being_old = softmax(probs_being_old, dim=-1)
                    given_labels_batch, pred_labels_new_batch, _ = self._gen_labels(targets_being_new, targets_unk, token_attention_mask, words_batch)
                    _, pred_labels_old_batch, words_batch = self._gen_labels(targets_being_old, targets_unk, token_attention_mask, words_batch)
                    given_labels += given_labels_batch
                    pred_labels_new += pred_labels_new_batch
                    pred_labels_old += pred_labels_old_batch
                    words += words_batch
                    img_names += img_names_batch
                    pbar.update()
                pbar.close()

                with open("unlabeled_pred.txt", "w+") as f:
                    for word_sent, label_sent, pred_new_sent, pred_old_sent, img_name in zip(words, given_labels, pred_labels_new, pred_labels_old, img_names):
                        f.write("IMGID: " + img_name + '\n')
                        for word, label, pred_n, pred_o in zip(word_sent, label_sent, pred_new_sent, pred_old_sent):
                            f.write(word + '\t' + label + '\t' + pred_n + '\t' + pred_o + '\n')
                        f.write('\n')
    
    def _step_labeled(self, batch, mode="train"):
        # Extract data from a batch #
        if self.args.use_prompt:
            char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_old, targets_new, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, _ = batch
        else:
            char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_old, targets_new, words, _ = batch
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))
        # Specify which images and auxiliary images are feeding into the networks #
        if self.model_dict["forward"].vt_model_name == "hvpnet":
            imgs = hvp_imgs
            aux_imgs = hvp_aux_imgs
        elif self.model_dict["forward"].vt_model_name == "mkgformer":
            imgs = mkg_imgs
            aux_imgs = mkg_aux_imgs
        else:
            imgs = None
            aux_imgs = None

        if mode == "train":
            t = torch.randint(0, self.args.train_steps, (targets_old.shape[0], 1), device=self.args.device)  # Random time step
            # new -> new #
            _, id_emissions_new = self.model_dict["forward"](t, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=targets_new, images=imgs, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
            # old -> old #
            _, id_emissions_old = self.model_dict["reverse"](t, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=targets_old, images=imgs, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
            # Identity Loss #
            id_loss_forward = self.crit_id(id_emissions_new, targets_new)
            id_loss_reverse = self.crit_id(id_emissions_old, targets_old)
            # old -> new #
            fake_targets_new, fake_emissions_new, fake_features_new = self.model_dict["forward"](t, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=targets_old, images=imgs, \
                aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs, return_features=True)
            # new -> old #
            fake_targets_old, fake_emissions_old, fake_features_old = self.model_dict["reverse"](t, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=targets_new, images=imgs, \
                aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs, return_features=True)
            # Edit Loss #
            edit_loss_forward = self.crit_edit(fake_emissions_new, targets_new)
            edit_loss_reverse = self.crit_edit(fake_emissions_old, targets_old)
            # old -> new -> old #
            _, recon_emissions_old = self.model_dict["reverse"](t, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=fake_emissions_new, images=imgs, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
            # new -> old -> new #
            _, recon_emissions_new = self.model_dict["forward"](t, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=fake_emissions_old, images=imgs, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
            # Cycle Loss #
            cycle_loss_old = self.crit_cycle(recon_emissions_old, targets_old)
            cycle_loss_new = self.crit_cycle(recon_emissions_new, targets_new)
            # Contrastive Loss #
            contrastive_loss = self.crit_contrast(fake_features_old, fake_features_new)
            # contrastive_loss = 0
            # All #
            loss = self.args.lambda_id * (id_loss_forward + id_loss_reverse) + self.args.lambda_edit * (edit_loss_forward + edit_loss_reverse) + self.args.lambda_cycle * (cycle_loss_old + cycle_loss_new) \
                + self.args.lambda_contrast * contrastive_loss
            # loss = self.args.lambda_id * (id_loss_forward + id_loss_reverse) + self.args.lambda_edit * (edit_loss_forward + edit_loss_reverse)
            return loss, fake_targets_new, fake_targets_old, targets_new, targets_old, token_attention_mask
        elif mode == "eval":
            targets_new_t, _ = self._reverse_diffusion(char_input_ids, token_input_ids, token_attention_mask, token_type_ids, imgs, aux_imgs, rcnn_imgs, dir="forward", targets=targets_old)
            targets_old_t, _ = self._reverse_diffusion(char_input_ids, token_input_ids, token_attention_mask, token_type_ids, imgs, aux_imgs, rcnn_imgs, dir="reverse", targets=targets_new)
            return targets_new_t, targets_old_t, targets_new, targets_old, token_attention_mask
        else:
            return
        
    def _step_predict(self, batch):
        # Extract data from a batch #
        if self.args.use_prompt:
            char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_unk, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, img_names = batch
        else:
            char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_unk, words, img_name = batch
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))
        # Specify which images and auxiliary images are feeding into the networks #
        if self.model_dict["forward"].vt_model_name == "hvpnet":
            imgs = hvp_imgs
            aux_imgs = hvp_aux_imgs
        elif self.model_dict["forward"].vt_model_name == "mkgformer":
            imgs = mkg_imgs
            aux_imgs = mkg_aux_imgs
        else:
            imgs = None
            aux_imgs = None
        # Make Predictions #
        targets_being_new, probs_being_new = self._reverse_diffusion(char_input_ids, token_input_ids, token_attention_mask, token_type_ids, imgs, aux_imgs, rcnn_imgs, dir="forward", targets=targets_unk)
        targets_being_old, probs_being_old = self._reverse_diffusion(char_input_ids, token_input_ids, token_attention_mask, token_type_ids, imgs, aux_imgs, rcnn_imgs, dir="reverse", targets=targets_unk)
        return targets_being_new, probs_being_new, targets_being_old, probs_being_old, targets_unk, token_attention_mask, words, img_names
    
    # def _step_flexmatch(self, labeled_batch, unlabeled_batch, class_thresholds_forward, class_thresholds_reverse, class_counters_forward, class_counters_reverse, \
    #                     total_counters_forward, total_counters_reverse, unlabeled_weights=None, unlabeled_weight_idx=None):
    #     # Extract labeled and unlabeled batches #
    #     if self.args.use_prompt:
    #         char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_old, targets_new, hvp_imgs_labeled, hvp_aux_imgs_labeled, \
    #             mkg_imgs_labeled, mkg_aux_imgs_labeled, rcnn_imgs_labeled, words, _ = labeled_batch
    #         weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_type_ids, weak_aug_token_attention_mask, weak_aug_targets_unk, \
    #             hvp_imgs_unlabeled, hvp_aux_imgs_unlabeled, mkg_imgs_unlabeled, mkg_aux_imgs_unlabeled, rcnn_imgs_unlabeled, \
    #             strong_aug_hvp_imgs_unlabeled, strong_aug_hvp_aux_imgs_unlabeled, strong_aug_mkg_imgs_unlabeled, strong_aug_mkg_aux_imgs_unlabeled, strong_aug_rcnn_imgs_unlabeled, weak_aug_words, _ = unlabeled_batch
    #     else:
    #         hvp_imgs_labeled, hvp_aux_imgs_labeled, mkg_imgs_labeled, mkg_aux_imgs_labeled, rcnn_imgs_labeled = None, None, None, None, None
    #         hvp_imgs_unlabeled, hvp_aux_imgs_unlabeled, mkg_imgs_unlabeled, mkg_aux_imgs_unlabeled, rcnn_imgs_unlabeled = None, None, None, None, None
    #         strong_aug_hvp_imgs_unlabeled, strong_aug_hvp_aux_imgs_unlabeled, strong_aug_mkg_imgs_unlabeled, strong_aug_mkg_aux_imgs_unlabeled, strong_aug_rcnn_imgs_unlabeled = None, None, None, None, None
    #         char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_old, targets_new, words, _ = labeled_batch
    #         weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_type_ids, weak_aug_token_attention_mask, weak_aug_targets_unk, weak_aug_words, _ = unlabeled_batch
    #     if self.model_dict["forward"].vt_model_name == "hvpnet":
    #         imgs_labeled = hvp_imgs_labeled
    #         aux_imgs_labeled = hvp_aux_imgs_labeled
    #         imgs_unlabeled = hvp_imgs_unlabeled
    #         aux_imgs_unlabeled = hvp_aux_imgs_unlabeled
    #         strong_aug_imgs_unlabeled = strong_aug_hvp_imgs_unlabeled
    #         strong_aug_aux_imgs_unlabeled = strong_aug_hvp_aux_imgs_unlabeled
    #     elif self.model_dict["forward"].vt_model_name == "mkgformer":
    #         imgs_labeled = mkg_imgs_labeled
    #         aux_imgs_labeled = mkg_aux_imgs_labeled
    #         imgs_unlabeled = mkg_imgs_unlabeled
    #         aux_imgs_unlabeled = mkg_aux_imgs_unlabeled
    #         strong_aug_imgs_unlabeled = strong_aug_mkg_imgs_unlabeled
    #         strong_aug_aux_imgs_unlabeled = strong_aug_mkg_aux_imgs_unlabeled
    #     else:
    #         imgs_labeled, imgs_unlabeled, strong_aug_imgs_unlabeled, aux_imgs_labeled, aux_imgs_unlabeled, strong_aug_aux_imgs_unlabeled = None, None, None, None, None, None
    #     words = list(map(list, zip(*words)))
    #     weak_aug_words = list(map(list, zip(*weak_aug_words)))
    #     # Generate pseudo labels #
    #     self.model_dict["forward"].eval()
    #     self.model_dict["reverse"].eval()        
    #     gen_targets_new_unlabeled, probs_being_new_unlabeled = self._reverse_diffusion(weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #         imgs_unlabeled, aux_imgs_unlabeled, rcnn_imgs_unlabeled, dir="forward", targets=weak_aug_targets_unk)
    #     gen_targets_old_unlabeled, probs_being_old_unlabeled = self._reverse_diffusion(weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #         imgs_unlabeled, aux_imgs_unlabeled, rcnn_imgs_unlabeled, dir="reverse", targets=weak_aug_targets_unk)
    #     pseudo_targets_new, pseudo_targets_old, confidence_scores_forward, confidence_scores_reverse, pseudo_targets_new_mask, pseudo_targets_old_mask = self._gen_pseudo_targets_flexmatch(gen_targets_new_unlabeled, \
    #         probs_being_new_unlabeled, gen_targets_old_unlabeled, probs_being_old_unlabeled, weak_aug_targets_unk, weak_aug_token_attention_mask, class_thresholds_forward, class_thresholds_reverse)
    #     # Trial #
    #     pseudo_targets_mask = pseudo_targets_new_mask | pseudo_targets_old_mask
    #     pseudo_targets_old_mask = pseudo_targets_mask
    #     pseudo_targets_new_mask = pseudo_targets_mask
    #     # Update class-wise progress and counters #
    #     with torch.no_grad():
    #         pseudo_targets_new_confidences = confidence_scores_forward
    #         pseudo_targets_old_confidences = confidence_scores_reverse
    #         for c in range(self.num_labels):
    #             high_conf_mask_new = (pseudo_targets_new == c) & (pseudo_targets_new_confidences >= class_thresholds_forward[c])
    #             high_conf_mask_old = (pseudo_targets_old == c) & (pseudo_targets_old_confidences >= class_thresholds_reverse[c])
    #             class_counters_forward[c] += high_conf_mask_new.sum().item()
    #             class_counters_reverse[c] += high_conf_mask_old.sum().item()
    #             total_counters_forward[c] += (pseudo_targets_new == c).sum().item()
    #             total_counters_reverse[c] += (pseudo_targets_old == c).sum().item()
    #     self.model_dict["forward"].train()
    #     self.model_dict["reverse"].train()
    #     # Training with labeled and unlabeled data #
    #     t_labeled = torch.randint(0, self.args.train_steps, (targets_old.shape[0], 1), device=self.args.device)  # Random time step
    #     t_unlabeled = torch.randint(0, self.args.train_steps, (pseudo_targets_old.shape[0], 1), device=self.args.device)  # Random time step
    #     # Labeled Identity Loss #
    #     _, id_emissions_new_labeled = self.model_dict["forward"](t_labeled, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=targets_new, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # new -> new
    #     _, id_emissions_old_labeled = self.model_dict["reverse"](t_labeled, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=targets_old, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # old -> old
    #     id_loss_forward_labeled = self.crit_id(id_emissions_new_labeled, targets_new)
    #     id_loss_reverse_labeled = self.crit_id(id_emissions_old_labeled, targets_old)
    #     # Labeled Edit Loss #
    #     fake_targets_new_labeled, fake_emissions_new_labeled = self.model_dict["forward"](t_labeled, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, \
    #         labels=targets_old, images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # old -> new #
    #     fake_targets_old_labeled, fake_emissions_old_labeled = self.model_dict["reverse"](t_labeled, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, \
    #         labels=targets_new, images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # new -> old #
    #     edit_loss_forward_labeled = self.crit_edit(fake_emissions_new_labeled, targets_new)
    #     edit_loss_reverse_labeled = self.crit_edit(fake_emissions_old_labeled, targets_old)
    #     # Labeled Cycle Loss #
    #     _, recon_emissions_old_labeled = self.model_dict["reverse"](t_labeled, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=fake_emissions_new_labeled, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # old -> new -> old #
    #     _, recon_emissions_new_labeled = self.model_dict["forward"](t_labeled, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, labels=fake_emissions_old_labeled, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # new -> old -> new #
    #     cycle_loss_old_labeled = self.crit_cycle(recon_emissions_old_labeled, targets_old)
    #     cycle_loss_new_labeled = self.crit_cycle(recon_emissions_new_labeled, targets_new)
    #     # Labeled Loss All #
    #     loss_labeled = self.args.lambda_id * (id_loss_forward_labeled + id_loss_reverse_labeled) + self.args.lambda_edit * (edit_loss_forward_labeled + edit_loss_reverse_labeled) \
    #         + self.args.lambda_cycle * (cycle_loss_old_labeled + cycle_loss_new_labeled)
    #     # Unlabeled Consistency Loss #
    #     _, id_emissions_new_unlabeled = self.model_dict["forward"](t_unlabeled, weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #         labels=pseudo_targets_new, images=strong_aug_imgs_unlabeled, aux_imgs=strong_aug_aux_imgs_unlabeled, rcnn_imgs=strong_aug_rcnn_imgs_unlabeled) # new -> new
    #     _, id_emissions_old_unlabeled = self.model_dict["reverse"](t_unlabeled, weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #         labels=pseudo_targets_old, images=strong_aug_imgs_unlabeled, aux_imgs=strong_aug_aux_imgs_unlabeled, rcnn_imgs=strong_aug_rcnn_imgs_unlabeled) # old -> old
    #     _, fake_emissions_new_unlabeled = self.model_dict["forward"](t_unlabeled, weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #         labels=pseudo_targets_old, images=strong_aug_imgs_unlabeled, aux_imgs=strong_aug_aux_imgs_unlabeled, rcnn_imgs=strong_aug_rcnn_imgs_unlabeled) # old -> new
    #     _, fake_emissions_old_unlabeled = self.model_dict["reverse"](t_unlabeled, weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #         labels=pseudo_targets_new, images=strong_aug_imgs_unlabeled, aux_imgs=strong_aug_aux_imgs_unlabeled, rcnn_imgs=strong_aug_rcnn_imgs_unlabeled) # new -> old
        
    #     consistency_loss_unlabeled = 0
    #     if unlabeled_weights is not None:
    #         bsz = weak_aug_char_input_ids.shape[0]
    #         batch_weights = unlabeled_weights[unlabeled_weight_idx:unlabeled_weight_idx + bsz]
    #         consistency_loss_forward_new = self.crit_id_weighted(id_emissions_new_unlabeled, pseudo_targets_new, pseudo_targets_new_mask, batch_weights)
    #         consistency_loss_reverse_old = self.crit_id_weighted(id_emissions_old_unlabeled, pseudo_targets_old, pseudo_targets_old_mask, batch_weights)
    #         consistency_loss_forward_old = self.crit_id_weighted(fake_emissions_new_unlabeled, pseudo_targets_new, pseudo_targets_new_mask, batch_weights)
    #         consistency_loss_reverse_new = self.crit_id_weighted(fake_emissions_old_unlabeled , pseudo_targets_old, pseudo_targets_old_mask, batch_weights)
    #     else:
    #         consistency_loss_forward_new = self.crit_id(id_emissions_new_unlabeled[pseudo_targets_new_mask], pseudo_targets_new[pseudo_targets_new_mask])
    #         consistency_loss_reverse_old = self.crit_id(id_emissions_old_unlabeled[pseudo_targets_old_mask], pseudo_targets_old[pseudo_targets_old_mask])
    #         consistency_loss_forward_old = self.crit_id(fake_emissions_new_unlabeled[pseudo_targets_new_mask], pseudo_targets_new[pseudo_targets_new_mask])
    #         consistency_loss_reverse_new = self.crit_id(fake_emissions_old_unlabeled[pseudo_targets_old_mask], pseudo_targets_old[pseudo_targets_old_mask])
    #     consistency_loss_unlabeled = consistency_loss_forward_new + consistency_loss_reverse_old + consistency_loss_forward_old + consistency_loss_reverse_new
    #     # Loss All #
    #     loss = loss_labeled + consistency_loss_unlabeled
    #     if unlabeled_weights is not None:
    #         unlabeled_weight_idx = unlabeled_weight_idx + bsz
    #         return loss, fake_targets_new_labeled, fake_targets_old_labeled, targets_new, targets_old, token_attention_mask, unlabeled_weight_idx
    #     else:
    #         return loss, fake_targets_new_labeled, fake_targets_old_labeled, targets_new, targets_old, token_attention_mask, None
        
    # def _step_mean_teacher(self, teacher_forward, teacher_reverse, labeled_batch, unlabeled_batch):
    #     # Extract data from a batch #
    #     if self.args.use_prompt:
    #         char_input_ids_labeled, token_input_ids_labeled, token_type_ids_labeled, token_attention_mask_labeled, targets_old, targets_new, hvp_imgs_labeled, hvp_aux_imgs_labeled, \
    #             mkg_imgs_labeled, mkg_aux_imgs_labeled, rcnn_imgs_labeled, words_labeled, _ = labeled_batch
    #         char_input_ids_unlabeled, token_input_ids_unlabeled, token_type_ids_unlabeled, token_attention_mask_unlabeled, targets_unk, hvp_imgs_unlabeled, \
    #             hvp_aux_imgs_unlabeled, mkg_imgs_unlabeled, mkg_aux_imgs_unlabeled, rcnn_imgs_unlabeled, words_unlabeled, _ = unlabeled_batch
    #     else:
    #         char_input_ids_labeled, token_input_ids_labeled, token_type_ids_labeled, token_attention_mask_labeled, targets_old, targets_new, words_labeled, _ = labeled_batch
    #         hvp_imgs_labeled, hvp_aux_imgs_labeled, mkg_imgs_labeled, mkg_aux_imgs_labeled, rcnn_imgs_labeled = None, None, None, None, None
    #         char_input_ids_unlabeled, token_input_ids_unlabeled, token_type_ids_unlabeled, token_attention_mask_unlabeled, targets_unk, words_unlabeled, _ = unlabeled_batch
    #         hvp_imgs_unlabeled, hvp_aux_imgs_unlabeled, mkg_imgs_unlabeled, mkg_aux_imgs_unlabeled, rcnn_imgs_unlabeled = None, None, None, None, None
    #     words_labeled = list(map(list, zip(*words_labeled)))
    #     words_unlabeled = list(map(list, zip(*words_unlabeled)))
    #     # Specify which images and auxiliary images are feeding into the networks #
    #     if self.model_dict["forward"].vt_model_name == "hvpnet":
    #         imgs_labeled = hvp_imgs_labeled
    #         aux_imgs_labeled = hvp_aux_imgs_labeled
    #         imgs_unlabeled = hvp_imgs_unlabeled
    #         aux_imgs_unlabeled = hvp_aux_imgs_unlabeled
    #     elif self.model_dict["forward"].vt_model_name == "mkgformer":
    #         imgs_labeled = mkg_imgs_labeled
    #         aux_imgs_labeled = mkg_aux_imgs_labeled
    #         imgs_unlabeled = mkg_imgs_unlabeled
    #         aux_imgs_unlabeled = mkg_aux_imgs_unlabeled
    #     else:
    #         imgs_labeled, imgs_unlabeled, aux_imgs_labeled, aux_imgs_unlabeled = None, None, None, None
    #     # Training with labeled and unlabeled data #
    #     t = torch.randint(0, self.args.train_steps, (targets_old.shape[0], 1), device=self.args.device)  # Random time step
    #     # Labeled Identity Loss #
    #     _, id_emissions_new_labeled = self.model_dict["forward"](t, char_input_ids_labeled, token_input_ids_labeled, token_attention_mask_labeled, token_type_ids_labeled, labels=targets_new, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # new -> new
    #     _, id_emissions_old_labeled = self.model_dict["reverse"](t, char_input_ids_labeled, token_input_ids_labeled, token_attention_mask_labeled, token_type_ids_labeled, labels=targets_old, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # old -> old
    #     id_loss_forward_labeled = self.crit_id(id_emissions_new_labeled, targets_new)
    #     id_loss_reverse_labeled = self.crit_id(id_emissions_old_labeled, targets_old)
    #     # Labeled Edit Loss #
    #     fake_targets_new_labeled, fake_emissions_new_labeled, fake_features_new_labeled = self.model_dict["forward"](t, char_input_ids_labeled, token_input_ids_labeled, token_attention_mask_labeled, token_type_ids_labeled, \
    #         labels=targets_old, images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled, return_features=True) # old -> new #
    #     fake_targets_old_labeled, fake_emissions_old_labeled, fake_features_old_labeled = self.model_dict["reverse"](t, char_input_ids_labeled, token_input_ids_labeled, token_attention_mask_labeled, token_type_ids_labeled, \
    #         labels=targets_new, images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled, return_features=True) # new -> old #
    #     edit_loss_forward_labeled = self.crit_edit(fake_emissions_new_labeled, targets_new)
    #     edit_loss_reverse_labeled = self.crit_edit(fake_emissions_old_labeled, targets_old)
    #     # Labeled Cycle Loss #
    #     _, recon_emissions_old_labeled = self.model_dict["reverse"](t, char_input_ids_labeled, token_input_ids_labeled, token_attention_mask_labeled, token_type_ids_labeled, labels=fake_emissions_new_labeled, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # old -> new -> old #
    #     _, recon_emissions_new_labeled = self.model_dict["forward"](t, char_input_ids_labeled, token_input_ids_labeled, token_attention_mask_labeled, token_type_ids_labeled, labels=fake_emissions_old_labeled, \
    #         images=imgs_labeled, aux_imgs=aux_imgs_labeled, rcnn_imgs=rcnn_imgs_labeled) # new -> old -> new #
    #     cycle_loss_old_labeled = self.crit_cycle(recon_emissions_old_labeled, targets_old)
    #     cycle_loss_new_labeled = self.crit_cycle(recon_emissions_new_labeled, targets_new)
    #     # Labeled Contrastive Loss #
    #     contrastive_loss_labeled = self.crit_contrast(fake_features_old_labeled, fake_features_new_labeled)
    #     # Labeled Loss All #
    #     loss_labeled = self.args.lambda_id * (id_loss_forward_labeled + id_loss_reverse_labeled) + self.args.lambda_edit * (edit_loss_forward_labeled + edit_loss_reverse_labeled) \
    #         + self.args.lambda_cycle * (cycle_loss_old_labeled + cycle_loss_new_labeled) + self.args.lambda_contrast * contrastive_loss_labeled
    #     # Mean Teacher Loss #
    #     with torch.no_grad():
    #         _, pseudo_emissions_new_unlabeled = teacher_forward(t, char_input_ids_unlabeled, token_input_ids_unlabeled, token_attention_mask_unlabeled, token_type_ids_unlabeled, labels=targets_unk, \
    #             images=imgs_unlabeled, aux_imgs=aux_imgs_unlabeled, rcnn_imgs=rcnn_imgs_unlabeled)
    #         _, pseudo_emissions_old_unlabeled = teacher_reverse(t, char_input_ids_unlabeled, token_input_ids_unlabeled, token_attention_mask_unlabeled, token_type_ids_unlabeled, labels=targets_unk, \
    #             images=imgs_unlabeled, aux_imgs=aux_imgs_unlabeled, rcnn_imgs=rcnn_imgs_unlabeled)
    #     _, student_emissions_new_unlabeled, student_features_new_unlabeled = self.model_dict["forward"](t, char_input_ids_unlabeled, token_input_ids_unlabeled, token_attention_mask_unlabeled, token_type_ids_unlabeled, labels=targets_unk, \
    #         images=imgs_unlabeled, aux_imgs=aux_imgs_unlabeled, rcnn_imgs=rcnn_imgs_unlabeled, return_features=True)
    #     _, student_emissions_old_unlabeled, student_features_old_unlabeled = self.model_dict["reverse"](t, char_input_ids_unlabeled, token_input_ids_unlabeled, token_attention_mask_unlabeled, token_type_ids_unlabeled, labels=targets_unk, \
    #         images=imgs_unlabeled, aux_imgs=aux_imgs_unlabeled, rcnn_imgs=rcnn_imgs_unlabeled, return_features=True)
    #     consistency_loss_new = self.crit_consistentcy(student_emissions_new_unlabeled, pseudo_emissions_new_unlabeled)
    #     consistency_loss_old = self.crit_consistentcy(student_emissions_old_unlabeled, pseudo_emissions_old_unlabeled)
    #     contrastive_loss_unlabeled = self.crit_contrast(student_features_old_unlabeled, student_features_new_unlabeled)
    #     mean_teacher_loss = consistency_loss_new + consistency_loss_old + contrastive_loss_unlabeled
    #     # Loss All #
    #     loss = loss_labeled + mean_teacher_loss
    #     return loss, fake_targets_new_labeled, fake_targets_old_labeled, targets_new, targets_old, token_attention_mask_labeled
        
        
    def _reverse_diffusion(self, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, images, aux_imgs, rcnn_imgs, dir="forward", targets=None):
        embeddings_t = self.model_dict[dir].get_label_embedding(targets, token_attention_mask)
        for t in reversed(range(self.args.eva_steps)):
            t_tensor = torch.full((targets.shape[0], 1), t).to(self.args.device)
            targets_t, emissions_t = self.model_dict[dir].denoise(embeddings_t, t_tensor, char_input_ids, token_input_ids, token_attention_mask, token_type_ids, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
            embeddings_t = self.model_dict[dir].get_label_embedding(targets_t, token_attention_mask)
        return targets_t, emissions_t
    
    def _gen_labels(self, logits, targets, token_attention_mask, words=None):
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().tolist()
        label_ids = targets.detach()
        token_attention_mask = token_attention_mask.detach()
        label_map = {idx:label for label, idx in self.label_map.items()}
        
        true_label_batch, pred_label_batch = [], []
        if words:
            words_batch = []
        for row in range(token_attention_mask.shape[0]):
            label_row_masked = torch.masked_select(label_ids[row], token_attention_mask[row].bool()).cpu().numpy()
            true_label_sent, pred_label_sent = [], []
            if words:
                word_sent = []
            for column in range(1, len(label_row_masked)):
                # if column == 0:
                #     continue
                if label_map[label_row_masked[column]] != "X" and label_map[label_row_masked[column]] != "[SEP]":
                    true_label_sent.append(label_map[label_row_masked[column]])
                    pred_label_sent.append(label_map[logits[row][column]])
                    if words:
                        word_sent.append(words[row][column])
            true_label_batch.append(true_label_sent)
            pred_label_batch.append(pred_label_sent)
            if words:
                words_batch.append(word_sent)
        
        if words:
            return true_label_batch, pred_label_batch, words_batch
        else:
            return true_label_batch, pred_label_batch
        
    # def _gen_pseudo_targets_flexmatch(self, gen_targets_new, gen_probs_being_new, gen_targets_old, gen_probs_being_old, targets_unk, token_attention_mask, class_thresholds_forward, class_thresholds_reverse):
    #     # Detach tensors #
    #     gen_targets_new = gen_targets_new.detach()
    #     gen_probs_being_new = gen_probs_being_new.detach()
    #     gen_targets_old = gen_targets_old.detach()
    #     gen_probs_being_old = gen_probs_being_old.detach()
    #     targets_unk = targets_unk.detach()
    #     token_attention_mask = token_attention_mask.detach()
    #     # Initialization #
    #     pseudo_targets_new, pseudo_targets_old, confidence_scores_forward, confidence_scores_reverse, pseudo_targets_new_mask, pseudo_targets_old_mask = [], [], [], [], [], []
    #     bsz, seq_sz = token_attention_mask.shape
    #     # Generate pseudo targets #
    #     for row in range(bsz):
    #         pseudo_targets_new.append(gen_targets_new[row])
    #         pseudo_targets_old.append(gen_targets_old[row])
    #         confidence_scores_forward_row = torch.softmax(gen_probs_being_new[row], dim=-1)[torch.arange(seq_sz), gen_targets_new[row]]
    #         confidence_scores_reverse_row = torch.softmax(gen_probs_being_old[row], dim=-1)[torch.arange(seq_sz), gen_targets_old[row]]
    #         confidence_scores_forward.append(confidence_scores_forward_row)
    #         confidence_scores_reverse.append(confidence_scores_reverse_row)
    #         pseudo_targets_new_mask.append(confidence_scores_forward_row >= class_thresholds_forward[gen_targets_new[row]])
    #         pseudo_targets_old_mask.append(confidence_scores_reverse_row >= class_thresholds_reverse[gen_targets_old[row]])
    #     # Convert lists to tensors #
    #     pseudo_targets_new = torch.stack(pseudo_targets_new).to(self.args.device)
    #     pseudo_targets_old = torch.stack(pseudo_targets_old).to(self.args.device)
    #     confidence_scores_forward = torch.stack(confidence_scores_forward).to(self.args.device)
    #     confidence_scores_reverse = torch.stack(confidence_scores_reverse).to(self.args.device)
    #     pseudo_targets_new_mask = torch.stack(pseudo_targets_new_mask).to(self.args.device)
    #     pseudo_targets_old_mask = torch.stack(pseudo_targets_old_mask).to(self.args.device)
    #     return pseudo_targets_new, pseudo_targets_old, confidence_scores_forward, confidence_scores_reverse, pseudo_targets_new_mask, pseudo_targets_old_mask
    
    # def _gen_sample_recon_errors(self, unlabeled_batch):
    #     # Extract unlabeled data #
    #     if self.args.use_prompt:
    #         weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_type_ids, weak_aug_token_attention_mask, weak_aug_targets_unk, \
    #             hvp_imgs_unlabeled, hvp_aux_imgs_unlabeled, mkg_imgs_unlabeled, mkg_aux_imgs_unlabeled, rcnn_imgs_unlabeled, \
    #             _, _, _, _, _, weak_aug_words, _ = unlabeled_batch
    #     else:
    #         hvp_imgs_unlabeled, hvp_aux_imgs_unlabeled, mkg_imgs_unlabeled, mkg_aux_imgs_unlabeled, rcnn_imgs_unlabeled = None, None, None, None, None
    #         weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_type_ids, weak_aug_token_attention_mask, weak_aug_targets_unk, weak_aug_words, _ = unlabeled_batch
    #     if self.model_dict["forward"].vt_model_name == "hvpnet":
    #         imgs_unlabeled = hvp_imgs_unlabeled
    #         aux_imgs_unlabeled = hvp_aux_imgs_unlabeled
    #     elif self.model_dict["forward"].vt_model_name == "mkgformer":
    #         imgs_unlabeled = mkg_imgs_unlabeled
    #         aux_imgs_unlabeled = mkg_aux_imgs_unlabeled
    #     else:
    #         imgs_unlabeled, aux_imgs_unlabeled = None, None
    #     weak_aug_words = list(map(list, zip(*weak_aug_words)))
    #     # Compute reconstruction errors and confidence scores #
    #     self.model_dict["forward"].eval()
    #     self.model_dict["reverse"].eval()
    #     with torch.no_grad():
    #         # Assume targets_unk being targets_old #
    #         _, fake_emissions_new_unlabeled = self._reverse_diffusion(weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #             imgs_unlabeled, aux_imgs_unlabeled, rcnn_imgs_unlabeled, dir="forward", targets=weak_aug_targets_unk)
    #         _, recon_emissions_old_unlabeled = self._reverse_diffusion(weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #             imgs_unlabeled, aux_imgs_unlabeled, rcnn_imgs_unlabeled, dir="reverse", targets=fake_emissions_new_unlabeled)
    #         errors_cycle_f2r = self.crit_id_per_sample(recon_emissions_old_unlabeled, weak_aug_targets_unk) # (bsz,)
    #         # Assume targets_unk being targets_new #
    #         _, fake_emissions_old_unlabeled = self._reverse_diffusion(weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #             imgs_unlabeled, aux_imgs_unlabeled, rcnn_imgs_unlabeled, dir="reverse", targets=weak_aug_targets_unk)
    #         _, recon_emissions_new_unlabeled = self._reverse_diffusion(weak_aug_char_input_ids, weak_aug_token_input_ids, weak_aug_token_attention_mask, weak_aug_token_type_ids, \
    #             imgs_unlabeled, aux_imgs_unlabeled, rcnn_imgs_unlabeled, dir="forward", targets=fake_emissions_old_unlabeled)
    #         errors_cycle_r2f = self.crit_id_per_sample(recon_emissions_new_unlabeled, weak_aug_targets_unk) # (bsz,)
    #         recon_errors = errors_cycle_f2r + errors_cycle_r2f
    #     self.model_dict["forward"].train()
    #     self.model_dict["reverse"].train()
    #     return recon_errors
        
    def _eval_labels(self, pbar, data, option="dev"): # option can be "dev" or "test"
        true_targets_new, gen_targets_new = [], []
        true_targets_old, gen_targets_old = [], []
        pbar.set_description_str(desc="Dev" if option == "eval" else "Testing")
        for batch in data:
            batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
            fake_targets_new, fake_targets_old, targets_new_batch, targets_old_batch, token_attention_mask = self._step_labeled(batch, "eval")
            true_targets_new_batch, gen_targets_new_batch = self._gen_labels(fake_targets_new, targets_new_batch, token_attention_mask)
            true_targets_old_batch, gen_targets_old_batch = self._gen_labels(fake_targets_old, targets_old_batch, token_attention_mask)
            true_targets_new += true_targets_new_batch
            gen_targets_new += gen_targets_new_batch
            true_targets_old += true_targets_old_batch
            gen_targets_old += gen_targets_old_batch
            pbar.update()
        pbar.close()

        precision_forward, recall_forward, f1_score_forward, precision_reverse, recall_reverse, f1_score_reverse = comp_f1_score(true_targets_new, true_targets_old, gen_targets_new, gen_targets_old)
        if option == "test":
            return precision_forward, recall_forward, f1_score_forward, precision_reverse, recall_reverse, f1_score_reverse
        else:
            return f1_score_forward, f1_score_reverse
    
    def training_settings_text_only(self):
        self.optimizer = optim.AdamW(list(self.model_dict["forward"].parameters() + self.model_dict["reverse"].parameters()))
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
                                                                num_training_steps=self.train_num_steps)
        self.model_dict["forward"].to(self.args.device)
        self.model_dict["reverse"].to(self.args.device)

    def training_settings_with_prompt(self):
        parameters_forward = self._get_model_parameters("forward")
        parameters_reverse = self._get_model_parameters("reverse")
        self.optimizer = optim.AdamW(list(parameters_forward + parameters_reverse))
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
                                                                num_training_steps=self.train_num_steps)
        self.model_dict["forward"].to(self.args.device)
        self.model_dict["reverse"].to(self.args.device)

    def _get_model_parameters(self, dir="forward"):
        # bert lr
        parameters = []
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model_dict[dir].named_parameters():
            if 'bert' in name or 'text' in name:
                params['params'].append(param)
        parameters.append(params)

        # prompt lr / vit lr
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model_dict[dir].named_parameters():
            if 'encoder_conv' in name or 'gates' in name or 'vision' in name:
                params['params'].append(param)
        parameters.append(params)

        # crf/fc/mlp lr
        params = {'lr':5e-2, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model_dict[dir].named_parameters():
            if 'crf' in name or name.startswith('fc') or name.endswith('_mlp') or 'label_layer_norm' in name:
                params['params'].append(param)
        parameters.append(params)

        # attention lr
        params = {'lr':5e-6, 'weight_decay':5e-8}
        params['params'] = []
        for name, param in self.model_dict[dir].named_parameters():
            if name.endswith('_attn'):
                params['params'].append(param)
        parameters.append(params)

        for name, par in self.model_dict[dir].named_parameters():
            # freeze resnet #
            if 'image_model' in name:   par.requires_grad = False
            # freeze char lstm #
            if 'char_lstm' in name: par.requires_grad = False

        return parameters