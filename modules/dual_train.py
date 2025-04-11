import torch, os
from torch import optim
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from .pretrain import BaseTrainer
from .metrics import comp_f1_score

class DualNERTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, ner_base_model="mkgformer", ner_model_old=None, ner_model_new=None, label_map=None, args=None, logger=None, writer=None) -> None:
        super().__init__(label_map, args, logger, writer)
        # Data #
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_num_steps = len(self.train_data) * args.num_epochs
        # Metrics #
        self.best_train_metric_old = 0
        self.best_train_epoch_old = None
        self.best_dev_metric_old = 0
        self.best_dev_epoch_old = None
        self.best_train_metric_new = 0
        self.best_train_epoch_new = None
        self.best_dev_metric_new = 0
        self.best_dev_epoch_new = None
        # Data Structures for Training #
        self.ner_base_model = ner_base_model
        self.model_dict = {"old": ner_model_old, "new": ner_model_new}
        self.optimizer_dict = {"old": None, "new": None}
        self.scheduler_dict = {"old": None, "new": None}

    def train(self):
        if self.args.use_prompt:
            self.training_settings_with_prompt("old")
            self.training_settings_with_prompt("new")
        else:
            self.training_settings_text_only("old")
            self.training_settings_text_only("new")

        self.model_dict["old"].train()
        self.model_dict["new"].train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        self.step = 0
        with tqdm(total=self.train_num_steps, postfix="loss:{0:<6.5f}", leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            avg_loss = 0
            for epoch in range(self.args.num_epochs):
                true_labels_new, pred_labels_new, true_labels_old, pred_labels_old = [], [], [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch+1, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    token_attention_mask, targets_new, targets_old, loss_new, loss_old, logits_new, logits_old, probs_new, probs_old = self._step(batch, mode="train")
                    # diff_loss = mse_loss(probs_new, probs_old)
                    # loss = loss_new + loss_old - diff_loss
                    loss = loss_new + loss_old
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer_dict["new"].step()
                    self.scheduler_dict["new"].step()
                    self.optimizer_dict["new"].zero_grad()
                    self.optimizer_dict["old"].step()
                    self.scheduler_dict["old"].step()
                    self.optimizer_dict["old"].zero_grad()

                    true_label_batch_old, pred_label_batch_old = self._gen_labels(logits_old, targets_old, token_attention_mask)
                    true_labels_old += true_label_batch_old
                    pred_labels_old += pred_label_batch_old
                    true_label_batch_new, pred_label_batch_new = self._gen_labels(logits_new, targets_new, token_attention_mask)
                    true_labels_new += true_label_batch_new
                    pred_labels_new += pred_label_batch_new

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str("loss: {:<6.5f}".format(avg_loss))
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                # results_new = classification_report(true_labels_new, pred_labels_new, digits=4) 
                # self.logger.info("***** Train Eval results New *****")
                # self.logger.info("\n%s", results_new)
                # f1_score_new = float(results_new.split('\n')[-4].split('      ')[0].split('    ')[3])
                # if self.writer:
                #     self.writer.add_scalar(tag='train_f1_new', scalar_value=f1_score_new, global_step=epoch)    # tensorbordx
                # self.logger.info("Epoch {}/{}, best train f1 new: {}, best epoch new: {}, current train f1 score new: {}."\
                #             .format(epoch, self.args.num_epochs, self.best_train_metric_new, self.best_train_epoch_new, f1_score_new))
                
                
                # results_old = classification_report(true_labels_old, pred_labels_old, digits=4) 
                # self.logger.info("***** Train Eval results Old *****")
                # self.logger.info("\n%s", results_old)
                # f1_score_old = float(results_old.split('\n')[-4].split('      ')[0].split('    ')[3])
                # if self.writer:
                #     self.writer.add_scalar(tag='train_f1_old', scalar_value=f1_score_old, global_step=epoch)    # tensorbordx
                # self.logger.info("Epoch {}/{}, best train f1 old: {}, best epoch old: {}, current train f1 score old: {}."\
                #             .format(epoch, self.args.num_epochs, self.best_train_metric_old, self.best_train_epoch_old, f1_score_old))

                precision_new, recall_new, f1_score_new, precision_old, recall_old, f1_score_old = comp_f1_score(true_labels_new, true_labels_old, pred_labels_new, pred_labels_old)
                if f1_score_new > self.best_train_metric_new:
                    self.best_train_metric_new = f1_score_new
                    self.best_train_epoch_new = epoch

                if f1_score_old > self.best_train_metric_old:
                    self.best_train_metric_old = f1_score_old
                    self.best_train_epoch_old = epoch
                
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.
            
            torch.cuda.empty_cache()
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance for new model at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_new, self.best_dev_metric_new))
            self.logger.info("Get best dev performance for old model at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch_old, self.best_dev_metric_old))

    def evaluate(self, epoch):
        self.model_dict["new"].eval()
        self.model_dict["old"].eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instance = %d", len(self.val_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                f1_score_new, f1_score_old = self._eval_labels(pbar, self.val_data, "dev")
                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}."
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric_new, self.best_dev_epoch_new, f1_score_new))
                
                if f1_score_new >= self.best_dev_metric_new:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch_new = epoch
                    self.best_dev_metric_new = f1_score_new # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model_dict["new"].state_dict(), self.args.save_path+"/best_model_new.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}."
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric_old, self.best_dev_epoch_old, f1_score_old))
                if f1_score_old >= self.best_dev_metric_old:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch_old = epoch
                    self.best_dev_metric_old = f1_score_old # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model_dict["old"].state_dict(), self.args.save_path+"/best_model_old.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model_dict["new"].train()
        self.model_dict["old"].train()

    def test(self):
        self.model_dict["new"].eval()
        self.model_dict["old"].eval()
        self.logger.info("\n***** Running test *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model_dict["new"].load_state_dict(torch.load(self.args.load_path + "_new.pth"))
            self.model_dict["new"].to(self.args.device)
            self.model_dict["old"].load_state_dict(torch.load(self.args.load_path + "_old.pth"))
            self.model_dict["old"].to(self.args.device)
            self.logger.info("Load model successful!")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                f1_score_new, f1_score_old = self._eval_labels(pbar, self.test_data, "test")
                self.logger.info("Test f1 score of the new and old models: {}, {}.".format(f1_score_new, f1_score_old))
                # if self.ner_base_model == "hvpnet":
                #     torch.save(self.model_dict["new"].core.state_dict(), self.args.save_path+"/hvp_core_model_new.pth")
                #     torch.save(self.model_dict["old"].core.state_dict(), self.args.save_path+"/hvp_core_model_old.pth")
                # else:
                #     torch.save(self.model_dict["new"].model.state_dict(), self.args.save_path+"/mkg_core_model_new.pth")
                #     torch.save(self.model_dict["old"].model.state_dict(), self.args.save_path+"/mkg_core_model_old.pth")

        self.model_dict["new"].train()
        self.model_dict["old"].train()

    def _eval_labels(self, pbar, data, option="dev"): # option can be "dev" or "test"
        true_labels_new, pred_labels_new, true_labels_old, pred_labels_old = [], [], [], []
        pbar.set_description_str(desc="Dev" if option == "dev" else "Testing")
        for batch in data:
            batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
            token_attention_mask, targets_new, targets_old, loss_new, loss_old, logits_new, logits_old, probs_new, probs_old = self._step(batch, mode="dev")
            true_label_batch_new, pred_label_batch_new = self._gen_labels(logits_new, targets_new, token_attention_mask)
            true_labels_new += true_label_batch_new
            pred_labels_new += pred_label_batch_new
            true_label_batch_old, pred_label_batch_old = self._gen_labels(logits_old, targets_old, token_attention_mask)
            true_labels_old += true_label_batch_old
            pred_labels_old += pred_label_batch_old
            pbar.update()
        pbar.close()

        # results_new = classification_report(true_labels_new, pred_labels_new, digits=4) 
        # self.logger.info("***** {} Eval results New *****".format("Dev" if option == "dev" else "Test"))
        # self.logger.info("\n%s", results_new)
        # f1_score_new = float(results_new.split('\n')[-4].split('      ')[-2].split('    ')[-1])
        # if self.writer:
        #     self.writer.add_scalar(tag="{}_f1".format(option), scalar_value=f1_score_new)    # tensorbordx

        # results_old = classification_report(true_labels_old, pred_labels_old, digits=4) 
        # self.logger.info("***** {} Eval results Old *****".format("Dev" if option == "dev" else "Test"))
        # self.logger.info("\n%s", results_old)
        # f1_score_old = float(results_old.split('\n')[-4].split('      ')[-2].split('    ')[-1])
        # if self.writer:
        #     self.writer.add_scalar(tag="{}_f1".format(option), scalar_value=f1_score_old)    # tensorbordx

        precision_new, recall_new, f1_score_new, precision_old, recall_old, f1_score_old = comp_f1_score(true_labels_new, true_labels_old, pred_labels_new, pred_labels_old)

        return f1_score_new, f1_score_old

    def _step(self, batch, mode="train"):
        if self.args.use_prompt:
            _, token_input_ids, token_type_ids, token_attention_mask, targets_old, targets_new, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, _ = batch
        else:
            _, token_input_ids, token_type_ids, token_attention_mask, targets_old, targets_new, words, _ = batch
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None
        words = list(map(list, zip(*words)))

        if self.ner_base_model == "hvpnet":
            loss_new, logits_new, probs_new = self.model_dict["new"](input_ids=token_input_ids, attention_mask=token_attention_mask, token_type_ids=token_type_ids, labels=targets_new, images=hvp_imgs, aux_imgs=hvp_aux_imgs)
            loss_old, logits_old, probs_old = self.model_dict["old"](input_ids=token_input_ids, attention_mask=token_attention_mask, token_type_ids=token_type_ids, labels=targets_old, images=hvp_imgs, aux_imgs=hvp_aux_imgs)
        else:
            loss_new, logits_new, probs_new = self.model_dict["new"](input_ids=token_input_ids, attention_mask=token_attention_mask, token_type_ids=token_type_ids, labels=targets_new, images=mkg_imgs, aux_imgs=mkg_aux_imgs, rcnn_imgs=rcnn_imgs)
            loss_old, logits_old, probs_old = self.model_dict["old"](input_ids=token_input_ids, attention_mask=token_attention_mask, token_type_ids=token_type_ids, labels=targets_old, images=mkg_imgs, aux_imgs=mkg_aux_imgs, rcnn_imgs=rcnn_imgs)

        if mode == "test":
            return token_attention_mask, targets_new, targets_old, loss_new, loss_old, logits_new, logits_old, probs_new, probs_old, words
        else:
            return token_attention_mask, targets_new, targets_old, loss_new, loss_old, logits_new, logits_old, probs_new, probs_old

    def training_settings_text_only(self, type="old"):
        self.optimizer_dict[type] = optim.AdamW(self.model_dict[type].parameters(), lr=self.args.lr)
        self.scheduler_dict[type] = get_linear_schedule_with_warmup(optimizer=self.optimizer_dict[type],
                                                                    num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
                                                                    num_training_steps=self.train_num_steps)
        self.model_dict[type].to(self.args.device)

    def training_settings_with_prompt(self, type="old"):
        # bert lr
        parameters = []
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model_dict[type].named_parameters():
            if 'bert' in name or 'text' in name:
                params['params'].append(param)
        parameters.append(params)

        # prompt lr / vit lr
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model_dict[type].named_parameters():
            if 'encoder_conv' in name or 'gates' in name or 'vision' in name:
                params['params'].append(param)
        parameters.append(params)

        # crf lr
        params = {'lr':5e-2, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model_dict[type].named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        self.optimizer_dict[type] = optim.AdamW(parameters)

        for name, par in self.model_dict[type].named_parameters(): # freeze resnet
            if 'image_model' in name:   par.requires_grad = False
        
        self.scheduler_dict[type] = get_linear_schedule_with_warmup(optimizer=self.optimizer_dict[type],
                                                                    num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
                                                                    num_training_steps=self.train_num_steps)
        self.model_dict[type].to(self.args.device)