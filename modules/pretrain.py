import torch
from torch import optim
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from models.modeling_clip import CLIPModel
from transformers import BertModel, AutoModel

class BaseTrainer(object):
    def __init__(self, label_map=None, args=None, logger=None, writer=None) -> None:
        self.label_map = label_map
        self.args = args
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.step = 0

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()
    
    def _gen_labels(self, logits, targets, token_attention_mask, words=None):
        if isinstance(logits, torch.Tensor):
            #logits = logits.argmax(-1).detach()
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
                if column == 0:
                    continue
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
    
class PreTrainer(BaseTrainer):
    def __init__(self, train_data=None, val_data=None, test_data=None, ner_model=None, ner_model_name=None, label_map=None, args=None, logger=None, writer=None) -> None:
        super().__init__(label_map, args, logger, writer)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        # self.label_map = label_map
        # self.args = args
        # self.logger = logger
        # self.writer = writer
        # self.refresh_step = 2
        self.best_train_metric = 0
        self.best_train_epoch = None
        self.best_dev_metric = 0
        self.best_dev_epoch = None
        self.model = ner_model
        self.ner_model_name = ner_model_name
        self.optimizer = None
        self.scheduler = None
        self.train_num_steps = len(self.train_data) * args.num_epochs
        # self.step = 0

        if self.model.__class__.__name__ == "UnimoCRFModel":
            clip_model = CLIPModel.from_pretrained("/home/yixin/workspace/huggingface/" + self.args.vit_name)
            clip_vit = clip_model.vision_model
            bert = BertModel.from_pretrained("/home/yixin/workspace/huggingface/" + self.args.lm_name)
            clip_model_dict = clip_vit.state_dict()
            bert_model_dict = bert.state_dict()

            vision_names, text_names = [], []
            model_dict = self.model.state_dict()
            for name in model_dict:
                if 'vision' in name:
                    clip_name = name.replace('vision_', '').replace('model.', '')
                    if clip_name in clip_model_dict:
                        vision_names.append(clip_name)
                        model_dict[name] = clip_model_dict[clip_name]
                elif 'text' in name:
                    text_name = name.replace('text_', '').replace('model.', '')
                    if text_name in bert_model_dict:
                        text_names.append(text_name)
                        model_dict[name] = bert_model_dict[text_name]
            assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                        (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
            self.model.load_state_dict(model_dict)

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
            #self.pbar = pbar
            avg_loss = 0
            for epoch in range(self.args.num_epochs):
                true_labels, pred_labels = [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch+1, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    token_attention_mask, targets_old, loss, logits, probs = self._step(batch, "train")
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    true_label_batch, pred_label_batch = self._gen_labels(logits, targets_old, token_attention_mask)
                    true_labels += true_label_batch
                    pred_labels += pred_label_batch

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str("loss:{:<6.5f}".format(avg_loss))
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                results = classification_report(true_labels, pred_labels, digits=4) 
                self.logger.info("***** Train Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                if self.writer:
                    self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch, f1_score))
                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch
                
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.
                
            torch.cuda.empty_cache()
            
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
    
    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instance = %d", len(self.val_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.val_data), leave=False, dynamic_ncols=True) as pbar:
                f1_score = self._eval_labels(pbar, self.val_data, "dev")
                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}."
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, f1_score))
                if f1_score >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()
    
    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running test *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path + ".pth"))
            self.model.to(self.args.device)
            self.logger.info("Load model successful!")

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                f1_score = self._eval_labels(pbar, self.test_data, "test")
                self.logger.info("Test f1 score: {}.".format(f1_score))
                if self.ner_model_name == "hvpnet":
                    torch.save(self.model.core.state_dict(), self.args.save_path+"/hvp_core_model.pth")
                else:
                    torch.save(self.model.model.state_dict(), self.args.save_path+"/mkg_core_model.pth")

        self.model.train()

    def _eval_labels(self, pbar, data, option="dev"): # option can be "dev" or "test"
        true_labels, pred_labels = [], []
        pbar.set_description_str(desc="Dev" if option == "dev" else "Testing")
        for batch in data:
            batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
            token_attention_mask, targets_old, loss, logits, probs = self._step(batch, "dev")
            true_label_batch, pred_label_batch = self._gen_labels(logits, targets_old, token_attention_mask)
            true_labels += true_label_batch
            pred_labels += pred_label_batch
            pbar.update()
        pbar.close()

        results = classification_report(true_labels, pred_labels, digits=4) 
        self.logger.info("***** {} Eval results *****".format("Dev" if option == "dev" else "Test"))
        self.logger.info("\n%s", results)
        f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])
        if self.writer:
            self.writer.add_scalar(tag="{}_f1".format(option), scalar_value=f1_score)    # tensorbordx

        return f1_score
    
    def _step(self, batch, mode="train"):
        if self.args.use_prompt:
            char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_old, hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs, words, _ = batch
        else:
            hvp_imgs, hvp_aux_imgs, mkg_imgs, mkg_aux_imgs, rcnn_imgs = None, None, None, None, None, None
            char_input_ids, token_input_ids, token_type_ids, token_attention_mask, targets_old, words, _ = batch

        words = list(map(list, zip(*words)))
        if self.model.__class__.__name__ == "HMNeTNERModel": # For hvpnet
            loss, logits, probs = self.model(input_ids=token_input_ids, attention_mask=token_attention_mask, token_type_ids=token_type_ids, labels=targets_old, images=hvp_imgs, aux_imgs=hvp_aux_imgs)
        else: # For mkgformer
            loss, logits, probs = self.model(input_ids=token_input_ids, attention_mask=token_attention_mask, token_type_ids=token_type_ids, labels=targets_old, images=mkg_imgs, aux_imgs=mkg_aux_imgs, rcnn_imgs=rcnn_imgs)
        
        if mode == "test":
            return token_attention_mask, targets_old, loss, logits, probs, words
        else:
            return token_attention_mask, targets_old, loss, logits, probs
    
    def training_settings_text_only(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                    num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
                                                                    num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def training_settings_with_prompt(self):
        # bert lr
        parameters = []
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name or 'text' in name:
                params['params'].append(param)
        parameters.append(params)

        # prompt lr / vit lr
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name or 'vision' in name:
                params['params'].append(param)
        parameters.append(params)

        # crf lr
        params = {'lr':5e-2, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters)

        for name, par in self.model.named_parameters(): # freeze resnet
            if 'image_model' in name:   par.requires_grad = False
        
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                    num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
                                                                    num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)