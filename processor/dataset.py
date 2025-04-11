import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, CLIPProcessor, AutoTokenizer
from utils.utils import remove_excluded_words, strong_augment_pil_image
import logging
logger = logging.getLogger(__name__)

excluded_words = ["RT", "&amp", "&gt", "--&gt"]

class LEDProcessor(object):
    def __init__(self, data_path, clstm_path, args):
        self.data_path = data_path
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.args.local_cache_path, self.args.lm_name), do_lower_case=True)
        self.char2int, self.int2char = torch.load(os.path.join(clstm_path, "char_vocab.pkl"))
        clip_processor = CLIPProcessor.from_pretrained(os.path.join(self.args.local_cache_path, self.args.vit_name))
        aux_processor = CLIPProcessor.from_pretrained(os.path.join(self.args.local_cache_path, self.args.vit_name))
        aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = self.args.aux_size, self.args.aux_size
        rcnn_processor = CLIPProcessor.from_pretrained(os.path.join(self.args.local_cache_path, self.args.vit_name))
        rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = self.args.rcnn_size, self.args.rcnn_size
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    def load_from_file(self, mode="supervsed"):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "supervised".
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # Make sure that the last line of the loading file is '\n' #
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, targets_old, targets_new = [], [], []
            word, target_old, target_new = [], [], []
            img_names = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_name = line.strip().split("IMGID:")[1] + ".jpg"
                    img_names.append(img_name)
                elif line != "\n":
                    tokens = line.strip().split("\t")
                    word.append(tokens[0])
                    target_old.append(tokens[1])
                    if mode == "supervised":
                        target_new.append(tokens[-1])
                else:
                    words.append(word)
                    targets_old.append(target_old)
                    word, target_old = [], []
                    if mode == "supervised":
                        targets_new.append(target_new)
                        target_new = []

        assert len(words) == len(targets_old) == len(img_names), "{}, {}, {}".format(
            len(words), len(targets_old), len(img_names))

        aux_img_dict_path = self.data_path["auximgs"]
        aux_img_dict = torch.load(aux_img_dict_path)
        rcnn_img_dict = torch.load(self.data_path["img2crop"])

        return {"words": words, "targets_old": targets_old, "targets_new": targets_new, "img_names": img_names, "aux_img_dict": aux_img_dict, "rcnn_img_dict": rcnn_img_dict}

    def get_label_mapping(self):
        LABELS = ["PAD", "O", "B-MISC", "I-MISC", "B-PER", "I-PER",
                  "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABELS)}
        return label_mapping


class LEDDataset(Dataset):
    def __init__(self, processor, transform, imgs_path=None, aux_imgs_path=None, max_seq_len=128, max_char_len=128, mode="supervised", ignore_idx=0, aux_size=128, rcnn_imgs_path="data/NER_data", rcnn_size=128):
        super().__init__()
        self.transform = transform
        self.processor = processor
        self.imgs_path = imgs_path
        self.aux_imgs_path = aux_imgs_path
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.mode = mode
        self.ignore_idx = ignore_idx
        self.aux_size = aux_size
        self.rcnn_imgs_path = rcnn_imgs_path
        self.rcnn_size = rcnn_size
        if self.mode == "subset":
            self.data_dict = None
        else:
            self.data_dict = processor.load_from_file(mode)

    def insert_data(self, words, img_names, targets_old, targets_new):
        assert len(words) == len(targets_old) == len(targets_new) == len(img_names)
        self.data_dict["words"] += words
        self.data_dict["targets_old"] += targets_old
        self.data_dict["targets_new"] += targets_new
        img_names = [name + ".jpg" for name in img_names]
        self.data_dict["img_names"] += img_names

    def delete_data(self, img_names):
        for name in img_names:
            if not name.endswith(".jpg"):
                idx = self.data_dict["img_names"].index(name+".jpg")
            else:
                idx = self.data_dict["img_names"].index(name)
            self.data_dict["words"].pop(idx)
            self.data_dict["targets_old"].pop(idx)
            if self.data_dict["targets_new"]:
                self.data_dict["targets_new"].pop(idx)
            self.data_dict["img_names"].pop(idx)

    def from_indices(self, indices):
        new_dataset = LEDDataset(processor=self.processor, transform=self.transform, imgs_path=self.imgs_path, aux_imgs_path=self.aux_imgs_path, max_seq_len=self.max_seq_len, \
                                 max_char_len=self.max_char_len, mode="subset", ignore_idx=self.ignore_idx, aux_size=self.aux_size, rcnn_imgs_path=self.rcnn_imgs_path, rcnn_size=self.rcnn_size)
        new_data_dict = {}
        new_data_dict["aux_img_dict"] = self.data_dict["aux_img_dict"]
        new_data_dict["rcnn_img_dict"] = self.data_dict["rcnn_img_dict"]
        new_data_dict["words"] = [self.data_dict["words"][i] for i in indices]
        new_data_dict["img_names"] = [self.data_dict["img_names"][i] for i in indices]
        new_data_dict["targets_old"] = [self.data_dict["targets_old"][i] for i in indices]
        if self.mode == "supervised":
            new_data_dict["targets_new"] = [self.data_dict["targets_new"][i] for i in indices]
        else:
            new_data_dict["targets_new"] = []
        new_dataset.data_dict = new_data_dict
        new_dataset.mode = self.mode
        return new_dataset

    def __len__(self):
        return len(self.data_dict["words"])

    def __getitem__(self, idx):
        if self.mode == "supervised":
            word_list, target_old_list, target_new_list, img_name = self.data_dict["words"][idx], self.data_dict["targets_old"][idx], self.data_dict["targets_new"][idx], self.data_dict["img_names"][idx]
            # Process sentences, chars, and labels #
            token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_old, targets_new, words = self._seq_proc(word_list, target_old_list=target_old_list, target_new_list=target_new_list)
            # Process images #
            hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs = self._img_proc(img_name, aug_option=False)
            # Return tensors #
            if self.imgs_path is not None and self.aux_imgs_path is not None and self.rcnn_imgs_path is not None:
                return torch.tensor(char_input_ids), torch.tensor(token_input_ids), torch.tensor(token_type_ids), torch.tensor(token_attention_mask), torch.tensor(targets_old), torch.tensor(targets_new), \
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words, img_name
            elif self.imgs_path is not None and self.aux_imgs_path is not None:
                return torch.tensor(char_input_ids), torch.tensor(token_input_ids), torch.tensor(token_type_ids), torch.tensor(token_attention_mask), torch.tensor(targets_old), torch.tensor(targets_new), \
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, words, img_name
            else:
                return torch.tensor(char_input_ids), torch.tensor(token_input_ids), torch.tensor(token_type_ids), torch.tensor(token_attention_mask), torch.tensor(targets_old), torch.tensor(targets_new), words, img_name
        elif self.mode == "predict" or self.mode == "mean_teacher":
            word_list, target_unk_list, img_name = self.data_dict["words"][idx], self.data_dict["targets_old"][idx], self.data_dict["img_names"][idx]
            # Process sentences, chars, and labels #
            token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_unk, words = self._seq_proc(word_list, target_old_list=target_unk_list)
            # Process images #
            hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs = self._img_proc(img_name, aug_option=False)
            # Return tensors #
            if self.imgs_path is not None and self.aux_imgs_path is not None and self.rcnn_imgs_path is not None:
                return torch.tensor(char_input_ids), torch.tensor(token_input_ids), torch.tensor(token_type_ids), torch.tensor(token_attention_mask), torch.tensor(targets_unk),  \
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words, img_name
            elif self.imgs_path is not None and self.aux_imgs_path is not None:
                return torch.tensor(char_input_ids), torch.tensor(token_input_ids), torch.tensor(token_type_ids), torch.tensor(token_attention_mask), torch.tensor(targets_unk), \
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, words, img_name
            else:
                return torch.tensor(char_input_ids), torch.tensor(token_input_ids), torch.tensor(token_type_ids), torch.tensor(token_attention_mask), torch.tensor(targets_unk), words, img_name
        elif self.mode == "flexmatch":
            word_list, target_unk_list, img_name = self.data_dict["words"][idx], self.data_dict["targets_old"][idx], self.data_dict["img_names"][idx]
            # Process sentences, chars, and labels #
            aug_word_list, aug_target_unk_list = remove_excluded_words(word_list, target_unk_list, excluded_words)
            aug_token_input_ids, aug_token_type_ids, aug_token_attention_mask, aug_char_input_ids, aug_targets_unk, aug_words = self._seq_proc(aug_word_list, target_old_list=aug_target_unk_list)
            # Process images #
            hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, aug_hvp_img, aug_hvp_aux_imgs, aug_mkg_img, aug_mkg_aux_imgs, aug_rcnn_imgs = self._img_proc(img_name, aug_option=True)
            if self.imgs_path is not None and self.aux_imgs_path is not None and self.rcnn_imgs_path is not None:
                return torch.tensor(aug_char_input_ids), torch.tensor(aug_token_input_ids), torch.tensor(aug_token_type_ids), torch.tensor(aug_token_attention_mask), torch.tensor(aug_targets_unk), \
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, \
                    aug_hvp_img, aug_hvp_aux_imgs, aug_mkg_img, aug_mkg_aux_imgs, aug_rcnn_imgs, aug_words, img_name
            elif self.imgs_path is not None and self.aux_imgs_path is not None:
                return torch.tensor(aug_char_input_ids), torch.tensor(aug_token_input_ids), torch.tensor(aug_token_type_ids), torch.tensor(aug_token_attention_mask), torch.tensor(aug_targets_unk), \
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, \
                    aug_hvp_img, aug_hvp_aux_imgs, aug_mkg_img, aug_mkg_aux_imgs, aug_words, img_name
            else:
                return torch.tensor(aug_char_input_ids), torch.tensor(aug_token_input_ids), torch.tensor(aug_token_type_ids), torch.tensor(aug_token_attention_mask), torch.tensor(aug_targets_unk), \
                    aug_words, img_name
            
    def _seq_proc(self, word_list, target_old_list=None, target_new_list=None):
        tokens, char_input_ids, targets_old, targets_new, words = [], [], [], [], []
        # Tokenize words #
        for i, word in enumerate(word_list):
            token = self.processor.tokenizer.tokenize(word)
            tokens.extend(token)
            for t in token:
                if t in self.processor.char2int:
                    char_input_ids.append([self.processor.char2int[t]] + [0]*(self.max_char_len-1))
                else:
                    char_input_ids.append([self.processor.char2int.get(c,0) for c in t] + [0]*(self.max_char_len-len(t)))
            if target_old_list is not None:
                target_old = target_old_list[i]
                for m in range(len(token)):
                    if m == 0:
                        targets_old.append(self.processor.get_label_mapping()[target_old])
                    else:
                        targets_old.append(self.processor.get_label_mapping()["X"])
                    words.append(word)
            if target_new_list is not None:
                target_new = target_new_list[i]
                for m in range(len(token)):
                    if m == 0:
                        targets_new.append(self.processor.get_label_mapping()[target_new])
                    else:
                        targets_new.append(self.processor.get_label_mapping()["X"])
        # Trucate sequences if needed #
        if len(tokens) >= self.max_seq_len - 1:
            tokens = tokens[:self.max_seq_len-2]
            char_input_ids = char_input_ids[:self.max_seq_len-2]
            words = words[:self.max_seq_len-2]
            if targets_old:
                targets_old = targets_old[:self.max_seq_len-2]
            if targets_new:
                targets_new = targets_new[:self.max_seq_len-2]
        # Digitalization #
        token_encode_dict = self.processor.tokenizer.encode_plus(tokens, max_length=self.max_seq_len, truncation=True, padding="max_length")
        token_input_ids, token_type_ids, token_attention_mask = token_encode_dict["input_ids"], token_encode_dict["token_type_ids"], token_encode_dict["attention_mask"]
        char_input_ids = [[self.processor.char2int["[CLS]"]] + [0]*(self.max_char_len-1)] + char_input_ids + [[self.processor.char2int["[SEP]"]] + [0]*(self.max_char_len-1)] \
            + [[self.processor.char2int["[PAD]"]]*self.max_char_len] * (self.max_seq_len-len(char_input_ids)-2)
        if targets_old:
            targets_old = [self.processor.get_label_mapping()["[CLS]"]] + targets_old + [self.processor.get_label_mapping()["[SEP]"]] + [self.ignore_idx] * (self.max_seq_len-len(targets_old)-2)
        if targets_new:
            targets_new = [self.processor.get_label_mapping()["[CLS]"]] + targets_new + [self.processor.get_label_mapping()["[SEP]"]] + [self.ignore_idx]*(self.max_seq_len-len(targets_new)-2)
        words = ["[CLS]"] + words + ["[SEP]"] + ["[PAD]"]*(self.max_seq_len-len(words)-2)

        if targets_new:
            return token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_old, targets_new, words
        else:
            return token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_old, words
        
    def _img_proc(self, img_name, aug_option=False):
        if self.imgs_path is not None:
            try:
                img_path = os.path.join(self.imgs_path, img_name)
                image = Image.open(img_path).convert("RGB")
            except:
                img_path = os.path.join(self.imgs_path, "inf.png")
                image = Image.open(img_path).convert("RGB")
            # Original #
            hvp_img = self.transform(image)
            mkg_img = self.processor.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            # Strong Augmented Images #
            if aug_option == True:
                aug_image = strong_augment_pil_image(image)
                aug_hvp_img = self.transform(aug_image)
                aug_mkg_img = self.processor.clip_processor(images=aug_image, return_tensors='pt')['pixel_values'].squeeze()
        if self.aux_imgs_path is not None:
            hvp_aux_imgs, mkg_aux_imgs, aux_img_paths = [], [], []
            if aug_option == True:
                aug_hvp_aux_imgs, aug_mkg_aux_imgs,  = [], []
            
            if img_name in self.data_dict["aux_img_dict"]:
                aux_img_paths = self.data_dict["aux_img_dict"][img_name]
                aux_img_paths = [os.path.join(self.aux_imgs_path, path) for path in aux_img_paths]
            for i in range(min(3, len(aux_img_paths))):
                aux_img = Image.open(aux_img_paths[i]).convert("RGB")
                # Original #
                hvp_aux_img = self.transform(aux_img)
                mkg_aux_img = self.processor.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                hvp_aux_imgs.append(hvp_aux_img)
                mkg_aux_imgs.append(mkg_aux_img)
                # Strong Augmented Images #
                if aug_option == True:
                    aug_aux_img = strong_augment_pil_image(aux_img)
                    aug_hvp_aux_img = self.transform(aug_aux_img)
                    aug_mkg_aux_img = self.processor.aux_processor(images=aug_aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aug_hvp_aux_imgs.append(aug_hvp_aux_img)
                    aug_mkg_aux_imgs.append(aug_mkg_aux_img)
            for i in range(3-len(aux_img_paths)):
                # Original #
                hvp_aux_imgs.append(torch.zeros(3, 224, 224))
                mkg_aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))
                # Strong Augmented Images #
                if aug_option == True:
                    aug_hvp_aux_imgs.append(torch.zeros(3, 224, 224))
                    aug_mkg_aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))
            # Original #
            hvp_aux_imgs = torch.stack(hvp_aux_imgs, dim=0)
            mkg_aux_imgs = torch.stack(mkg_aux_imgs, dim=0)
            assert len(hvp_aux_imgs) == 3
            assert len(mkg_aux_imgs) == 3
            # Strong Augmented Images #
            if aug_option == True:
                aug_hvp_aux_imgs = torch.stack(aug_hvp_aux_imgs, dim=0)
                aug_mkg_aux_imgs = torch.stack(aug_mkg_aux_imgs, dim=0)
                assert len(aug_hvp_aux_imgs) == 3
                assert len(aug_mkg_aux_imgs) == 3
        if self.rcnn_imgs_path is not None:
            rcnn_imgs, rcnn_img_paths = [], []
            if aug_option == True:
                aug_rcnn_imgs = []
            img_name = img_name.split('.')[0]
            if img_name in self.data_dict["rcnn_img_dict"]:
                rcnn_img_paths = self.data_dict["rcnn_img_dict"][img_name]
                rcnn_img_paths = [os.path.join(self.rcnn_imgs_path, path) for path in rcnn_img_paths]
            for i in range(min(3, len(rcnn_img_paths))):
                rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                if aug_option == True:
                    aug_rcnn_img = strong_augment_pil_image(rcnn_img)
                # Original #
                rcnn_img = self.processor.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                rcnn_imgs.append(rcnn_img)
                # Strong Augmented Images #
                if aug_option == True:
                    aug_rcnn_img = self.processor.rcnn_processor(images=aug_rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                    aug_rcnn_imgs.append(aug_rcnn_img)
            # Original #
            for i in range(3-len(rcnn_imgs)):
                rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))
            rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
            assert len(rcnn_imgs) == 3
            # Strong Augmented Images #
            if aug_option == True:
                for i in range(3-len(aug_rcnn_imgs)):
                    aug_rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size))) 
                aug_rcnn_imgs = torch.stack(aug_rcnn_imgs, dim=0)
                assert len(aug_rcnn_imgs) == 3
        
        if aug_option == True:
            return hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, aug_hvp_img, aug_hvp_aux_imgs, aug_mkg_img, aug_mkg_aux_imgs, aug_rcnn_imgs
        else:
            return hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs