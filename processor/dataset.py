import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, CLIPProcessor, BertModel
from utils.utils import remove_excluded_words
import logging

logger = logging.getLogger(__name__)

class LEDProcessor:
    def __init__(self, data_path, clstm_path, args):
        """Initialize processor for LED dataset with BERT and CLIP models."""
        self.data_path = data_path
        self.args = args
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.lm_name), do_lower_case=True
        )
        self.bert = BertModel.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.lm_name)
        )
        # Load character vocabulary
        try:
            self.char2int, self.int2char = torch.load(os.path.join(clstm_path, "char_vocab.pkl"))
        except FileNotFoundError:
            logger.error(f"Char vocab file not found at {clstm_path}/char_vocab.pkl")
            raise
        # Initialize CLIP processors for mkgformer (main, auxiliary, RCNN images)
        self.clip_processor = CLIPProcessor.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.vit_name)
        )
        self.aux_processor = CLIPProcessor.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.vit_name)
        )
        self.aux_processor.feature_extractor.size = self.args.aux_size
        self.aux_processor.feature_extractor.crop_size = self.args.aux_size
        self.rcnn_processor = CLIPProcessor.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.vit_name)
        )
        self.rcnn_processor.feature_extractor.size = self.args.rcnn_size
        self.rcnn_processor.feature_extractor.crop_size = self.args.rcnn_size
        # Define NER labels
        self.LABELS = ["[PAD]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def load_from_file(self, mode="finetune"):
        """
        Load dataset from file based on mode.

        Args:
            mode (str): Dataset mode ('pretrain' for unlabeled, 'finetune' for labeled).

        Returns:
            dict: Contains words, targets_old, targets_new (for finetune), img_names, aux_img_dict, rcnn_img_dict.
        """
        load_file = self.data_path.get(mode)
        if not load_file or not os.path.exists(load_file):
            logger.error(f"Data file for mode '{mode}' not found at {load_file}")
            raise FileNotFoundError(f"Data file for mode '{mode}' not found")
        
        logger.info(f"Loading data from {load_file}")
        words, targets_old, targets_new, img_names = [], [], [], []
        word, target_old, target_new = [], [], []
        missing_images = 0
        
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("IMGID:"):
                    img_names.append(line.split("IMGID:")[1] + ".jpg")
                elif line:
                    tokens = line.split("\t")
                    if len(tokens) < 2:
                        logger.warning(f"Invalid line format: {line}")
                        continue
                    word.append(tokens[0])
                    target_old.append(tokens[1])
                    if mode == "finetune" and len(tokens) > 2:
                        target_new.append(tokens[-1])
                    elif mode == "finetune":
                        target_new.append(tokens[1])  # Fallback to target_old if target_new missing
                elif word:
                    words.append(word)
                    targets_old.append(target_old)
                    targets_new.append(target_new if mode == "finetune" else [])
                    word, target_old, target_new = [], [], []
        
        if word:
            words.append(word)
            targets_old.append(target_old)
            targets_new.append(target_new if mode == "finetune" else [])
            img_names.append(img_names[-1] if img_names else "missing.jpg")
        
        # Validate data lengths
        if len(words) != len(targets_old) or len(words) != len(img_names):
            logger.error(f"Data mismatch: words={len(words)}, targets_old={len(targets_old)}, img_names={len(img_names)}")
            raise ValueError("Data length mismatch")
        if mode == "finetune" and len(words) != len(targets_new):
            logger.error(f"Data mismatch in finetune mode: words={len(words)}, targets_new={len(targets_new)}")
            raise ValueError("Data length mismatch in targets_new")
        
        # Load auxiliary and RCNN image dictionaries
        try:
            aux_img_dict = torch.load(self.data_path.get("auximgs", ""))
        except FileNotFoundError:
            logger.warning(f"Aux image dict not found at {self.data_path.get('auximgs')}")
            aux_img_dict = {}
            missing_images += 1
        try:
            rcnn_img_dict = torch.load(self.data_path.get("img2crop", ""))
        except FileNotFoundError:
            logger.warning(f"RCNN image dict not found at {self.data_path.get('img2crop')}")
            rcnn_img_dict = {}
            missing_images += 1
        
        logger.info(f"Loaded {len(words)} samples, {missing_images} missing image dicts")
        return {
            "words": words,
            "targets_old": targets_old,
            "targets_new": targets_new,
            "img_names": img_names,
            "aux_img_dict": aux_img_dict,
            "rcnn_img_dict": rcnn_img_dict
        }

    def get_label_mapping(self):
        """Return dictionary mapping labels to indices."""
        return {label: idx for idx, label in enumerate(self.LABELS)}
    
    def get_label_embedding(self):
        """Generate BERT embeddings for labels, averaging multi-token words."""
        label2word_mapping = {
            "[PAD]": "[PAD]",
            "O": "other",
            "B-MISC": "miscellaneous",
            "I-MISC": "miscellaneous",
            "B-PER": "person",
            "I-PER": "person",
            "B-ORG": "organization",
            "I-ORG": "organization",
            "B-LOC": "location",
            "I-LOC": "location",
            "X": "unknown",
            "[CLS]": "[CLS]",
            "[SEP]": "[SEP]"
        }
        embeddings = []
        for label in self.LABELS:
            word = label2word_mapping[label]
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not token_ids or token_ids[0] == self.tokenizer.unk_token_id:
                logger.warning(f"Word '{word}' not in vocab, using [UNK] embedding")
                token_ids = [self.tokenizer.unk_token_id]
            
            token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
            with torch.no_grad():
                embedding = self.bert.get_input_embeddings()(token_id_tensor)  # (num_tokens, 768)
                embedding = embedding.mean(dim=0)  # Average multi-token embeddings
                embeddings.append(embedding)
        
        embedding_table = torch.stack(embeddings)  # (num_labels, 768)
        return embedding_table

class LEDDataset(Dataset):
    def __init__(self, processor, transform, imgs_path=None, aux_imgs_path=None, max_seq_len=128, max_char_len=128, mode="finetune", ignore_idx=0, aux_size=128, rcnn_imgs_path=None, rcnn_size=128):
        """
        Initialize dataset for LED (pretrain or finetune mode).

        Args:
            processor: LEDProcessor instance.
            transform: Image transform for hvpnet.
            imgs_path: Path to main images.
            aux_imgs_path: Path to auxiliary images.
            max_seq_len: Maximum sequence length.
            max_char_len: Maximum character length.
            mode: Dataset mode ('pretrain' or 'finetune').
            ignore_idx: Index for ignored labels.
            aux_size: Size for auxiliary images.
            rcnn_imgs_path: Path to RCNN images.
            rcnn_size: Size for RCNN images.
        """
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
        self.data_dict = None if mode == "subset" else processor.load_from_file(mode)

    def insert_data(self, words, img_names, targets_old, targets_new=None):
        """Insert new data into dataset."""
        if not self.data_dict:
            self.data_dict = {"words": [], "targets_old": [], "targets_new": [], "img_names": [], "aux_img_dict": {}, "rcnn_img_dict": {}}
        targets_new = targets_new or [[] for _ in targets_old]
        assert len(words) == len(targets_old) == len(targets_new) == len(img_names)
        self.data_dict["words"].extend(words)
        self.data_dict["targets_old"].extend(targets_old)
        self.data_dict["targets_new"].extend(targets_new)
        self.data_dict["img_names"].extend([name if name.endswith(".jpg") else name + ".jpg" for name in img_names])

    def delete_data(self, img_names):
        """Remove data by image names."""
        if not self.data_dict:
            return
        for name in img_names:
            name = name if name.endswith(".jpg") else name + ".jpg"
            try:
                idx = self.data_dict["img_names"].index(name)
                self.data_dict["words"].pop(idx)
                self.data_dict["targets_old"].pop(idx)
                self.data_dict["targets_new"].pop(idx)
                self.data_dict["img_names"].pop(idx)
            except ValueError:
                logger.warning(f"Image {name} not found in dataset")

    def from_indices(self, indices):
        """Create subset dataset from indices."""
        new_dataset = LEDDataset(
            processor=self.processor, transform=self.transform, imgs_path=self.imgs_path,
            aux_imgs_path=self.aux_imgs_path, max_seq_len=self.max_seq_len, max_char_len=self.max_char_len,
            mode="subset", ignore_idx=self.ignore_idx, aux_size=self.aux_size,
            rcnn_imgs_path=self.rcnn_imgs_path, rcnn_size=self.rcnn_size
        )
        new_data_dict = {
            "aux_img_dict": self.data_dict.get("aux_img_dict", {}),
            "rcnn_img_dict": self.data_dict.get("rcnn_img_dict", {}),
            "words": [self.data_dict["words"][i] for i in indices],
            "img_names": [self.data_dict["img_names"][i] for i in indices],
            "targets_old": [self.data_dict["targets_old"][i] for i in indices],
            "targets_new": [self.data_dict["targets_new"][i] for i in indices] if self.mode == "finetune" else []
        }
        new_dataset.data_dict = new_data_dict
        new_dataset.mode = self.mode
        return new_dataset

    def __len__(self):
        """Return dataset size."""
        return len(self.data_dict["words"]) if self.data_dict else 0

    def __getitem__(self, idx):
        """Get item by index, returning processed sequence and image data."""
        word_list = self.data_dict["words"][idx]
        img_name = self.data_dict["img_names"][idx]
        targets_old_list = self.data_dict["targets_old"][idx]
        targets_new_list = self.data_dict["targets_new"][idx] if self.mode == "finetune" else None

        # Process sequences
        seq_data = self._seq_proc(word_list, targets_old_list, targets_new_list)
        if self.mode == "finetune":
            token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_old, targets_new, words = seq_data
        else:  # pretrain
            token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_unk, words = seq_data

        # Process images (skip if use_prompt=False)
        image_data = self._img_proc(img_name) if self.imgs_path and self.processor.args.use_prompt else (None, None, None, None, None)
        hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs = image_data

        # Return tensors
        if self.imgs_path and self.aux_imgs_path and self.rcnn_imgs_path and self.processor.args.use_prompt:
            if self.mode == "finetune":
                return (
                    torch.tensor(targets_old, dtype=torch.long),
                    torch.tensor(targets_new, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words, img_name
                )
            else:  # pretrain
                return (
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words, img_name
                )
        elif self.imgs_path and self.aux_imgs_path and self.processor.args.use_prompt:
            if self.mode == "finetune":
                return (
                    torch.tensor(targets_old, dtype=torch.long),
                    torch.tensor(targets_new, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, words, img_name
                )
            else:  # pretrain
                return (
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, words, img_name
                )
        else:
            if self.mode == "finetune":
                return (
                    torch.tensor(targets_old, dtype=torch.long),
                    torch.tensor(targets_new, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    words, img_name
                )
            else:  # pretrain
                return (
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    words, img_name
                )
            
    def _seq_proc(self, word_list, target_old_list=None, target_new_list=None):
        """Process sequence data (tokens, chars, labels)."""
        tokens, char_input_ids, targets_old, targets_new, words = [], [], [], [], []
        label_map = self.processor.get_label_mapping()

        for i, word in enumerate(word_list):
            token = self.processor.tokenizer.tokenize(word)
            tokens.extend(token)
            char_ids = []
            for t in token:
                if t in self.processor.char2int:
                    char_ids.append([self.processor.char2int[t]] + [0] * (self.max_char_len - 1))
                else:
                    char_ids.append([self.processor.char2int.get(c, 0) for c in t[:self.max_char_len]] + [0] * (self.max_char_len - len(t)))
            char_input_ids.extend(char_ids)
            
            if target_old_list:
                target_old = target_old_list[i]
                for m in range(len(token)):
                    targets_old.append(label_map[target_old] if m == 0 else label_map["X"])
                    words.append(word)
            if target_new_list:
                target_new = target_new_list[i]
                for m in range(len(token)):
                    targets_new.append(label_map[target_new] if m == 0 else label_map["X"])

        # Truncate sequences
        if len(tokens) >= self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
            char_input_ids = char_input_ids[:self.max_seq_len - 2]
            words = words[:self.max_seq_len - 2]
            if targets_old:
                targets_old = targets_old[:self.max_seq_len - 2]
            if targets_new:
                targets_new = targets_new[:self.max_seq_len - 2]

        # Encode tokens
        token_encode_dict = self.processor.tokenizer.encode_plus(
            tokens, max_length=self.max_seq_len, truncation=True, padding="max_length"
        )
        token_input_ids = token_encode_dict["input_ids"]
        token_type_ids = token_encode_dict["token_type_ids"]
        token_attention_mask = token_encode_dict["attention_mask"]

        # Pad char inputs and labels
        char_input_ids = ([[self.processor.char2int["[CLS]"]] + [0] * (self.max_char_len - 1)] + 
                         char_input_ids + 
                         [[self.processor.char2int["[SEP]"]] + [0] * (self.max_char_len - 1)] + 
                         [[self.processor.char2int["[PAD]"]] * self.max_char_len] * (self.max_seq_len - len(char_input_ids) - 2))
        targets_unk = [label_map["[CLS]"]] + targets_old + [label_map["[SEP]"]] + [self.ignore_idx] * (self.max_seq_len - len(targets_old) - 2)
        if targets_new:
            targets_new = [label_map["[CLS]"]] + targets_new + [label_map["[SEP]"]] + [self.ignore_idx] * (self.max_seq_len - len(targets_new) - 2)
        words = ["[CLS]"] + words + ["[SEP]"] + ["[PAD]"] * (self.max_seq_len - len(words) - 2)

        if targets_new:
            return token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_unk, targets_new, words
        return token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_unk, words
        
    def _img_proc(self, img_name):
        """Process main, auxiliary, and RCNN images."""
        hvp_img = mkg_img = hvp_aux_imgs = mkg_aux_imgs = rcnn_imgs = None
        missing_images = 0
        
        if self.imgs_path:
            img_path = os.path.join(self.imgs_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                if image.size[0] < 10 or image.size[1] < 10:
                    logger.warning(f"Image {img_path} too small, using placeholder")
                    image = Image.new("RGB", (224, 224), color="white")
                    missing_images += 1
            except (FileNotFoundError, OSError):
                logger.warning(f"Image {img_path} not found, using placeholder")
                image = Image.new("RGB", (224, 224), color="white")
                missing_images += 1
            # hvpnet processing
            hvp_img = self.transform(image) if self.transform else torch.zeros(3, 224, 224)
            # mkgformer processing
            mkg_img = self.processor.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

        if self.aux_imgs_path and self.data_dict.get("aux_img_dict"):
            hvp_aux_imgs, mkg_aux_imgs = [], []
            aux_img_paths = self.data_dict["aux_img_dict"].get(img_name, [])
            aux_img_paths = [os.path.join(self.aux_imgs_path, path) for path in aux_img_paths[:3]]
            
            for path in aux_img_paths:
                try:
                    aux_img = Image.open(path).convert("RGB")
                    if aux_img.size[0] < 10 or aux_img.size[1] < 10:
                        logger.warning(f"Aux image {path} too small, using placeholder")
                        aux_img = Image.new("RGB", (self.aux_size, self.aux_size), color="white")
                        missing_images += 1
                    # hvpnet processing
                    hvp_aux_imgs.append(self.transform(aux_img) if self.transform else torch.zeros(3, 224, 224))
                    # mkgformer processing
                    mkg_aux_imgs.append(self.processor.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze())
                except (FileNotFoundError, OSError):
                    logger.warning(f"Aux image {path} not found, using placeholder")
                    hvp_aux_imgs.append(torch.zeros(3, 224, 224))
                    mkg_aux_imgs.append(torch.zeros(3, self.aux_size, self.aux_size))
                    missing_images += 1
            
            while len(hvp_aux_imgs) < 3:
                hvp_aux_imgs.append(torch.zeros(3, 224, 224))
                mkg_aux_imgs.append(torch.zeros(3, self.aux_size, self.aux_size))
            
            hvp_aux_imgs = torch.stack(hvp_aux_imgs)
            mkg_aux_imgs = torch.stack(mkg_aux_imgs)

        if self.rcnn_imgs_path and self.data_dict.get("rcnn_img_dict"):
            rcnn_imgs = []
            img_key = img_name.split('.')[0]
            rcnn_img_paths = self.data_dict["rcnn_img_dict"].get(img_key, [])
            rcnn_img_paths = [os.path.join(self.rcnn_imgs_path, path) for path in rcnn_img_paths[:3]]
            
            for path in rcnn_img_paths:
                try:
                    rcnn_img = Image.open(path).convert("RGB")
                    if rcnn_img.size[0] < 10 or rcnn_img.size[1] < 10:
                        logger.warning(f"RCNN image {path} too small, using placeholder")
                        rcnn_img = Image.new("RGB", (self.rcnn_size, self.rcnn_size), color="white")
                        missing_images += 1
                    # mkgformer processing
                    rcnn_imgs.append(self.processor.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze())
                except (FileNotFoundError, OSError):
                    logger.warning(f"RCNN image {path} not found, using placeholder")
                    rcnn_imgs.append(torch.zeros(3, self.rcnn_size, self.rcnn_size))
                    missing_images += 1
            
            while len(rcnn_imgs) < 3:
                rcnn_imgs.append(torch.zeros(3, self.rcnn_size, self.rcnn_size))
            
            rcnn_imgs = torch.stack(rcnn_imgs)

        # Log missing images
        if missing_images > 0:
            logger.info(f"Processed {img_name} with {missing_images} missing or invalid images")

        # Return tensors, using zeros if None
        return (
            hvp_img if hvp_img is not None else torch.zeros(3, 224, 224),
            hvp_aux_imgs if hvp_aux_imgs is not None else torch.zeros(3, 3, self.aux_size, self.aux_size),
            mkg_img if mkg_img is not None else torch.zeros(3, 224, 224),
            mkg_aux_imgs if mkg_aux_imgs is not None else torch.zeros(3, 3, self.aux_size, self.aux_size),
            rcnn_imgs if rcnn_imgs is not None else torch.zeros(3, 3, self.rcnn_size, self.rcnn_size)
        )