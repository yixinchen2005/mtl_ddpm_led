import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, CLIPProcessor, BertModel
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class LEDProcessor:
    def __init__(self, args, data_path='data', clstm_path='clstm'):
        """Initialize processor for LED dataset with BERT and CLIP models."""
        self.data_path = data_path
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.lm_name), do_lower_case=True
        )
        self.bert = BertModel.from_pretrained(
            os.path.join(self.args.local_cache_path, self.args.lm_name)
        )
        try:
            self.char2int, self.int2char = torch.load(os.path.join(clstm_path, "char_vocab.pkl"))
        except FileNotFoundError:
            logger.error(f"Char vocab file not found at {clstm_path}/char_vocab.pkl")
            raise
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
        self.LABELS = ["[PAD]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def load_from_file(self, mode="finetune"):
        """
        Load dataset from file based on mode.

        Args:
            mode (str): Dataset mode ('pretrain' for unlabeled, 'finetune' for labeled).

        Returns:
            dict: Contains words, targets_unk, targets_new (finetune only), img_names, aux_img_dict, rcnn_img_dict.
        """
        load_file = self.data_path.get(mode)
        if not load_file or not os.path.exists(load_file):
            logger.error(f"Data file for mode '{mode}' not found at {load_file}")
            raise FileNotFoundError(f"Data file for mode '{mode}' not found")
        
        logger.info(f"Loading data from {load_file}")
        words, targets_unk, targets_new, img_names = [], [], [], []
        word, target_unk, target_new = [], [], []
        current_imgid = None
        missing_images = 0
        malformed_lines = 0
        sentence_count = 0

        def process_sentence():
            nonlocal words, targets_unk, targets_new, img_names, sentence_count
            if word and current_imgid is not None:
                # Assert length equality
                assert len(word) == len(target_unk), (
                    f"Length mismatch in IMGID:{current_imgid}: words={len(word)}, targets_unk={len(target_unk)}"
                )
                if mode == "finetune":
                    assert len(word) == len(target_new), (
                        f"Length mismatch in IMGID:{current_imgid}: words={len(word)}, targets_new={len(target_new)}"
                    )
                words.append(word)
                targets_unk.append(target_unk)
                targets_new.append(target_new if mode == "finetune" else [])
                img_names.append(current_imgid + ".jpg")
                sentence_count += 1

        with open(load_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("IMGID:"):
                    word, target_unk, target_new = [], [], []
                    current_imgid = line.split("IMGID:")[1]
                elif line:
                    if current_imgid is None:
                        logger.warning(f"Line before IMGID: {line}, skipping")
                        malformed_lines += 1
                        continue
                    tokens = line.split("\t")
                    expected_columns = 3 if mode == "finetune" else 2
                    if len(tokens) != expected_columns:
                        logger.warning(f"Malformed line in IMGID:{current_imgid}: {line} (expected {expected_columns} columns, got {len(tokens)})")
                        malformed_lines += 1
                        continue
                    word.append(tokens[0])
                    target_unk.append(tokens[1])
                    if mode == "finetune":
                        target_new.append(tokens[2])
                else:  # Empty line indicates end of sentence
                    process_sentence()
                    word, target_unk, target_new = [], [], []

        # Process the final sentence
        process_sentence()

        # Validate data lengths
        assert len(words) == len(targets_unk) == len(img_names), (
            f"Data mismatch: words={len(words)}, targets_unk={len(targets_unk)}, img_names={len(img_names)}"
        )
        if mode == "finetune":
            assert len(words) == len(targets_new), (
                f"Data mismatch in finetune mode: words={len(words)}, targets_new={len(targets_new)}"
            )

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
        
        logger.info(f"Loaded {len(words)} samples, {sentence_count} sentences, {missing_images} missing image dicts, {malformed_lines} malformed lines")
        return {
            "words": words,
            "targets_unk": targets_unk,
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
            "[PAD]": "[PAD]", "O": "other", "B-MISC": "miscellaneous", "I-MISC": "miscellaneous",
            "B-PER": "person", "I-PER": "person", "B-ORG": "organization", "I-ORG": "organization",
            "B-LOC": "location", "I-LOC": "location", "X": "unknown", "[CLS]": "[CLS]", "[SEP]": "[SEP]"
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
                embedding = self.bert.get_input_embeddings()(token_id_tensor)
                embedding = embedding.mean(dim=0)
                embeddings.append(embedding)
        return torch.stack(embeddings)

class LEDDataset(Dataset):
    def __init__(self, processor, transform, imgs_path=None, aux_imgs_path=None, max_seq_len=128, max_char_len=128, 
                 mode="finetune", ignore_idx=0, aux_size=128, rcnn_imgs_path=None, rcnn_size=128):
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

    def from_indices(self, indices):
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
            "targets_unk": [self.data_dict["targets_unk"][i] for i in indices],
            "targets_new": [self.data_dict["targets_new"][i] for i in indices] if self.mode == "finetune" else []
        }
        new_dataset.data_dict = new_data_dict
        new_dataset.mode = self.mode
        return new_dataset

    def __len__(self):
        return len(self.data_dict["words"]) if self.data_dict else 0

    def __getitem__(self, idx):
        word_list = self.data_dict["words"][idx]
        img_name = self.data_dict["img_names"][idx]
        targets_unk_list = self.data_dict["targets_unk"][idx]
        targets_new_list = self.data_dict["targets_new"][idx] if self.mode == "finetune" else None

        # Validate input lengths
        assert len(word_list) == len(targets_unk_list), (
            f"Length mismatch at index {idx} (IMGID:{img_name}): words={len(word_list)}, targets_unk={len(targets_unk_list)}"
        )
        if self.mode == "finetune":
            assert len(word_list) == len(targets_new_list), (
                f"Length mismatch at index {idx} (IMGID:{img_name}): words={len(word_list)}, targets_new={len(targets_new_list)}"
            )

        seq_data = self._seq_proc(word_list, targets_unk_list, targets_new_list)
        token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_unk, targets_new, words = seq_data

        image_data = self._img_proc(img_name) if self.imgs_path and self.processor.args.use_prompt else (None, None, None, None, None)
        hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs = image_data

        if self.imgs_path and self.aux_imgs_path and self.rcnn_imgs_path and self.processor.args.use_prompt:
            if self.mode == "finetune":
                return (
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(targets_new, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs, words, img_name
                )
            else:
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
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(targets_new, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, words, img_name
                )
            else:
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
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(targets_new, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    words, img_name
                )
            else:
                return (
                    torch.tensor(targets_unk, dtype=torch.long),
                    torch.tensor(char_input_ids, dtype=torch.long),
                    torch.tensor(token_input_ids, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(token_attention_mask, dtype=torch.long),
                    words, img_name
                )

    def _seq_proc(self, word_list, target_unk_list=None, target_new_list=None):
        tokens, char_input_ids, targets_unk, targets_new, words = [], [], [], [], []
        label_map = self.processor.get_label_mapping()

        # Validate input lengths
        assert len(word_list) == len(target_unk_list), (
            f"Length mismatch in _seq_proc: words={len(word_list)}, targets_unk={len(target_unk_list)}"
        )
        if target_new_list:
            assert len(word_list) == len(target_new_list), (
                f"Length mismatch in _seq_proc: words={len(word_list)}, targets_new={len(target_new_list)}"
            )

        for i, word in enumerate(word_list):
            token = self.processor.tokenizer.tokenize(word)
            tokens.extend(token)
            char_ids = []
            for t in token:
                if t in self.processor.char2int:
                    char_ids.append([self.processor.char2int[t]] + [0] * (self.max_char_len - 1))
                else:
                    char_ids.append([self.processor.char2int.get(c, 0) for c in t[:self.max_char_len]] + 
                                    [0] * (self.max_char_len - len(t)))
            char_input_ids.extend(char_ids)
            
            target_unk = target_unk_list[i]
            for m in range(len(token)):
                targets_unk.append(label_map[target_unk] if m == 0 else label_map["X"])
                words.append(word)
            if target_new_list:
                target_new = target_new_list[i]
                for m in range(len(token)):
                    targets_new.append(label_map[target_new] if m == 0 else label_map["X"])

        if len(tokens) >= self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
            char_input_ids = char_input_ids[:self.max_seq_len - 2]
            words = words[:self.max_seq_len - 2]
            targets_unk = targets_unk[:self.max_seq_len - 2]
            if target_new_list:
                targets_new = targets_new[:self.max_seq_len - 2]

        token_encode_dict = self.processor.tokenizer.encode_plus(
            tokens, max_length=self.max_seq_len, truncation=True, padding="max_length"
        )
        token_input_ids = token_encode_dict["input_ids"]
        token_type_ids = token_encode_dict["token_type_ids"]
        token_attention_mask = token_encode_dict["attention_mask"]

        char_input_ids = ([[self.processor.char2int["[CLS]"]] + [0] * (self.max_char_len - 1)] + 
                          char_input_ids + 
                          [[self.processor.char2int["[SEP]"]] + [0] * (self.max_char_len - 1)] + 
                          [[self.processor.char2int["[PAD]"]] * self.max_char_len] * (self.max_seq_len - len(char_input_ids) - 2))
        targets_unk = [label_map["[CLS]"]] + targets_unk + [label_map["[SEP]"]] + [self.ignore_idx] * (self.max_seq_len - len(targets_unk) - 2)
        if target_new_list:
            targets_new = [label_map["[CLS]"]] + targets_new + [label_map["[SEP]"]] + [self.ignore_idx] * (self.max_seq_len - len(targets_new) - 2)
        words = ["[CLS]"] + words + ["[SEP]"] + ["[PAD]"] * (self.max_seq_len - len(words) - 2)

        # Validate output lengths
        assert len(token_input_ids) == self.max_seq_len, (
            f"Token input_ids length mismatch: {len(token_input_ids)}, expected={self.max_seq_len}"
        )
        assert len(token_type_ids) == self.max_seq_len, (
            f"Token type_ids length mismatch: {len(token_type_ids)}, expected={self.max_seq_len}"
        )
        assert len(token_attention_mask) == self.max_seq_len, (
            f"Token attention_mask length mismatch: {len(token_attention_mask)}, expected={self.max_seq_len}"
        )
        assert len(char_input_ids) == self.max_seq_len, (
            f"Char input_ids length mismatch: {len(char_input_ids)}, expected={self.max_seq_len}"
        )
        assert all(len(c) == self.max_char_len for c in char_input_ids), (
            f"Char input sub-length mismatch: {[len(c) for c in char_input_ids]}, expected={self.max_char_len}"
        )
        assert len(targets_unk) == self.max_seq_len, (
            f"Targets_unk length mismatch: {len(targets_unk)}, expected={self.max_seq_len}"
        )
        if target_new_list:
            assert len(targets_new) == self.max_seq_len, (
                f"Targets_new length mismatch: {len(targets_new)}, expected={self.max_seq_len}"
            )
        assert len(words) == self.max_seq_len, (
            f"Words length mismatch: {len(words)}, expected={self.max_seq_len}"
        )

        return token_input_ids, token_type_ids, token_attention_mask, char_input_ids, targets_unk, targets_new, words

    def _img_proc(self, img_name):
        hvp_img = mkg_img = hvp_aux_imgs = mkg_aux_imgs = rcnn_imgs = None
        missing_images = 0
        
        if self.imgs_path is None:
            return hvp_img, hvp_aux_imgs, mkg_img, mkg_aux_imgs, rcnn_imgs
        
        if self.imgs_path:
            img_path = os.path.join(self.imgs_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                if image.size[0] < 10 or image.size[1] < 10:
                    logger.debug(f"Image {img_path} too small (size: {image.size}), using placeholder")
                    image = Image.new("RGB", (224, 224), color="white")
                    missing_images += 1
                hvp_img = self.transform(image) if self.transform else torch.zeros(3, 224, 224)
                mkg_img = self.processor.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
                logger.debug(f"Processed main image {img_path}, size: {image.size}, hvp_img shape: {hvp_img.shape if hvp_img is not None else 'None'}, mkg_img shape: {mkg_img.shape if mkg_img is not None else 'None'}")
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Image {img_path} not found or invalid ({str(e)}), using placeholder")
                image = Image.new("RGB", (224, 224), color="white")
                missing_images += 1
                hvp_img = torch.zeros(3, 224, 224)
                mkg_img = torch.zeros(3, 224, 224)

            if self.aux_imgs_path:
                hvp_aux_imgs, mkg_aux_imgs = [], []
                aux_img_paths = self.data_dict.get("aux_img_dict", {}).get(img_name, [])
                aux_img_paths = [os.path.join(self.aux_imgs_path, path) for path in aux_img_paths[:3]]
                
                for path in aux_img_paths:
                    try:
                        aux_img = Image.open(path).convert("RGB")
                        if aux_img.size[0] < 10 or aux_img.size[1] < 10:
                            logger.debug(f"Aux image {path} too small (size: {aux_img.size}), using placeholder")
                            aux_img = Image.new("RGB", (self.aux_size, self.aux_size), color="white")
                            missing_images += 1
                        hvp_aux_imgs.append(self.processor.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze())
                        mkg_aux_imgs.append(self.processor.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze())
                        logger.debug(f"Processed aux image {path}, size: {aux_img.size}, hvp_aux_img shape: {hvp_aux_imgs[-1].shape}, mkg_aux_img shape: {mkg_aux_imgs[-1].shape}")
                    except (FileNotFoundError, OSError) as e:
                        logger.warning(f"Aux image {path} not found or invalid ({str(e)}), using placeholder")
                        hvp_aux_imgs.append(torch.zeros(3, self.aux_size, self.aux_size))
                        mkg_aux_imgs.append(torch.zeros(3, self.aux_size, self.aux_size))
                        missing_images += 1
                
                while len(hvp_aux_imgs) < 3:
                    hvp_aux_imgs.append(torch.zeros(3, self.aux_size, self.aux_size))
                    mkg_aux_imgs.append(torch.zeros(3, self.aux_size, self.aux_size))
                    logger.debug(f"Added placeholder aux image, shape: [3, {self.aux_size}, {self.aux_size}]")
                
                hvp_aux_imgs = torch.stack(hvp_aux_imgs)
                mkg_aux_imgs = torch.stack(mkg_aux_imgs)

            if self.rcnn_imgs_path and self.data_dict.get("rcnn_img_dict"):
                rcnn_imgs = []
                img_key = img_name.split('.')[0]
                rcnn_img_paths = self.data_dict.get("rcnn_img_dict", {}).get(img_key, [])
                rcnn_img_paths = [os.path.join(self.rcnn_imgs_path, path) for path in rcnn_img_paths[:3]]
                
                for path in rcnn_img_paths:
                    try:
                        rcnn_img = Image.open(path).convert("RGB")
                        if rcnn_img.size[0] < 10 or rcnn_img.size[1] < 10:
                            logger.debug(f"RCNN image {path} too small (size: {rcnn_img.size}), using placeholder")
                            rcnn_img = Image.new("RGB", (self.rcnn_size, self.rcnn_size), color="white")
                            missing_images += 1
                        rcnn_imgs.append(self.processor.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze())
                        logger.debug(f"Processed RCNN image {path}, size: {rcnn_img.size}, rcnn_img shape: {rcnn_imgs[-1].shape}")
                    except (FileNotFoundError, OSError) as e:
                        logger.warning(f"RCNN image {path} not found or invalid ({str(e)}), using placeholder")
                        rcnn_imgs.append(torch.zeros(3, self.rcnn_size, self.rcnn_size))
                        missing_images += 1
                
                while len(rcnn_imgs) < 3:
                    rcnn_imgs.append(torch.zeros(3, self.rcnn_size, self.rcnn_size))
                    logger.debug(f"Added placeholder RCNN image, shape: [3, {self.rcnn_size}, {self.rcnn_size}]")
                
                rcnn_imgs = torch.stack(rcnn_imgs)

            if missing_images > 0:
                logger.info(f"Processed {img_name} with {missing_images} missing or invalid images")

            return (
                hvp_img if hvp_img is not None else torch.zeros(3, 224, 224),
                hvp_aux_imgs if hvp_aux_imgs is not None else torch.zeros(3, 3, self.aux_size, self.aux_size),
                mkg_img if mkg_img is not None else torch.zeros(3, 224, 224),
                mkg_aux_imgs if mkg_aux_imgs is not None else torch.zeros(3, 3, self.aux_size, self.aux_size),
                rcnn_imgs if rcnn_imgs is not None else torch.zeros(3, 3, self.rcnn_size, self.rcnn_size)
            )