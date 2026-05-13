import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional, List, Tuple
import uuid

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

LANG_SYSTEM_PROMPT = {
    "en": "You are a helpful assistant. Answer in English.",
    "zh": "你是一个有帮助的助手。请用中文回答。",
    "ar": "أنت مساعد مفيد. أجب باللغة العربية.",
    "es": "Eres un asistente útil. Responde en español.",
    "fr": "Tu es un assistant utile. Réponds en français.",
    "de": "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
    "ru": "Ты полезный помощник. Отвечай по-русски.",
    "ja": "あなたは有能なアシスタントです。日本語で答えてください。",
    "ko": "당신은 유능한 도우미입니다. 한국어로 답변하세요.",
    "vi": "Bạn là một trợ lý hữu ích. Hãy trả lời bằng tiếng Việt.",
    "it": "Sei un assistente utile. Rispondi in italiano.",
    "pl": "Jesteś pomocnym asystentem. Odpowiedz po polsku.",
    "pt": "Você é um assistente útil. Responda em português.",
    "id": "Anda adalah asisten yang membantu. Jawablah dalam bahasa Indonesia.",
    "th": "คุณเป็นผู้ช่วยที่เป็นประโยชน์ โปรดตอบเป็นภาษาไทย",
}

language_to_code = {
    "arabic": "ar",
    "chinese": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "vietnamese": "vi",
}


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class MultilangRepeatDataset(Dataset):

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        
        langs_sorted = sorted(set(language_to_code.values()))
        uniform_mix = ",".join([f"{lang}:1" for lang in langs_sorted])

        self.topic_lang_mix = {
            "General Knowledge": uniform_mix,
            "Regional Knowledge": uniform_mix,
            "Reasoning / Logic": uniform_mix,
            "Safety / Ethics": uniform_mix,
            "Chat / Conversational": uniform_mix,
        }
        self.region_lang_mix = {
            "Russia": uniform_mix,
            "Southeast Asia": uniform_mix,
            "Latin America": uniform_mix,
            "Japan": uniform_mix,
            "South Asia": uniform_mix,
            "North Korea": uniform_mix,
            "Anglosphere": uniform_mix,
            "Europe": uniform_mix,
            "Africa": uniform_mix,
            "China": uniform_mix,
            "Israel": uniform_mix,
            "France": uniform_mix,
            "Turkey": uniform_mix,
            "Germany": uniform_mix,
            "South Korea": uniform_mix,
            "Iran": uniform_mix,
            "Spain": uniform_mix,
            "Poland": uniform_mix,
            "Vietnam": uniform_mix,
            "Polynesia": uniform_mix,
            "India": uniform_mix,
            "Central Asia": uniform_mix,
            "Arab World": uniform_mix,
            "Italy": uniform_mix
        }
        self.topic_lang_seq = {key: self._parse_lang_mix(self.topic_lang_mix[key]) for key in self.topic_lang_mix}
        self.region_lang_seq = {key: self._parse_lang_mix(self.region_lang_mix[key]) for key in self.region_lang_mix}
        self.n = 8

        self.id_key: str = config.get("id_key", "id")
        self._uid_cache = {}

        self._download()
        self._read_files_and_tokenize()

    @staticmethod
    def _parse_lang_mix(mix: str) -> list[str]:
        # "en:4,zh:2" -> ["en","en","en","en","zh","zh"]
        out = []
        for seg in mix.split(","):
            lang, k = seg.split(":")
            out += [lang.strip()] * int(k)
        return out

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key
            
            all_langs = set()
            all_langs.update(language_to_code.values())
            langs_unique = list(all_langs)
            
            # def build_messages_from_lang(doc, ln: str) -> list[dict]:
            #     system_prompt = LANG_SYSTEM_PROMPT.get(ln, f"You are a helpful assistant. Answer in {ln}.")
            #     q = doc.get("question", None)
            #     if not q:
            #         return None
            #     return [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]

            def build_messages_from_lang(doc, ln: str) -> list[dict]:
                tag = "<|out_lang_{ln}|>"
                q = doc.get("question", None)
                q=f"{q.strip()}\n{tag}"
                if not q:
                    return None
                return [{"role": "user", "content": q}]

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:
                def doc2len(doc) -> int:
                    lengths = []
                    for ln in langs_unique:
                        m = build_messages_from_lang(doc, ln)
                        if m is None:
                            continue
                        s = tokenizer.apply_chat_template(
                            m, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                        )
                        ids = tokenizer(s, add_special_tokens=False, return_attention_mask=False, return_tensors=None)["input_ids"]
                        lengths.append(len(ids))
                    return max(lengths) if lengths else 0

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe) * self.n

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        base_idx = item // self.n
        rep_idx = item % self.n
        row_dict: dict = self.dataframe[base_idx]

        ability = row_dict.get("ability", None)
        topic = ability
        region = row_dict.get("extra_info", {}).get("region", None)
        orig_lang_name = row_dict.get("extra_info", {}).get("language", None)
        orig_lang = language_to_code.get(orig_lang_name, "en") if orig_lang_name else "en"
        fixed_lang = orig_lang if ability == "Translation" else None
        lang = fixed_lang or orig_lang

        question = row_dict.get(f"question", None)
        answer = row_dict.get(f"answer", None)
        if not question or not answer:
            raise KeyError(f"Missing question/answer for lang={lang} at base_idx={base_idx} rep_idx={rep_idx}, question={question}, answer={answer}")
        # system_prompt = LANG_SYSTEM_PROMPT.get(lang, f"You are a helpful assistant. Answer in {lang}.")
        # messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

        row_dict["question_text"] = question
        tag = f"<|out_lang_{lang}|>"
        question = f"{question.strip()}\n{tag}"
        messages = [{"role": "user", "content": question}]
        
        model_inputs = {}
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        row_dict['prompt']=messages
        row_dict["reward_model"] = {"style": "rule", "ground_truth": answer}

        if base_idx not in self._uid_cache:
            self._uid_cache[base_idx] = str(uuid.uuid4())
        row_dict["uid"] = self._uid_cache[base_idx]

        row_dict["lang"] = lang
        row_dict["topic"] = topic
        row_dict["region"] = region
        row_dict["orig_lang"] = orig_lang
        row_dict["fixed_lang"] = fixed_lang

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
