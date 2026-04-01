import re
import torch
import logging
import warnings

from .qwen2_vl.prompt import Qwen2VLPromptMixin
from .qwen2_vl.model import ensure_image_url, ensure_video_url
from .base import BaseModel
from ..smp import *


class Qwen3VLChat(Qwen2VLPromptMixin, BaseModel):
    """Qwen3-VL model for both Instruct and Thinking variants."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        thinking: bool = False,
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.model_path = model_path
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2

        # Thinking models need sampling; Instruct can use greedy
        if thinking:
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=max(temperature, 0.6),  # thinking needs temp >= 0.6
                do_sample=True,
            )
        else:
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )

        try:
            import flash_attn  # noqa: F401
            attn_impl = 'flash_attention_2'
        except ImportError:
            attn_impl = 'sdpa'

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map='auto',
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs, dataset=None):
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found: pip install qwen-vl-utils")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=text, images=images, videos=videos,
            padding=True, return_tensors='pt',
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        response = out[0]

        # Strip thinking tags from output if present
        if self.thinking:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
