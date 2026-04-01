import torch
from PIL import Image

from .base import BaseModel
from ..smp import *


class InternVL3Chat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='OpenGVLab/InternVL3-8B', **kwargs):
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model_path = model_path
        self.device = 'cuda'

        default_kwargs = dict(
            do_sample=False,
            max_new_tokens=2048,
            top_p=None,
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def generate_inner(self, message, dataset=None):
        # Build prompt with <image> placeholders
        image_count = 0
        prompt_parts = []
        pixel_values_list = []

        for msg in message:
            if msg['type'] == 'image':
                image_count += 1
                img = Image.open(msg['value']).convert('RGB')
                pixel_values_list.append(
                    self._load_image(msg['value']).to(self.device, dtype=torch.bfloat16)
                )
                prompt_parts.append(f'<image>\n')
            elif msg['type'] == 'text':
                prompt_parts.append(msg['value'])

        prompt = ''.join(prompt_parts)

        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            num_patches_list = [pv.size(0) for pv in pixel_values_list]
        else:
            pixel_values = None
            num_patches_list = []

        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt,
            generation_config=self.kwargs,
            verbose=False,
        )
        return response

    @staticmethod
    def _load_image(image_path, max_num=6, input_size=448):
        """Load and preprocess image into tiles, matching InternVL3's expected format."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        image = Image.open(image_path).convert('RGB')
        pixel_values = transform(image).unsqueeze(0)
        return pixel_values
