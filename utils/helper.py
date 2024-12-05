import requests
from io import BytesIO

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
from PIL import Image

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from segment_anything import SamPredictor, build_sam


def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, "Status code error: {}.".format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)

    print(
        "Image downloaded from url: {} and saved to: {}.".format(url, image_file_path)
    )


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)
    ).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    """
    This helper function downloads the Grounding Dino model from HuggingFace
    """

    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def get_sam_predictor(checkpoint):
    return SamPredictor(build_sam(checkpoint=checkpoint))


def get_stable_diffusion_pipeline(model):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
    )

    pipe = pipe.to("cuda")

    return pipe
