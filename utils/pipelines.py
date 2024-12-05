from PIL import Image
import torch

from GroundingDINO.groundingdino.util import box_ops
from groundingdino.util.inference import annotate, load_image, predict

from utils.helper import load_model_hf, show_mask
from utils.config import repo_id, filename, config_filename

groundingdino_model = load_model_hf(repo_id, filename, config_filename)

def inpaint_image(image_path, target_object, prompt, sam_predictor, stable_diffusion_pipeline):
    # Load the image from the provided path
    image_source, image = load_image(image_path)

    # Predict bounding boxes and masks for the target object in the image
    boxes, logits, phrases = predict(   
        model=groundingdino_model,
        image=image,
        caption=target_object,
        box_threshold=0.3,
        text_threshold=0.25
    )

    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    sam_predictor.set_image(image_source)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
    image_mask = masks[0][0].cpu().numpy()
    image_source_pil = Image.fromarray(image_source)
    annotated_frame_pil = Image.fromarray(annotated_frame)
    image_mask_pil = Image.fromarray(image_mask)
    annotated_frame_with_mask_pil = Image.fromarray(annotated_frame_with_mask)

    image_source_for_inpaint = image_source_pil.resize((512, 512))
    image_mask_for_inpaint = image_mask_pil.resize((512, 512))
    image_inpainting = stable_diffusion_pipeline(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
    image_inpainting = image_inpainting.resize((image_source_pil.size[0], image_source_pil.size[1]))
    return image_inpainting
