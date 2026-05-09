import cv2
import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoModel


def calculate_ctr(mask: np.ndarray) -> float:
    # mask.ndim = 2, (height, width)
    lungs = np.zeros_like(mask)
    lungs[mask == 1] = 1
    lungs[mask == 2] = 1
    heart = (mask == 3).astype("int")
    y, x = np.stack(np.where(lungs == 1))
    lung_min = x.min()
    lung_max = x.max()
    y, x = np.stack(np.where(heart == 1))
    heart_min = x.min()
    heart_max = x.max()
    lung_range = lung_max - lung_min
    heart_range = heart_max - heart_min
    return heart_range / lung_range


def make_overlay(
    img: np.ndarray, mask: np.ndarray, alpha: float = 0.7
) -> np.ndarray[np.uint8]:
    overlay = alpha * img + (1 - alpha) * mask
    return overlay.astype(np.uint8)


@spaces.GPU
def predict(Radiograph, invert: bool = False):
    # --- предобработка изображения ---
    if invert:
        Radiograph = cv2.bitwise_not(Radiograph)  # негатив (инверсия яркости)
    rg = cv2.cvtColor(Radiograph, cv2.COLOR_GRAY2RGB)
    x = cxr_info_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")

    with torch.inference_mode():
        info_out = cxr_info_model(x)

    info_mask = info_out["mask"]
    h, w = rg.shape[:2]
    info_mask = F.interpolate(info_mask, size=(h, w), mode="bilinear")
    info_mask = info_mask.argmax(1)[0]
    info_mask_3ch = F.one_hot(info_mask, num_classes=4)[..., 1:]
    info_mask_3ch = (info_mask_3ch.cpu().numpy() * 255).astype(np.uint8)
    info_overlay = make_overlay(rg, info_mask_3ch[..., ::-1])

    view = info_out["view"].argmax(1).item()
    info_string = ""
    if view in {0, 1}:
        info_string += "This is a frontal chest radiograph "
        if view == 0:
            info_string += "(AP projection)."
        elif view == 1:
            info_string += "(PA projection)."
    elif view == 2:
        info_string += "This is a lateral chest radiograph."

    age = info_out["age"].item()
    info_string += f"\nThe patient's predicted age is {round(age)} years."
    sex = info_out["female"].item()
    if sex < 0.5:
        sex = "male"
    else:
        sex = "female"
    info_string += f"\nThe patient's predicted sex is {sex}."

    if view in {0, 1}:
        ctr = calculate_ctr(info_mask.cpu().numpy())
        info_string += f"\nThe estimated cardiothoracic ratio (CTR) is {ctr:0.2f}."
        if view == 0:
            info_string += (
                "\nNote that the cardiac silhuoette is magnified in the AP projection."
            )

    if view == 2:
        info_string += (
            "\nNOTE: The below outputs are NOT VALID for lateral radiographs."
        )

    x = pna_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")

    with torch.inference_mode():
        pna_out = pna_model(x)

    pna_mask = pna_out["mask"]
    h, w = rg.shape[:2]
    pna_mask = F.interpolate(pna_mask, size=(h, w), mode="bilinear")
    pna_mask = (pna_mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    pna_mask = cv2.applyColorMap(pna_mask, cv2.COLORMAP_JET)
    pna_overlay = make_overlay(rg, pna_mask[..., ::-1])

    x = ptx_model.preprocess(Radiograph)
    x = torch.from_numpy(x).float().to(device)
    x = rearrange(x, "h w -> 1 1 h w")

    with torch.inference_mode():
        ptx_out = ptx_model(x)

    ptx_mask = ptx_out["mask"]
    h, w = rg.shape[:2]
    ptx_mask = F.interpolate(ptx_mask, size=(h, w), mode="bilinear")
    ptx_mask = (ptx_mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    ptx_mask = cv2.applyColorMap(ptx_mask, cv2.COLORMAP_JET)
    ptx_overlay = make_overlay(rg, ptx_mask[..., ::-1])

    preds = {"Pneumonia": pna_out["cls"].item(), "Pneumothorax": ptx_out["cls"].item()}
    return [info_string, preds, info_overlay, pna_overlay, ptx_overlay]


image = gr.Image(image_mode="L")
# --- новые контролы ---
invert_checkbox = gr.Checkbox(label="⬜ Негатив (инверсия)", value=False)

info_textbox = gr.Textbox(show_label=True,label="age detection",lines=5)
labels = gr.Label(show_label=True, show_heading=True,label="progress bar")
heatmap1 = gr.Image(image_mode="RGB", label="Heart & Lungs")
heatmap2 = gr.Image(image_mode="RGB", label="Pneumonia")
heatmap3 = gr.Image(image_mode="RGB", label="Pneumothorax")


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Deep Learning for Radiology

    This demo uses 3 models for chest radiographs:
    1) Heart + lungs segmentation, with age, view, and sex prediction
    2) Pneumonia classification and segmentation
    3) Pneumothorax classification and segmentation

    Note that the pneumonia and pneumothorax heatmaps produced by this model are based on pixel-level segmentation maps.

    The example radiograph is my friend and my mom, from when he has pneumonia. 

    This training for demonstration purposes only, was never approved by any regulatory agency for clinical use.
    The user assumes any and all responsibility regarding their own use of this model and its outputs.
    Do NOT upload any images containing protected health information, as this demonstration is not compliant with patient privacy laws.

    You can use this x-ray image from examples<hr>

    """
    )
    #gr.Image(value="examples/mymom1.jpg", label="my mom x-ray example")
    gr.Markdown(
        """
    # Main interface

    Upload x-ray image and Submit for analyze

        """
    )
    gr.Interface(
        fn=predict,
        inputs=[image, invert_checkbox],   # ← добавили чекбоксы
        outputs=[info_textbox, labels, heatmap1, heatmap2, heatmap3],
        examples=[
            ["examples/mymom1.jpg", True, "Пневмония (мама друга)"],
            ["examples/cxr.png",    False, "Стандартный снимок PA"],
        ],
        cache_examples=True,
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device `{device}` ...")

    cxr_info_model = (AutoModel.from_pretrained("Dimaodessa/chest-x-ray-basic", trust_remote_code=True).eval().to(device))
    pna_model =      (AutoModel.from_pretrained("Dimaodessa/pneumonia-cxr",     trust_remote_code=True).eval().to(device))
    ptx_model =      (AutoModel.from_pretrained("Dimaodessa/pneumothorax-cxr",  trust_remote_code=True).eval().to(device))

    demo.launch(share=False, debug=True, allowed_paths=["examples"],server_name="0.0.0.0", server_port=7860)
