import cv2
import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F

from einops import rearrange
from gradio_imageslider import ImageSlider          # ← слайдер негатива
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
) -> np.ndarray:
    overlay = alpha * img + (1 - alpha) * mask
    return overlay.astype(np.uint8)


def make_neg_slider(overlay: np.ndarray) -> tuple:
    """
    Возвращает кортеж (оригинал, негатив) для ImageSlider.
    Инверсия: 255 - каждый канал RGB.
    """
    negative = (255 - overlay).astype(np.uint8)
    return (overlay, negative)


@spaces.GPU
def predict(radiograph_data, invert: bool = False):
    # ImageEditor возвращает dict с ключами background / layers / composite.
    # composite — снимок со всеми нарисованными слоями поверх.
    composite = radiograph_data["composite"]          # shape (H, W) или (H, W, C)
    # Приводим к grayscale независимо от числа каналов на выходе ImageEditor
    if composite.ndim == 2:
        Radiograph = composite                        # уже grayscale
    elif composite.shape[2] == 4:
        Radiograph = cv2.cvtColor(composite, cv2.COLOR_RGBA2GRAY)
    elif composite.shape[2] == 3:
        Radiograph = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
    else:
        Radiograph = composite[:, :, 0]              # берём первый канал
    # --- сохраняем оригинал до возможной инверсии ---
    original_rg = cv2.cvtColor(Radiograph, cv2.COLOR_GRAY2RGB)

    # --- предобработка изображения ---
    if invert:
        Radiograph = cv2.bitwise_not(Radiograph)  # негатив входного снимка
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

    # Каждый хитмап оборачиваем в кортеж (оригинал, негатив) для ImageSlider
    return [
        info_string,
        preds,
        make_neg_slider(original_rg),    # исходный снимок
        make_neg_slider(info_overlay),   # Heart & Lungs
        make_neg_slider(pna_overlay),    # Pneumonia
        make_neg_slider(ptx_overlay),    # Pneumothorax
    ]


# --- входные контролы ---
image = gr.ImageEditor(
    image_mode="L",
     canvas_size=(1024, 2768),  # Set custom canvas size
#     fixed_canvas=True,        # Keep canvas size fixed
#     type="pil",
    type="numpy",
    label="Загрузите снимок (можно рисовать поверх)",
    brush=gr.Brush(
        colors=["#ff0000", "#ffff00", "#00ff00"],
        default_color="#ff0000",
        default_size=5,
    ),
)
invert_checkbox = gr.Checkbox(label="⬜ Негатив (инверсия)", value=False)

# --- выходные контролы ---
info_textbox = gr.Textbox(show_label=True, label="age detection", lines=5)
labels = gr.Label(show_label=True, show_heading=True, label="progress bar")

# ImageSlider: слева — оригинальный хитмап, справа — его негатив.
# Вертикальная «шторка» двигается мышью / пальцем.
radiograph_slider = ImageSlider(label="Radiograph     ◀ original │ negative ▶", type="numpy", position=0.5)
heatmap1 = ImageSlider(label="Heart & Lungs  ◀ original │ negative ▶", type="numpy", position=0.5)
heatmap2 = ImageSlider(label="Pneumonia      ◀ original │ negative ▶", type="numpy", position=0.5)
heatmap3 = ImageSlider(label="Pneumothorax   ◀ original │ negative ▶", type="numpy", position=0.5)


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
    gr.Markdown(
        """
    # Main interface

    Upload x-ray image and Submit for analyze.  
    **Перетащите вертикальный слайдер на хитмапах**, чтобы сравнить оригинал и негатив.

        """
    )
    gr.Interface(
        fn=predict,
        inputs=[image, invert_checkbox],
        outputs=[info_textbox, labels, radiograph_slider, heatmap1, heatmap2, heatmap3],
        examples=[
            ["examples/mymom1.jpg", True],
            ["examples/cxr.png",    False],
        ],
        cache_examples=True,
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device `{device}` ...")

    cxr_info_model = (AutoModel.from_pretrained("Dimaodessa/chest-x-ray-basic", trust_remote_code=True).eval().to(device))
    pna_model =      (AutoModel.from_pretrained("Dimaodessa/pneumonia-cxr",     trust_remote_code=True).eval().to(device))
    ptx_model =      (AutoModel.from_pretrained("Dimaodessa/pneumothorax-cxr",  trust_remote_code=True).eval().to(device))

    demo.launch(share=False, debug=True, allowed_paths=["examples"], server_name="0.0.0.0", server_port=7860)
