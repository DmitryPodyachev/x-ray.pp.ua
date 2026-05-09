import gradio as gr
from PIL import ImageFilter, Image, ImageOps
from gradio_imageslider import ImageSlider

def neg_image(input_img, neg_intensity):
    if input_img is None:
        return None
    # Конвертируем PIL Image в RGB (на случай RGBA)
    img = input_img.convert("RGB")    
    # Создаем полную инверсию
    inverted_img = ImageOps.invert(img)
    return inverted_img

def img_to_slider(im):
    if not im:
        return im
    return (im, neg_image(im,90))


with gr.Blocks() as demo:
    gr.Markdown("## img to img slider")
    with gr.Row():
        img1 = gr.Image(label="SRC image", type="pil")
        img2 = gr.ImageSlider(label="NEGATIVE image", type="pil")
    btn = gr.Button("Process")
    btn.click(img_to_slider, inputs=img1, outputs=img2)

if __name__ == "__main__":
  demo.launch(share=False, debug=True, allowed_paths=["examples"],server_name="0.0.0.0", server_port=7860)
