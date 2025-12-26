import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float32)

def scale_table(Q, quality):
    if quality < 1:
        quality = 1
    if quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    Q_scaled = np.floor((Q * scale + 50) / 100)
    Q_scaled[Q_scaled < 1] = 1

    return Q_scaled

def jpeg_compression(image, quality):
    h, w = image.shape
    compressed = np.zeros((h, w), dtype=np.float32)
    Q = scale_table(quantization_table, quality)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8]

            if block.shape != (8, 8):
                continue

            block = block - 128
            dct_block = cv2.dct(block)

            quantized = np.round(dct_block / Q)
            dequantized = quantized * Q

            idct_block = cv2.idct(dequantized) + 128
            compressed[i:i+8, j:j+8] = idct_block

    compressed = np.clip(compressed, 0, 255)
    return compressed

def evaluate_metrics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)
    return mse, psnr

# User Interface

st.set_page_config(page_title="JPEG Image Compression", layout="wide")
st.title("🖼️ Image Compression menggunakan JPEG")

uploaded_file = st.file_uploader(
    "Upload Gambar (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image, dtype=np.float32)

    st.sidebar.header("⚙️ Quality Settings")
    quality = st.sidebar.slider(
        "JPEG Quality",
        min_value=1,
        max_value=100,
        value=50
    )

    compressed = jpeg_compression(image_np, quality)
    mse, psnr = evaluate_metrics(image_np, compressed)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np.astype(np.uint8), clamp=True)

    with col2:
        st.subheader(f"Compressed Image (Quality = {quality})")
        st.image(compressed.astype(np.uint8), clamp=True)

    st.markdown("---")
    st.subheader("📊 Evaluasi Kualitas")

    col3, col4 = st.columns(2)
    col3.metric("MSE", f"{mse:.2f}")
    col4.metric("PSNR (dB)", f"{psnr:.2f}")

    # Download result
    result_image = Image.fromarray(compressed.astype(np.uint8))
    buf = io.BytesIO()
    result_image.save(buf, format="JPEG")

    st.download_button(
        label="⬇️ Download Gambar",
        data=buf.getvalue(),
        file_name="compressed_jpeg.jpg",
        mime="image/jpeg"
    )

else:
    st.info("Silakan upload gambar...")