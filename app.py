import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "outputs/digit_cnn.h5"


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognizer - CNN",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognizer - CNN")
st.write(
    "Upload a handwritten digit image and the CNN will predict the digit."
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model, None

    except Exception as e:
        return None, f"Model exists but failed to load: {e}"


model, model_error = load_trained_model()


if model is None:

    st.error(f"❌ {model_error}")

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_digit_image(uploaded_image: Image.Image):

    """
    Convert uploaded handwritten digit image into
    MNIST-style 28x28 input for the CNN.

    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Estimate background from image borders
    4. Extract foreground digit
    5. Crop digit
    6. Resize digit to maximum 20x20
    7. Place on 28x28 black canvas
    8. Center using intensity-weighted center of mass
    9. Normalize pixels to [0,1]
    10. Reshape to (1,28,28,1)
    """

    # --------------------------------------------------------
    # 1. Convert to grayscale
    # --------------------------------------------------------

    image = uploaded_image.convert("L")


    # --------------------------------------------------------
    # 2. Gaussian blur
    # --------------------------------------------------------

    image = image.filter(
        ImageFilter.GaussianBlur(radius=0.4)
    )


    # --------------------------------------------------------
    # 3. Convert image to NumPy
    # --------------------------------------------------------

    arr = np.array(image).astype(np.float32)


    # --------------------------------------------------------
    # 4. Estimate background from border pixels
    # --------------------------------------------------------

    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ])

    background = np.median(border)


    # --------------------------------------------------------
    # 5. Extract foreground digit
    # --------------------------------------------------------

    if background > 127:

        # White/light background
        # Dark handwritten digit

        foreground = np.clip(
            background - arr,
            0,
            255
        )

        mask = foreground > max(
            20,
            background * 0.18
        )

    else:

        # Dark background
        # Bright handwritten digit

        foreground = np.clip(
            arr - background,
            0,
            255
        )

        mask = foreground > max(
            20,
            (255 - background) * 0.18
        )


    # --------------------------------------------------------
    # 6. Find digit bounding box
    # --------------------------------------------------------

    ys, xs = np.where(mask)


    if len(xs) > 20:

        x1 = xs.min()
        x2 = xs.max() + 1

        y1 = ys.min()
        y2 = ys.max() + 1


        # Add padding around digit

        margin = max(
            2,
            int(
                0.08 *
                max(
                    x2 - x1,
                    y2 - y1
                )
            )
        )


        x1 = max(
            0,
            x1 - margin
        )

        y1 = max(
            0,
            y1 - margin
        )

        x2 = min(
            foreground.shape[1],
            x2 + margin
        )

        y2 = min(
            foreground.shape[0],
            y2 + margin
        )


        digit_crop = foreground[
            y1:y2,
            x1:x2
        ]

    else:

        digit_crop = foreground


    # --------------------------------------------------------
    # 7. Convert crop to PIL image
    # --------------------------------------------------------

    digit_img = Image.fromarray(
        np.clip(
            digit_crop,
            0,
            255
        ).astype(np.uint8),
        "L"
    )


    # --------------------------------------------------------
    # 8. Resize to maximum side 20
    # --------------------------------------------------------

    max_side = max(
        digit_img.size
    )


    if max_side > 0:

        scale = 20.0 / max_side

        new_w = max(
            1,
            int(
                digit_img.width *
                scale
            )
        )

        new_h = max(
            1,
            int(
                digit_img.height *
                scale
            )
        )

        digit_img = digit_img.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )


    # --------------------------------------------------------
    # 9. Place digit on 28x28 canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )


    paste_x = (
        28 - digit_img.width
    ) // 2

    paste_y = (
        28 - digit_img.height
    ) // 2


    canvas.paste(
        digit_img,
        (paste_x, paste_y)
    )


    # --------------------------------------------------------
    # 10. Center digit using center of mass
    # --------------------------------------------------------

    a = np.array(
        canvas
    ).astype(np.float32)


    total_intensity = a.sum()


    if total_intensity > 0:

        yy, xx = np.indices(
            a.shape
        )


        cx = (
            (xx * a).sum()
            / total_intensity
        )

        cy = (
            (yy * a).sum()
            / total_intensity
        )


        dx = int(
            round(13.5 - cx)
        )

        dy = int(
            round(13.5 - cy)
        )


        centered_canvas = Image.new(
            "L",
            (28, 28),
            0
        )


        centered_canvas.paste(
            canvas,
            (dx, dy)
        )


        canvas = centered_canvas


    # --------------------------------------------------------
    # 11. Normalize to [0,1]
    # --------------------------------------------------------

    final_arr = (
        np.array(canvas)
        .astype("float32")
        / 255.0
    )


    # --------------------------------------------------------
    # 12. Reshape for CNN
    # --------------------------------------------------------

    input_tensor = final_arr.reshape(
        1,
        28,
        28,
        1
    )


    return canvas, input_tensor


# ============================================================
# USER INTERFACE
# ============================================================

st.markdown("---")

st.header("📸 Upload Image")


uploaded_file = st.file_uploader(
    "Choose a handwritten digit image (PNG, JPG, JPEG)",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    original_img = Image.open(
        uploaded_file
    )


    # --------------------------------------------------------
    # Two-column image display
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "🖼️ Original Image"
        )

        st.image(
            original_img,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    processed_canvas, input_array = (
        preprocess_digit_image(
            original_img
        )
    )


    with col2:

        st.subheader(
            "🖼️ Preprocessed Image"
        )

        st.caption(
            "Reshaped & Centered to 28x28 for CNN input"
        )

        st.image(
            processed_canvas.resize(
                (280, 280),
                Image.Resampling.NEAREST
            ),
            use_container_width=True
        )


    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        input_array,
        verbose=0
    )[0]


    predicted_digit = int(
        np.argmax(probabilities)
    )


    confidence_pct = (
        float(
            probabilities[
                predicted_digit
            ]
        )
        * 100.0
    )


    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    st.markdown("---")

    st.header(
        "🎯 Prediction Result"
    )


    res_col1, res_col2 = st.columns(2)


    with res_col1:

        st.success(
            f"### Predicted Digit: **{predicted_digit}**"
        )


    with res_col2:

        st.metric(
            label="Confidence",
            value=f"{confidence_pct:.2f}%"
        )


    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    st.markdown("---")

    st.header(
        "📊 Class Probabilities"
    )


    prob_dict = {
        f"Digit {i}":
        float(probabilities[i])
        for i in range(10)
    }


    st.bar_chart(
        prob_dict
    )


    st.info(
        "💡 **Note:** The confidence score represents "
        "the Softmax output probability of the CNN for "
        "each digit (0-9)."
    )


else:

    st.info(
        "👆 Please upload a handwritten digit image "
        "above to generate predictions."
    )