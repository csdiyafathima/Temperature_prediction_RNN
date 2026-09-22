import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("temperature_prediction_rnn.keras")

st.set_page_config(
    page_title="Machine Temperature Predictor",
    page_icon="🌡️",
    layout="centered"
)

st.title("🌡️ Machine Temperature Predictor")
st.write("Enter the previous two machine readings.")

st.subheader("Previous Timestamp 1")

temperature_1 = st.number_input(
    "Temperature 1 (°C)",
    value=60.0,
    step=0.1
)

vibration_1 = st.number_input(
    "Vibration 1",
    value=2.1,
    step=0.1
)

st.subheader("Previous Timestamp 2")

temperature_2 = st.number_input(
    "Temperature 2 (°C)",
    value=62.0,
    step=0.1
)

vibration_2 = st.number_input(
    "Vibration 2",
    value=2.3,
    step=0.1
)

if st.button("Predict Next Temperature"):

    input_data = np.array([
        [
            [temperature_1, vibration_1],
            [temperature_2, vibration_2]
        ]
    ], dtype=np.float32)

    prediction = model.predict(input_data, verbose=0)

    predicted_temperature = float(prediction[0][0])

    st.success(
        f"🌡️ Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
