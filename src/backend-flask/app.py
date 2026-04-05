from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from pydub import AudioSegment
from voice import get_labels, preprocess_live_audio
import numpy as np


app = Flask(__name__)
CORS(app)

SAMPLE_RATE = 16000
labels = get_labels()
key = ["down", "up", "go", "left"]

#model = tf.keras.models.load_model("./../../models/speech_cnn.keras")
interpreter = tf.lite.Interpreter(model_path = "./../../models/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

audio = AudioSegment.from_file("../../public/Down4.m4a", format="m4a")
audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0 

audio_buffer = np.array([], dtype=np.float32)
curr = 0
count = 0

def predict_tflite(spectrogram):
    if hasattr(spectrogram, "numpy"):
        spectrogram = spectrogram.numpy()
    expected_frames = input_details[0]['shape'][1]
    current_frames = spectrogram.shape[1]
    if current_frames < expected_frames:
        pad_width = expected_frames - current_frames
        spectrogram = np.pad(
            spectrogram, 
        ((0,0), (0,pad_width),(0,0),(0,0)),
        mode = "constant",
        constant_values = 0
        )
    interpreter.set_tensor(input_details[0]["index"], spectrogram.astype(np.float32))
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])
    return logits

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1, keepdims=True)

def process_audio_chunk(audio_chunk):
    global audio_buffer, curr

    if audio_chunk is None or len(audio_chunk) == 0:
        return {"status": "bad chunk"}

    audio_buffer = np.concatenate([audio_buffer, audio_chunk])

    if len(audio_buffer) < SAMPLE_RATE:
        return {"status": "buffering"}

    window = audio_buffer[:SAMPLE_RATE]
    HOP = 16000
    audio_buffer = audio_buffer[HOP:]

    spectrogram = preprocess_live_audio(window)
    logits = predict_tflite(spectrogram)
    pred_id = int(np.argmax(logits, axis=-1)[0])
    prediction = labels[pred_id]
    probs = softmax(logits)
    confidence = float(probs[0, pred_id])

    if labels[pred_id] == key[curr]:
        curr += 1
        if curr >= len(key):
            curr = 0
            return {
                "prediction": prediction,
                "confidence": confidence,
                "status": "SOS DETECTED SENDING HELP"
            }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "status": "NO SOS DETECTED"
    }

@app.route("/predict", methods=["POST"])
def predict():
    raw_data = request.data
    if not raw_data:
        return jsonify({"error": "No data"}), 400
    audio_chunk = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = process_audio_chunk(audio_chunk)
    return jsonify(result)

@app.route("/test_file", methods=["GET"])
def test_file():
    global curr, count
    curr = 0
    count = 0
    audio_buffer = samples.copy() 
    results = []
    HOP = 8000

    while len(audio_buffer) >= SAMPLE_RATE:
        window = audio_buffer[:SAMPLE_RATE]
        audio_buffer = audio_buffer[HOP:]

        spectrogram = preprocess_live_audio(window)
        logits = predict_tflite(spectrogram)
        pred_id = int(np.argmax(logits, axis=-1).numpy()[0])
        prediction = labels[pred_id]
        probs = softmax(logits)
        confidence = float(probs[0, pred_id])

        status = "NO SOS DETECTED"
        if labels[pred_id] == key[curr]:
            curr += 1
            if curr >= len(key):
                curr = 0
                status = "SOS DETECTED SENDING HELP"

        results.append({
            "prediction": prediction,
            "confidence": confidence,
            "status": status
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)