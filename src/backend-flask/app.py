from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from pydub import AudioSegment
from voice import get_labels, preprocess_live_audio, decode_chunk
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------
# Config / Globals
# ----------------------------
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from pydub import AudioSegment
from voice import get_labels, preprocess_live_audio, decode_chunk
import numpy as np

app = Flask(__name__)
CORS(app)

SAMPLE_RATE = 16000
labels = get_labels()
key = ["down", "down", "down", "down"]

model = tf.keras.models.load_model("./../../models/speech_cnn.keras")

audio = AudioSegment.from_file("../../public/Down4.m4a", format="m4a")
audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0 

audio_buffer = np.array([], dtype=np.float32)
curr = 0
count = 0


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
    logits = model.predict(spectrogram, verbose=0)
    pred_id = tf.argmax(logits, axis=-1).numpy()[0]
    prediction = labels[pred_id]
    probs = tf.nn.softmax(logits)
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


@app.route("/predict", methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = app.make_response("")
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    chunk = request.files.get("chunk")
    if not chunk:
        return jsonify({"error": "No chunk"}), 400

    chunk_bytes = chunk.read()
    audio_chunk = decode_chunk(chunk_bytes)

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
        logits = model.predict(spectrogram, verbose=0)
        pred_id = tf.argmax(logits, axis=-1).numpy()[0]
        prediction = labels[pred_id]
        probs = tf.nn.softmax(logits)
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