from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from voice import get_labels, preprocess_live_audio, decode_chunk
import time
import numpy as np


app = Flask(__name__)
cors= CORS(app, resources={r"/*": {"origins": "*"}})


model = tf.keras.models.load_model("./../../models/speech_cnn.keras")
labels = get_labels()
SAMPLE_RATE = 16000
audio_buffer = np.array([],dtype=np.float32)
key = ["go", "down", "off", "right"]
curr=0
count = 0

@app.route("/predict", methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = app.make_response("")
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response
    if request.method == 'POST':
        global curr
        print("FILES:", request.files)
        print("FORM:", request.form)

        if "chunk" not in request.files:
            return jsonify({"error": "No chunk"}), 400

        chunk = request.files["chunk"]

        print("Filename:", chunk.filename)
        print("Content-Type:", chunk.content_type)

        chunk_bytes = chunk.read()
        print("Bytes length:", len(chunk_bytes))
        print("Received request!", request.files) 
        global audio_buffer, curr, count

        audio_chunk = decode_chunk(chunk_bytes)
        audio_buffer = np.concatenate([audio_buffer,audio_chunk])

        if (len(audio_buffer) < SAMPLE_RATE):
            return jsonify({
                "status" : "buffering"
            })
        window = audio_buffer[:SAMPLE_RATE]
        HOP = 8000 
        audio_buffer = audio_buffer[HOP:]

        spectrogram = preprocess_live_audio(window)
        logits = model.predict(spectrogram, verbose = 0)
        pred_id = tf.argmax(logits, axis=-1).numpy()[0]
        prediction = labels[pred_id]
        probs = tf.nn.softmax(logits)
        confidence = float(probs[0, pred_id])

        if labels[pred_id] == key[curr]:
            curr += 1
            if curr >= len(key):
                curr = 0
                return jsonify({
                    "prediction": prediction,
                    "confidence": confidence,
                    "status": "SOS DETECTED SENDING HELP"
                })
        return jsonify({
            "prediction" : prediction,
            "confidence" : confidence,
            "status" : "NO SOS DETECTED"
        })

if __name__ == "__main__":
    app.run(debug=True, port=8080)