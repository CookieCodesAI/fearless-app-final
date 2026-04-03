import tensorflow as tf
import numpy as np
import soundfile as sf
import subprocess

def get_labels():
    with open("label.labels.txt", "r") as f:
        labels = [line.strip() for line in f]
    return labels

'''def record_audio(seconds=2):
    print("Listening...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()'''

def preprocess_live_audio(audio):

    audio = audio[:16000]
    audio = np.pad(audio, (0, max(0, 16000 - len(audio))))

    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    spectrogram = tf.signal.stft(
        audio,
        frame_length=640,
        frame_step=160,
        fft_length=640
    )
    spectrogram = tf.abs(spectrogram)

    spectrogram = spectrogram[:, :129]

    spectrogram = (spectrogram - tf.reduce_mean(spectrogram)) / (
        tf.math.reduce_std(spectrogram) + 1e-6
    )

    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.expand_dims(spectrogram, 0)

    return spectrogram

model = tf.keras.models.load_model("./../../models/speech_cnn.keras")

labels = get_labels()

SAMPLE_RATE = 16000

key = ["go", "down", "off", "right"]
curr = 0
status = False
count = 0
'''while not status:
    audio = record_audio()
    spectrogram = preprocess_live_audio(audio)
    logits = model.predict(spectrogram)
    pred_id = tf.argmax(logits, axis=-1).numpy()[0]
    if labels[pred_id] == key[curr]:
        count=count+1
        curr+=1
    print("Predicted:", labels[pred_id])
    probs = tf.nn.softmax(logits)
    confidence = probs[0, pred_id]
    print(f"Confidence: {confidence:.2f}")
    if (count==4):
        status = True
    else:
        print("NO SOS DETECTED")
    time.sleep(0.5)
print("SOS DETECTED SENDING HELP")

'''