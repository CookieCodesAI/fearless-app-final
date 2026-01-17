import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import sounddevice as sd

def get_labels(info):
    label_names = info.features['label'].names
    return label_names

def get_data():
    dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True, download=True, data_dir="./data")
    train = dataset['train']
    test = dataset['test']
    val = dataset['validation']
    return train, test, val , info

def record_audio(seconds=2):
    print("Listening...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()

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

model = tf.keras.models.load_model("models/speech_cnn.keras")

_,_,_, info = get_data()

labels = get_labels(info)

SAMPLE_RATE = 16000

key = ["up", "down", "left", "right"]
vocal = [0,0,0,0]
count = 0
for i in range(4):
    audio = record_audio()
    spectrogram = preprocess_live_audio(audio)
    logits = model.predict(spectrogram)
    pred_id = tf.argmax(logits, axis=-1).numpy()[0]
    vocal[i]=labels[pred_id]
    if vocal[i] == key[i]:
        count=count+1
    print("Predicted:", labels[pred_id])
    probs = tf.nn.softmax(logits)
    confidence = probs[0, pred_id]
    print(f"Confidence: {confidence:.2f}")
if count >= 3:
    print("DANGER HELP NEEDED SENDING SOS")
else:
    print("NO SOS DETECTED")

