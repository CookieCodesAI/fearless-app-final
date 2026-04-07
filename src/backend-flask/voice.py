import tensorflow as tf
import numpy as np

def get_labels():
    with open("label.labels.txt", "r") as f:
        labels = [line.strip() for line in f]
    return labels

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
