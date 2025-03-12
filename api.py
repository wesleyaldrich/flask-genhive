import os
import numpy as np
import librosa

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.image import resize # type: ignore

MODEL_FOLDER = os.path.join(os.getcwd(), 'model')

def genhive(file_path):
    # get the model
    model_path = os.path.join(MODEL_FOLDER, "GenHive_model_tuned.h5")
    model = tf.keras.models.load_model(model_path)

    # define genres
    classes = ['Alternative/Indie',
               'EDM', 'J-Pop', 'K-Pop',
               'Pop', 'RNB', 'Rock',
               'Trap/Hip-Hop']
    
    # preprocess the target file
    X_test = load_and_preprocess_data(file_path)

    # predict the genre
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)

    unique_elements, counts = np.unique(predicted_categories, return_counts=True)

    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]

    # Calculate percentage of each genre
    total_predictions = len(predicted_categories)
    genre_percentages = {classes[element]: (count / total_predictions) * 100 
                         for element, count in zip(unique_elements, counts)}

    # return the predicted genre
    return classes[max_elements[0]], genre_percentages

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 2  # seconds
    overlap_duration = 1  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples)
                             / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        chunk = audio_data[start:end]

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)
