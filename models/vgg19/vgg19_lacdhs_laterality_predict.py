import sys
import argparse
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import csv
import logging
from pathlib import Path, PurePath
import json


@keras.saving.register_keras_serializable()
def f1_score_normal(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)


def prediction(model_path, cropped_image_path, output_dir, best_hyperparameters_json_path):
        # Load best parameters from JSON
    with open(best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)
        
    graded_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'2SK0': 0, '2SK2': 1}
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'f1_score_normal': f1_score_normal})

    graded_test_generator = graded_test_datagen.flow_from_directory(
        cropped_image_path,
        target_size=(224, 224),
        batch_size=best_params['batch_size'],
        class_mode='binary',
        classes=classes,
        shuffle=False
    )

    # Initialize lists to store file names, true labels, and predicted labels
    filenames = []
    y_true = []
    y_pred = []
    scores = []

    for i in range(len(graded_test_generator)):
        # Get a batch of data
        batch_data = graded_test_generator[i]
        image_batch, label_batch = batch_data[0], batch_data[1]
        batch_filenames = graded_test_generator.filenames[
                          i * graded_test_generator.batch_size: (i + 1) * graded_test_generator.batch_size]

        # Make predictions
        predictions = model.predict_on_batch(image_batch).flatten()

        # append bath data to lists
        scores.extend(predictions)

        # Binarize the predictions
        predictions = tf.where(predictions < 0.5, 0, 1).numpy()

        # Append batch data to lists
        filenames.extend(batch_filenames)
        y_true.extend(label_batch)
        y_pred.extend(predictions)

    # Write to CSV file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(output_dir)+"/predictions_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Score'])

        for i in range(len(filenames)):
            writer.writerow([filenames[i], y_true[i], y_pred[i], scores[i]])

    logging.info("Data saved to predictions.csv")

    return str(output_dir)+"/predictions_results.csv"


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the prediction model')
    parser.add_argument('--cropped_image_path', type=str, required=True, help='Path to the cropped images')
    parser.add_argument('--output_dir', type=str, required=False, help='Path to the output CSV')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')

    # parse the arguments
    args = parser.parse_args()

    sys.exit(prediction(args.model_path,
                        args.cropped_image_path,
                        args.output_dir,
                        args.best_hyperparameters_json_path))
