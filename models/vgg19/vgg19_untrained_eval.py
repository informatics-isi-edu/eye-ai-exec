import argparse
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, 
                             f1_score, precision_score, recall_score, accuracy_score, 
                             balanced_accuracy_score, matthews_corrcoef)

def create_untrained_model():
    base_model = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def evaluate_untrained_model(test_data_path, output_dir):
    model = create_untrained_model()
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)
    
    classes = {'2SKC_No_Glaucoma': 0, '2SKA_Suspected_Glaucoma': 1}
    
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=classes,
        shuffle=False
    )

    # Initialize lists to store scores, true labels, and predicted labels
    scores = []
    y_true = []
    y_pred = []

    # Iterate over all batches in the test generator
    for i in range(len(test_generator)):
        # Get a batch of data
        image_batch, label_batch = test_generator[i]
        
        # Make predictions
        predictions = model.predict_on_batch(image_batch).flatten()
        
        # Compute the scores
        scores.extend(predictions)
        
        # Binarize the predictions
        binary_predictions = tf.where(predictions < 0.5, 0, 1)
        
        # Extend the true labels and predicted labels lists
        y_true.extend(label_batch)
        y_pred.extend(binary_predictions.numpy())

    # Get file names
    filenames = test_generator.filenames

    # Write results to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "untrained_vgg19_predictions.csv"
    
    with output_file.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Score'])
        
        for filename, true_label, pred_label, score in zip(filenames, y_true, y_pred, scores):
            writer.writerow([filename, true_label, pred_label, score])
    
    logging.info(f"Predictions saved to {output_file}")

    # Calculate metrics using scikit-learn
    sklearn_roc_auc = roc_auc_score(y_true, scores)
    sklearn_f1_score = f1_score(y_true, y_pred, average='macro')
    sklearn_f1_score_normal = f1_score(y_true, y_pred)
    sklearn_precision = precision_score(y_true, y_pred)
    sklearn_recall = recall_score(y_true, y_pred)
    sklearn_accuracy = accuracy_score(y_true, y_pred)
    sklearn_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    sklearn_matthews_corrcoef = matthews_corrcoef(y_true, y_pred)

    print(f'\nScikit-learn Metrics:')
    print(f'ROC AUC: {sklearn_roc_auc}')
    print(f'F1 Score: {sklearn_f1_score}')
    print(f'F1 Score Normal: {sklearn_f1_score_normal}')
    print(f'Precision: {sklearn_precision}')
    print(f'Recall: {sklearn_recall}')
    print(f'Accuracy: {sklearn_accuracy}')
    print(f'Balanced Accuracy: {sklearn_balanced_accuracy}')
    print(f'Matthews correlation coefficient: {sklearn_matthews_corrcoef}')

    # Generate a classification report
    print('\nClassification Report:\n', classification_report(y_true, y_pred))

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()

    logging.info(f"ROC curve saved to {output_dir / 'roc_curve.png'}")

    return str(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the predictions CSV and ROC curve')
    args = parser.parse_args()

    evaluate_untrained_model(args.test_data_path, args.output_dir)
