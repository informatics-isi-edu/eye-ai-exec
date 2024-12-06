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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize

@keras.saving.register_keras_serializable()
def f1_score_normal(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def plot_confusion_matrix(y_true, y_pred, classes, output_path, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(Path(output_path) / f'{model_name}_confusion_matrix.png', dpi=300)
    plt.close()
    logging.info(f"Confusion matrix saved as {model_name}_confusion_matrix.png (300 DPI)")

def plot_roc_curves(y_true, y_pred, classes, output_path, model_name):
    plt.figure(figsize=(10, 8), dpi=300)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(Path(output_path) / f'{model_name}_roc_curve.png', dpi=300)
    plt.close()
    logging.info(f"ROC curve saved as {model_name}_roc_curve.png (300 DPI)")

def prediction(model_path, cropped_image_path, output_dir, best_hyperparameters_json_path):
    with open(best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)
    
    model_name = Path(model_path).stem
    
    graded_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'2SKC_No_Glaucoma': 0, '2SKA_Suspected_Glaucoma': 1}
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_score_normal': f1_score_normal})

    graded_test_generator = graded_test_datagen.flow_from_directory(
        cropped_image_path,
        target_size=(224, 224),
        batch_size=best_params['batch_size'],
        class_mode='binary',
        classes=classes,
        shuffle=False
    )

    filenames = graded_test_generator.filenames
    y_true = graded_test_generator.classes
    y_pred = model.predict(graded_test_generator).flatten()  # Flatten the predictions
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    roc_auc = roc_auc_score(y_true, y_pred)

    # Print and log metrics
    print(f"\nMetrics for {model_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    logging.info(f"\nMetrics for {model_name}:")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC AUC: {roc_auc}")

    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=classes.keys(), output_dict=True)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred_classes, target_names=classes.keys()))
    logging.info(f"Classification Report for {model_name}:")
    logging.info(json.dumps(report, indent=2))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, classes.keys(), output_dir, model_name)

    # Plot ROC curve
    plot_roc_curves(y_true, y_pred, classes.keys(), output_dir, model_name)

    # Write to CSV file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = f"{model_name}_predictions_results.csv"
    with open(output_dir / csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Score'])

        for i in range(len(filenames)):
            writer.writerow([filenames[i], y_true[i], y_pred_classes[i], y_pred[i]])

    logging.info(f"Data saved to {csv_filename}")

    # Save metrics to JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "classification_report": report
    }
    metrics_filename = f"{model_name}_metrics.json"
    with open(output_dir / metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_filename}")

    return str(output_dir / csv_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the prediction model')
    parser.add_argument('--cropped_image_path', type=str, required=True, help='Path to the cropped images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sys.exit(prediction(args.model_path,
                        args.cropped_image_path,
                        args.output_dir,
                        args.best_hyperparameters_json_path))
