import sys
import argparse
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import csv
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve

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

def predict_and_evaluate(model_path, image_path, output_dir, best_hyperparameters_json_path):
    # Load best parameters from JSON
    with open(best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)
    
    # Prepare the data generator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'692J': 0, '690J': 1}  # 692J: Bad, 690J: Good
    class_names = list(classes.keys())
    
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_score_normal': f1_score_normal})

    generator = datagen.flow_from_directory(
        image_path,
        target_size=(224, 224),
        batch_size=best_params['batch_size'],
        class_mode='binary',
        classes=classes,
        shuffle=False
    )

    # Make predictions
    y_pred = model.predict(generator)
    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = generator.classes

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    roc_auc = roc_auc_score(y_true, y_pred)

    # Print and log metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    logging.info("\nEvaluation Metrics:")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC AUC: {roc_auc}")

    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    logging.info("Classification Report:")
    logging.info(json.dumps(report, indent=2))

    # Save predictions to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'quality_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Score'])
        for i, filename in enumerate(generator.filenames):
            writer.writerow([filename, class_names[y_true[i]], class_names[y_pred_classes[i][0]], y_pred[i][0]])

    print("Predictions saved to quality_predictions.csv")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()

    print("Confusion matrix saved as confusion_matrix.png")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'roc_curve.png', dpi=300)
    plt.close()

    print("ROC curve saved as roc_curve.png")

    return str(output_dir / "quality_predictions.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the prediction model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the images for prediction')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sys.exit(predict_and_evaluate(args.model_path, args.image_path, args.output_dir, args.best_hyperparameters_json_path))
