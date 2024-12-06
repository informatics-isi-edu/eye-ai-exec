import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize

def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def f1_score_metric(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)
    return tf.py_function(lambda yt, yp: f1_score(yt, yp, average='macro'), [y_true_classes, y_pred_classes], tf.float64)

def evaluate_model(model_path, test_path, output_path, best_params):
    # Load the model
    custom_objects = {'f1_score_metric': f1_score_metric}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Prepare the test data generator
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'2SK6': 0, '2SK4': 1, '2SK8': 2}

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=best_params['batch_size'],
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )

    # Evaluate the model
    results = model.evaluate(test_generator)
    result_names = ['loss', 'roc_auc_score', 'accuracy_score', 'f1_score_metric']
    for name, value in zip(result_names, results):
        print(f"{name}: {value}")
        logging.info(f"{name}: {value}")

    # Make predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())

    # Calculate detailed metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision_macro = precision_score(y_true, y_pred_classes, average='macro')
    recall_macro = recall_score(y_true, y_pred_classes, average='macro')
    f1_macro = f1_score(y_true, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_true, y_pred_classes, average='weighted')

    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Macro Precision: {precision_macro}")
    print(f"Macro Recall: {recall_macro}")
    print(f"Macro F1 Score: {f1_macro}")
    print(f"Weighted F1 Score: {f1_weighted}")

    logging.info("\nDetailed Metrics:")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Macro Precision: {precision_macro}")
    logging.info(f"Macro Recall: {recall_macro}")
    logging.info(f"Macro F1 Score: {f1_macro}")
    logging.info(f"Weighted F1 Score: {f1_weighted}")

    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    logging.info("Classification Report:")
    logging.info(json.dumps(report, indent=2))

    # Compute macro-averaged ROC AUC
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
    macro_roc_auc = roc_auc_score(y_true_bin, y_pred, multi_class='ovr', average='macro')
    print(f"Macro-averaged ROC AUC: {macro_roc_auc}")
    logging.info(f"Macro-averaged ROC AUC: {macro_roc_auc}")

    # Save predictions to CSV
    filenames = test_generator.filenames
    with open(Path(output_path) / 'angle_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Scores'])
        for i in range(len(filenames)):
            writer.writerow([filenames[i], y_true[i], y_pred_classes[i], y_pred[i]])

    print("Predictions saved to angle_predictions.csv")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8), dpi=300)  # Set DPI to 300
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes.keys(), rotation=45)
    plt.yticks(tick_marks, classes.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(Path(output_path) / 'confusion_matrix.png', dpi=300)  # Save at 300 DPI
    plt.close()

    print("Confusion matrix saved as confusion_matrix.png (300 DPI)")

    # Plot ROC curves
    plt.figure(figsize=(10, 8), dpi=300)  # Set DPI to 300
    for i, class_name in enumerate(classes.keys()):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(Path(output_path) / 'roc_curves.png', dpi=300)  # Save at 300 DPI
    plt.close()

    print("ROC curves saved as roc_curves.png (300 DPI)")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--output_path', type=str, required=True, help='Path where evaluation results should be saved')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load best parameters from JSON
    with open(args.best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)

    evaluate_model(args.model_path, args.test_path, args.output_path, best_params)
