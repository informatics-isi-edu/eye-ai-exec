import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os
import argparse

# Custom F1 score metric (needed to load the model)
@tf.keras.utils.register_keras_serializable()
def f1_score_normal(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
    return f1_val

def visualize_model_architecture(model_path, output_path):
    # Load the model
    model = load_model(model_path, custom_objects={'f1_score_normal': f1_score_normal})
    
    # Plot the model with 300 DPI
    plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True, dpi=300)
    print(f"Model architecture diagram saved to {output_path} at 300 DPI")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Keras model architecture")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved Keras model (.h5 file)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the architecture diagram")
    args = parser.parse_args()

    visualize_model_architecture(args.model_path, args.output_path)
