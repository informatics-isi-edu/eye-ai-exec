import argparse

from functools import partial
import json

import numpy as np
import optuna
import keras
import tensorflow as tf

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, Hinge, SquaredHinge, LogCosh
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
import os
from sklearn.utils import class_weight
import gc
import random
import tensorflow_addons as tfa


# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(42)

# Set the built-in python random seed
random.seed(42)

# Set the TensorFlow random seed
tf.random.set_seed(42)

num_workers = os.cpu_count()

def f1_score_normal(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def f1_score_macro(y_true, y_pred):
    def f1(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    f1_pos = f1(y_true, y_pred)
    f1_neg = f1(1 - y_true, 1 - y_pred)
    return (f1_pos + f1_neg) / 2

import tensorflow as tf

def matthews_correlation(y_true, y_pred):
    y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = tf.reduce_sum(y_pos * y_pred_pos)
    tn = tf.reduce_sum(y_neg * y_pred_neg)

    fp = tf.reduce_sum(y_neg * y_pred_pos)
    fn = tf.reduce_sum(y_pos * y_pred_neg)

    numerator = (tp * tn) - (fp * fn)
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + tf.keras.backend.epsilon())

# Define data generators
def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)
  
# Define objective function for Optuna study
def objective(trial, train_path, valid_path, graded_test_path):
    # Define data augmentations to be tuned

    # Suggest values for the hyperparameters
    rotation_range = trial.suggest_int('rotation_range', -10, 10)
    width_shift_range = trial.suggest_float('width_shift_range', 0.0, 0.10)
    height_shift_range = trial.suggest_float('height_shift_range', 0.0, 0.10)
    horizontal_flip = trial.suggest_categorical('horizontal_flip', [True, False])
    vertical_flip = trial.suggest_categorical('vertical_flip', [True, False])
    zoom_range = trial.suggest_float('zoom_range', -0.05, 0.05)  # Changed to float for more precise control
    brightness_range = trial.suggest_uniform('brightness_range', -0.05, 0.05)  # Changed to suggest_uniform for a float range

    # Define data generators with tuned augmentations
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        zoom_range=[1 + zoom_range, 1 - zoom_range],  # Adjusted for correct usage
        brightness_range=[1 - brightness_range, 1 + brightness_range] if brightness_range != 0 else None
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    graded_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'2SKC_No_Glaucoma': 0, '2SKA_Suspected_Glaucoma': 1}

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        class_mode = 'binary',
        classes = classes
        # class_mode='binary'
    )

    validation_generator = val_datagen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        class_mode = 'binary',
        classes = classes
        # class_mode='binary'
    )

    # test_generator = test_datagen.flow_from_directory(
    #     '/content/Test_2-277C_Field_2_Cropped_to_Optic_Disc_Test_Ready_without_2_277M_without_no_optic_disc_images/',
    #     target_size=(224, 224),
    #     class_mode = 'binary',
    #     classes = classes

    #     # class_mode='binary'
    # )

    graded_test_generator = graded_test_datagen.flow_from_directory(
        graded_test_path,
        target_size=(224, 224),
        class_mode = 'binary',
        classes = classes

        # class_mode='binary'
    )


    print(train_generator.class_indices)
    print(graded_test_generator.class_indices)
    # Add this after creating your generators
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    class_weights = None
    if use_class_weights:
        # Compute class weights
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(train_generator.classes),
                                                          y=train_generator.classes)
        class_weights = dict(enumerate(class_weights))

    # Load pre-trained VGG16
    K.clear_session()  # Clear session to avoid memory issue
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base_model

    # Define new model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)  # We need to set training=True for the base_model to take into account the unfreezing

    # Choose between Flatten and GlobalAveragePooling2D
    if trial.suggest_categorical('pooling', ['flatten', 'global_average']):
        x = Flatten()(x)
    else:
        x = GlobalAveragePooling2D()(x)

    # Add dense layers with choice of activation functions
    activation_functions = ['relu', 'sigmoid', 'tanh', 'elu', ]
    num_dense_layers = trial.suggest_int('dense_layers', 1, 3)

    for i in range(num_dense_layers):
        num_units = trial.suggest_categorical('units_layer_{}'.format(i), [16, 32, 64, 128, 256, 512])
        x = Dense(num_units, activation=trial.suggest_categorical('activation_func_{}'.format(i), activation_functions))(x)

        # Decide whether to use Batch Normalization
        if trial.suggest_categorical('batch_norm_{}'.format(i), [True, False]):
            x = BatchNormalization()(x)

        # Set dropout rate
        x = Dropout(trial.suggest_float('dropout_{}'.format(i), 0.0, 0.5))(x)


    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)



    base_model.trainable = True  # Unfreeze the base_model. Note that it's necessary to unfreeze the base_model just before training

    # Fine-tune from this layer onwards
    fine_tune_at = trial.suggest_int('fine_tune_at', 0, len(base_model.layers))

    # Freeze all the layers before the `fine_tune_at` layer
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= fine_tune_at

    # Fine-tuning
    # Choose optimizer
    optimizer = Adam(
        learning_rate=trial.suggest_float("fine_tuning_learning_rate_adam", 1e-6, 1e-3, log=True),
    )


    loss = BinaryCrossentropy()

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[  # ROC A
        # tf.keras.metrics.AUC(curve="PR",name="pr_auc_score"),
        tf.keras.metrics.AUC(curve="ROC",name="roc_auc_score"),
        f1_score_normal,
        # f1_score_macro,
        # tf.keras.metrics.Precision(name="precision_score"),
        # tf.keras.metrics.Recall(name="recall_score"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy_score"),
        # tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2, name='mcc_score')
        # matthews_correlation
                   ]
    )

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])


    history_fine = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                   EarlyStopping(monitor='val_roc_auc_score', mode='max', verbose=1, patience=4, restore_best_weights=True),
                   ],  # Add reduce_lr to callbacks tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=1e-8)
        batch_size=batch_size,
        class_weight=class_weights,
        workers=num_workers,
        use_multiprocessing=True,
    )
    # Generate predictions using the model
    # Initialize lists to store scores, true labels, and predicted labels
    scores = []
    y_true = []
    y_pred = []

    # Iterate over all batches in the validation generator
    for i in range(len(validation_generator)):
        # Get a batch of data
        image_batch, label_batch = validation_generator[i]

        # Make predictions
        predictions = model.predict_on_batch(image_batch).flatten()

        # Compute the scores
        scores.extend(predictions)

        # Binarize the predictions
        predictions = tf.where(predictions < 0.5, 0, 1)

        # Extend the true labels and predicted labels lists
        y_true.extend(label_batch)
        y_pred.extend(predictions.numpy())

    print(f'Test Generator Values : len of test_generator: {len(validation_generator)}, len of scores: {len(scores)}, len y pred : {len(y_pred)}, len of y true : {len(y_true)}')#scor;;;;{}

    # Calculate metrics using TensorFlow
    tf_roc_auc = tf.keras.metrics.AUC(curve="ROC")(y_true, scores).numpy()
    tf_pr_auc = tf.keras.metrics.AUC(curve="PR")(y_true, scores).numpy()
        # Convert lists to tensors
    y_truef = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_predf = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # tf_f1_score_macro = f1_score_macro(y_truef, y_predf)
    tf_f1_score_normal = f1_score_normal(y_truef,y_predf)
    tf_precision = tf.keras.metrics.Precision()(y_true, y_pred).numpy()
    tf_recall = tf.keras.metrics.Recall()(y_true, y_pred).numpy()
    tf_accuracy = tf.keras.metrics.BinaryAccuracy()(y_true, y_pred).numpy()
    # tf_mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)(y_true, y_pred).numpy()

    print(f'TensorFlow Metrics:')
    print(f'ROC AUC: {tf_roc_auc}')
    print(f'PR AUC: {tf_pr_auc}')
    # print(f'F1 Score: {tf_f1_score_macro}')
    print(f'F1 Score Normal: {tf_f1_score_normal}')
    print(f'Precision: {tf_precision}')
    print(f'Recall: {tf_recall}')
    print(f'Binary Accuracy: {tf_accuracy}')
    # print(f'MCC: {tf_mcc}')

    # custom_tf_mcc = matthews_correlation(y_truef, y_predf)
    # print(f'Own MCC : {custom_tf_mcc}')


    # Calculate metrics using scikit-learn
    sklearn_roc_auc = roc_auc_score(y_true, scores)
    # sklearn_f1_score_macro = f1_score(y_true, y_pred, average='macro')
    sklearn_f1_score_normal = f1_score(y_true, y_pred)
    # sklearn_f1_score_binary = f1_score(y_true, y_pred, average='binary')
    # sklearn_f1_score_weighted = f1_score(y_true, y_pred, average='weighted')

    sklearn_precision = precision_score(y_true, y_pred)
    sklearn_recall = recall_score(y_true, y_pred)
    sklearn_accuracy = accuracy_score(y_true, y_pred)
    sklearn_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    sklearn_matthews_corrcoef = matthews_corrcoef(y_true, y_pred)

    print(f'\nScikit-learn Metrics:')
    print(f'ROC AUC: {sklearn_roc_auc}')
    # print(f'F1 Score Macro: {sklearn_f1_score_macro}')
    print(f'F1 Score Normal: {sklearn_f1_score_normal}') #t
    # print(f'F1 Score binary: {sklearn_f1_score_binary}')
    # print(f'F1 Score Weighted: {sklearn_f1_score_weighted}')
    print(f'Precision: {sklearn_precision}')
    print(f'Recall: {sklearn_recall}')
    print(f'Accuracy: {sklearn_accuracy}')
    print(f'Balanced Accuracy: {sklearn_balanced_accuracy}')
    print(f'Matthews correlation coefficient: {sklearn_matthews_corrcoef}')
        # Print the classification report
    print("Classification Report:\n", classification_report(y_true, y_pred))


    del train_generator
    del validation_generator
    # del test_generator

    gc.collect()

    return sklearn_f1_score_normal #sklearn_roc_auc i


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

def main(train_path, valid_path, graded_test_path, output_path, n_trials, vgg19_hyperparameter_study_name, best_parameters_json_file_name):
    os.makedirs(output_path, exist_ok=True)
  
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(),
        study_name='vgg19_catalog_Optimization_F1_Score_Score'
    )
  
    # Partial function setup to pass additional arguments to the objective function
    objective_with_paths = partial(objective, train_path=train_path, valid_path=valid_path, graded_test_path=graded_test_path)
    
    # Optimization
    study.optimize(objective_with_paths, n_trials=n_trials, callbacks=[print_best_callback])
  
    # Save the study and the best trial parameters as JSON
    joblib.dump(study, os.path.join(output_path, f'{vgg19_hyperparameter_study_name}.pkl'))

    best_params = study.best_trial.params
    best_params_path = os.path.join(output_path, f'{best_parameters_json_file_name}.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best trial :")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--graded_test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--vgg19_hyperparameter_study_name', type=str, required=True)
    parser.add_argument('--best_parameters_json_file_name', type=str, required=True)
    args = parser.parse_args()
    main(args.train_path, args.valid_path, args.graded_test_path, args.output_path, args.n_trials)

