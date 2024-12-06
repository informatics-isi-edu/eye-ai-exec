import sys
import argparse
import numpy as np
import random
import os
import gc
from pathlib import Path, PurePath
import logging
import json

import pandas as pd
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, f1_score

import keras
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

def set_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def get_data_generators(train_path, valid_path, test_path, best_params):
    # Data generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19,
        rotation_range=best_params['rotation_range'],
        width_shift_range=best_params['width_shift_range'],
        height_shift_range=best_params['height_shift_range'],
        # horizontal_flip=best_params['horizontal_flip'], #flips are not required for angle prediction
        # vertical_flip=best_params['vertical_flip'],
        zoom_range=[1 + best_params['zoom_range'], 1 - best_params['zoom_range']],
        brightness_range=[1 - best_params['brightness_range'], 1 + best_params['brightness_range']] if best_params['brightness_range'] != 0 else None
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    classes = {'2SK6': 0, '2SK4': 1, '2SK8': 2}

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        class_mode='categorical',
        classes=classes
    )
    
    validation_generator = val_datagen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        class_mode='categorical',
        classes=classes
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        class_mode='categorical',
        classes=classes
    )
    
    print("train_generator.class_indices : ", train_generator.class_indices)
    print("validation_generator.class_indices : ", validation_generator.class_indices)
    print("test_generator.class_indices : ", test_generator.class_indices)
    
    return train_generator, validation_generator, test_generator

def f1_score_metric(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)
    return tf.py_function(lambda yt, yp: f1_score(yt, yp, average='macro'), [y_true_classes, y_pred_classes], tf.float64)

def train_and_evaluate(train_path, valid_path, test_path, output_path, best_params, model_name):
    set_seeds()

    train_generator, validation_generator, test_generator = get_data_generators(train_path, valid_path, test_path, best_params)
    
    # Model building
    K.clear_session()  # Clear session
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    
    # GlobalAveragePooling2D or Flatten based on best_params
    if best_params['pooling'] == 'global_average':
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    
    # Add dense layers
    for i in range(best_params['dense_layers']):
        num_units = best_params[f'units_layer_{i}']
        activation = best_params[f'activation_func_{i}']
        x = Dense(num_units, activation=activation)(x)
    
        if best_params[f'batch_norm_{i}']:
            x = BatchNormalization()(x)
    
        x = Dropout(best_params[f'dropout_{i}'])(x)
    
    outputs = Dense(3, activation='softmax')(x)  # 3 units for 3 classes with softmax activation
    model = Model(inputs, outputs)
    
    # Unfreeze the base_model
    base_model.trainable = True
    for layer in base_model.layers[:best_params['fine_tune_at']]:
        layer.trainable = False
    
    # Compile model
    optimizer = Adam(learning_rate=best_params['fine_tuning_learning_rate_adam'])
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=3, name="roc_auc_score"),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy_score"),
            f1_score_metric
        ]
    )
    
    # Training
    class_weights = None
    if best_params['use_class_weights']:
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = dict(enumerate(class_weights))

    num_workers = os.cpu_count()

    training_log = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        batch_size=best_params['batch_size'],
        class_weight=class_weights,
        workers=num_workers,
        use_multiprocessing=True,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            EarlyStopping(monitor='val_accuracy_score', mode='max', verbose=1, patience=8, restore_best_weights=True),
            EarlyStopping(monitor='val_f1_score_metric', mode='max', verbose=1, patience=8, restore_best_weights=True),
        ],
    )

    # Evaluate the model on the test set
    results = model.evaluate(test_generator)
    logging.info(f"Test results - {results}")
    print(f"Model Eval results: {results}")

    # Detailed evaluation metrics
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())
    
    # Calculate and log detailed metrics
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    logging.info("Detailed Classification Report:")
    logging.info(json.dumps(report, indent=2))

    # Calculate and log multi-class F1 score
    f1_macro = f1_score(y_true, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_true, y_pred_classes, average='weighted')
    logging.info(f"Macro F1 Score: {f1_macro}")
    logging.info(f"Weighted F1 Score: {f1_weighted}")

    if model_name:
        model.save(os.path.join(output_path, f'{model_name}.h5'))
    else:
        model.save(os.path.join(output_path, 'VGG19_Catalog_LAC_DHS_Angle_Trained_model.h5'))
    
    # Convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(training_log.history) 
    hist_df.to_csv(os.path.join(output_path, f'training_history_{model_name}.csv'), index=False)
    
    logging.info(f"{model_name} Model trained, Model and training history are saved successfully.")

def main(train_path, valid_path, test_path, output_path, best_hyperparameters_json_path, model_name):
    logging.basicConfig(level=logging.INFO)
    
    # Load best parameters from JSON
    with open(best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)

    train_and_evaluate(train_path, valid_path, test_path, output_path, best_params, model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training images')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the validation images')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the trained model should be saved')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the Trained model with best hyperparameters')
    args = parser.parse_args()

    main(args.train_path, args.valid_path, args.test_path, args.output_path, args.best_hyperparameters_json_path, args.model_name)
