import sys
import argparse
import numpy as np
import random
import os
import gc
from pathlib import Path, PurePath
import logging
import json
import csv
import ast

import pandas as pd
from sklearn.utils import class_weight

import keras
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, Hinge, SquaredHinge, LogCosh
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef


def set_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

@keras.saving.register_keras_serializable()
def f1_score_normal(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def metrics_score(y_true, y_pred):
    f1_val = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return f1_val, recall, precision, accuracy


def preprocess_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def get_data_generators(train_path, valid_path, test_path, best_params, classes = {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}):
    train_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input_vgg19,
        rotation_range=best_params['rotation_range'],
        width_shift_range=best_params['width_shift_range'],
        height_shift_range=best_params['height_shift_range'],
        horizontal_flip=best_params['horizontal_flip'],
        vertical_flip=best_params['vertical_flip'],
        zoom_range=[1 + best_params['zoom_range'], 1 - best_params['zoom_range']],
        brightness_range=[1 - best_params['brightness_range'], 1 + best_params['brightness_range']] if best_params['brightness_range'] != 0 else None
    )
    
    val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)
    
    test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes
    )

    validation_generator = val_datagen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes
    )

    print("train path: ", train_path)
    print("validation path: ", valid_path)
    print("test path: ", test_path)
    
    print("train_generator.class_indices : ", train_generator.class_indices)
    print("validation_generator.class_indices : ", validation_generator.class_indices)
    print("test_generator.class_indices : ", test_generator.class_indices)

    return train_generator, validation_generator, test_generator

def evaluate_model(model, model_name, test_generator, output_dir):
    filenames = []
    y_true = []
    y_pred = []
    scores = []

    ptr = 0

    for i in range(len(test_generator)):
  
        batch_data = test_generator[i]
        image_batch, label_batch = batch_data[0], batch_data[1]
        curr_batch_size = image_batch.shape[0]

        batch_filenames = test_generator.filenames[ptr : ptr + curr_batch_size]
        ptr += curr_batch_size
        
        predictions = model.predict_on_batch(image_batch).flatten()

        scores.extend(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1).numpy()

        filenames.extend(batch_filenames)
        y_true.append(label_batch.flatten())
        y_pred.append(predictions.flatten())
        
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    
    f1_val, recall, precision, accuracy = metrics_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = None 

    # Write to CSV file
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_results  = output_dir / f"{model_name}_predictions_results.csv"
    metrics_summary = output_dir / f"{model_name}_metrics_summary.csv"
    with open(predictions_results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Score'])

        for i in range(len(filenames)):
            writer.writerow([filenames[i], y_true[i], y_pred[i], scores[i]])

    with open(metrics_summary, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['F1 Score', f1_val])
        writer.writerow(['Precision', precision])
        writer.writerow(['Recall', recall])
        writer.writerow(['Accuracy', accuracy])
        if roc_auc is not None:
            writer.writerow(['ROC-AUC', roc_auc])
        else:
            writer.writerow(['ROC-AUC', 'Undefined (only one class)'])

    logging.info(f"Predictions saved to {model_name}_predictions_results.csv")
    logging.info(f"Metrics saved to {model_name}_metrics_summary.csv")

    return predictions_results, metrics_summary

def evaluate_only(model_path, model_name, test_path, output_dir, classes = {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}):
    test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input_vgg19)

    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'f1_score_normal': f1_score_normal})

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        class_mode='binary',
        classes = classes,
        shuffle = False,
    )
    return evaluate_model(model, model_name, test_generator, output_dir)

def train_and_evaluate(train_path, 
                       valid_path, 
                       test_path, 
                       model_path,
                       log_path,
                       eval_path,
                       model_name,
                       best_hyperparameters_json_path = None, 
                       classes = {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1},
                        ):

    logging.basicConfig(level=logging.INFO)
    if best_hyperparameters_json_path is None:
        best_params = {
            "rotation_range": 5,
            "width_shift_range": 0.04972485058923855,
            "height_shift_range": 0.03008783098167697,
            "horizontal_flip": True,
            "vertical_flip": True,
            "zoom_range": -0.044852124875001065,
            "brightness_range": -0.02213535357633886,
            "use_class_weights": True,
            "pooling": "global_average",
            "dense_layers": 3,
            "units_layer_0": 64,
            "activation_func_0": "sigmoid",
            "batch_norm_0": True,
            "dropout_0": 0.09325925519992712,
            "units_layer_1": 64,
            "activation_func_1": "tanh",
            "batch_norm_1": True,
            "dropout_1": 0.17053317552512925,
            "units_layer_2": 32,
            "activation_func_2": "relu",
            "batch_norm_2": True,
            "dropout_2": 0.31655072863663397,
            "fine_tune_at": 7,
            "fine_tuning_learning_rate_adam": 0.00001115908855034341,
            "batch_size": 32
        }

    else:
        with open(best_hyperparameters_json_path, 'r') as file:
            best_params = json.load(file)
    
    set_seeds()
    train_generator, validation_generator, test_generator = get_data_generators(train_path, valid_path, test_path, best_params, classes)
        
    K.clear_session()
    strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    with strategy.scope():
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        
        if best_params['pooling'] == 'global_average':
            x = GlobalAveragePooling2D()(x)
        else:
            x = Flatten()(x)
        
        for i in range(best_params['dense_layers']):
            num_units = best_params[f'units_layer_{i}']
            activation = best_params[f'activation_func_{i}']
            x = Dense(num_units, activation=activation)(x)
        
            if best_params[f'batch_norm_{i}']:
                x = BatchNormalization()(x)
        
            x = Dropout(best_params[f'dropout_{i}'])(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        
        base_model.trainable = True
        for layer in base_model.layers[:best_params['fine_tune_at']]:
            layer.trainable = False
        
        optimizer = Adam(learning_rate=best_params['fine_tuning_learning_rate_adam'])
        model.compile(
        
            optimizer=optimizer,
            loss=BinaryCrossentropy(),
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
        print("Model weights device location:", model.weights[0].device)
        # Training
        class_weights = None
        if best_params['use_class_weights']:
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
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
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                        EarlyStopping(monitor='val_roc_auc_score', mode='max', verbose=1, patience=8, restore_best_weights=True),
                           ], )
        predictions_results, metrics_summary = evaluate_model(model=model, 
                   model_name=model_name, 
                   test_generator=test_generator, 
                   output_dir=eval_path,
                 )
        
    if model_name:
        model_save_path = os.path.join(model_path, f'{model_name}.h5')
    else:
        model_save_path = os.path.join(model_path,'Trained_model.h5')
    model.save(model_save_path)

    hist_df = pd.DataFrame(training_log.history) 
    training_history_csv = os.path.join(log_path, f'training_history_{model_name}.csv')
    hist_df.to_csv(training_history_csv, index=False)
    logging.info(f"{model_name} Model trained, Model and training history are saved successfully.")
    return  predictions_results, metrics_summary, model_save_path, training_history_csv

def predict_single_image(img_path, model_path, classes={'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}):
    model = tf.keras.models.load_model(model_path,custom_objects={'f1_score_normal': f1_score_normal})
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input_vgg19(img_array)

    prediction = model.predict(img_array).flatten()[0]
    predicted_class = int(prediction > 0.5)
    class_label_map = {v: k for k, v in classes.items()}

    print("Model predicted: ", class_label_map[predicted_class])
    return  class_label_map[predicted_class]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'predict'], help='Operation mode')
    parser.add_argument('--model_path', type=str,  help='Path to load or save model')
    parser.add_argument('--model_name', type=str, help='Model name for saving or loading')
    parser.add_argument('--train_path', type=str, help='Path to training images')
    parser.add_argument('--valid_path', type=str, help='Path to validation images')
    parser.add_argument('--test_path', type=str, help='Path to test images')
    parser.add_argument('--log_path', type=str, help='Path to save training logs')
    parser.add_argument('--eval_path', type=str, help='Path to save evaluation results')
    parser.add_argument('--hyperparameters_json_path', required=False, default=None, type=str, help='Path to hyperparameters JSON')
    parser.add_argument('--image_path', type=str, help='Path to a single image for prediction')
    parser.add_argument('--classes_definition', type=str, required=False, default=None, help='A dictionary of classes definition')

    args = parser.parse_args()

    if args.classes_definition:
        try:
            args.classes_definition = ast.literal_eval(args.classes_definition)
            if not isinstance(args.classes_definition, dict):
                print("Classes definition has to be a dictionary, e.g. {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}")
                args.classes_definition = None  
        except Exception as e:
            print("Classes definition has to be a dictionary, e.g. {'No_Glaucoma': 0, 'Suspected_Glaucoma': 1}")
            args.classes_definition = None 
    else:
        args.classes_definition = None

    logging.basicConfig(level=logging.INFO)

    if args.mode == 'train':
        required_args = ['train_path', 'valid_path', 'test_path', 'log_path', 'eval_path', 'model_name']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(f"Missing required arguments for training: {', '.join(missing_args)}")
            
        return train_and_evaluate(
            train_path=args.train_path,
            valid_path=args.valid_path,
            test_path=args.test_path,
            model_path=args.model_path,
            log_path=args.log_path,
            eval_path=args.eval_path,
            model_name=args.model_name,
            best_hyperparameters_json_path=args.hyperparameters_json_path,
            classes= args.classes_definition
        )

    elif args.mode == 'evaluate':
        required_args = ['model_name', 'model_path', 'test_path', 'eval_path']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(f"Missing required arguments for prediction: {', '.join(missing_args)}")

        return evaluate_only(
            model_path=args.model_path,
            model_name=args.model_name,
            test_path=args.test_path,
            output_dir=Path(args.eval_path),
            classes = args.classes_definition
        )

    elif args.mode == 'predict':
        required_args = ['image_path', 'model_path']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(f"Missing required arguments for prediction: {', '.join(missing_args)}")
            
        return  predict_single_image(
            img_path=args.image_path,
            model_path=args.model_path,
            classes= ast.literal_eval(args.classes_definition)
        )
   
    
if __name__ == '__main__':
    main()
