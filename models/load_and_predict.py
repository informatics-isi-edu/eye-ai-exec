import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Set number of workers based on the number of available CPUs
num_workers = os.cpu_count()


# Define the function to preprocess input
def preprocess_input_vgg16(x):
    return tf.keras.applications.vgg16.preprocess_input(x)


# Compute class weights
class_weights = {0: 0.8263807775738361, 1: 1.265976482617587}

# Load pre-trained VGG16
K.clear_session()  # Clear session to avoid memory issue
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune from this layer onwards
fine_tune_at = 5
for i, layer in enumerate(base_model.layers):
    layer.trainable = i >= fine_tune_at

# Add global average pooling layer
x = GlobalAveragePooling2D()(base_model.layers[-1].output)

# Add dense layers
for i in range(3):
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3978963347180505)(x)

outputs = Dense(1, activation='sigmoid')(x)

# Define new model
model = Model(inputs=base_model.inputs, outputs=outputs)

# Define the optimizer
optimizer = Adam(learning_rate=2.70835918680168e-05)

# Define loss function
loss = BinaryCrossentropy()


# Define custom F1 score metric
def f1_score_normal(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        AUC(curve="PR", name="pr_auc_score"),
        AUC(curve="ROC", name="roc_auc_score"),
        Precision(name="precision_score"),
        Recall(name="recall_score"),
        BinaryAccuracy(name="accuracy_score"),
        f1_score_normal
    ]
)

model.summary()

batch_size = 32


# Load the model
model.load_weights('/content/drive/Shareddrives/ISI Dataset/presentation_model_for_Seminar.h5')

# Define the path
path = '/content/images/cropped/'

# Iterate over each folder in the path
for folder in os.listdir(path):
    # Get the full path of the folder
    folder_path = os.path.join(path, folder)

    # Iterate over each image in the folder
    for image_name in os.listdir(folder_path):
        # Get the full path of the image
        image_path = os.path.join(folder_path, image_name)

        # Load the image
        image = load_img(image_path, target_size=(224, 224))

        # Convert the image to a numpy array
        image = img_to_array(image)

        # Expand the dimensions of the image
        image = np.expand_dims(image, axis=0)

        # Preprocess the image
        image = tf.keras.applications.vgg16.preprocess_input(image)

        # Make prediction
        prediction = model.predict(image)

        # Check if the prediction is less than 0.5
        if prediction < 0.5:
            class_name = 'No Glaucoma'
        else:
            class_name = 'Glaucoma Suspect'

        # Print the prediction and class name
        print(f'Prediction for {image_name}: {prediction}, Class name: {class_name}')
