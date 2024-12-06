import sys
import argparse
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import csv
import logging
from pathlib import Path, PurePath

def train_model():
    pass
def main(*, model_path, image_path, output_path):
    train_model()

if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the prediction model')
    parser.add_argument('--cropped_image_path', type=str, required=True, help='Path to the cropped images')
    parser.add_argument('--output_dir', type=str, required=False, help='Path to the output CSV')

    # parse the arguments
    args = parser.parse_args()
    main(model_path=args.model_path, image_path=args.cropped_image_path, output_path=args.output_dir)

    sys.exit(prediction(args.model_path,
                        args.cropped_image_path,
                        args.output_dir))
