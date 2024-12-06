# # cleaned 1.0.3 with bounding box output as well

import os
import sys
import argparse
import time
import cv2 as cv
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.callbacks import Callback, EarlyStopping
import logging

IMG_SIZE = (300, 300)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

SAVED_MODEL_NAME_acc_vgg19_300_auc = 'optic_disk_crop_model.hdf5'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 17

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    intermediate_layer_1 = tf.keras.layers.Dense(128, activation='relu')
    intermediate_layer_2 = tf.keras.layers.Dense(128, activation='relu')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)  # , activation='sigmoid')

    inputs = tf.keras.Input(shape=(300, 300, 3))
    # x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = intermediate_layer_1(x, )  # activation='relu')
    x = tf.keras.layers.Dropout(0.2)(x)
    x = intermediate_layer_2(x)  # , activation='relu')
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    val_acc_es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3, restore_best_weights=True)
    # train_auc_es = EarlyStopping(monitor=auc, mode='max', verbose=1, patience=6, restore_best_weights=True)
    # val_auc_es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=4, restore_best_weights=True)

    model_checkpoint_callback_val_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVED_MODEL_NAME_acc_vgg19_300_auc,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    auc = tf.keras.metrics.AUC()

    check = tf.keras.callbacks.ModelCheckpoint(
        'AUC ' + SAVED_MODEL_NAME_acc_vgg19_300_auc,
        monitor='val_auc',  # even use the generated handle for monitoring the training AUC
        save_best_only=True,
        save_weights_only=True,
        mode='max')  # determine better models according to "max" AUC.

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', 'AUC'])

    # Load the pre-trained model

    model.load_weights(model_path)
    return model


def save_svg(output_directory, annotation_type_rid, rid, raw_image_size, bbox, annotation_type_name):
    # Set the viewBox to the size of the raw image
    view_box = f"0 0 {raw_image_size['width']} {raw_image_size['height']}"
    # SVG canvas size should match the raw image size
    svg_width = raw_image_size['width']
    svg_height = raw_image_size['height']
    group_id = f"eye-ai:{annotation_type_rid},{annotation_type_name}"
    rect_stroke_color = "#ff0000"

    svg_content = f'''<svg width="{svg_width}px" height="{svg_height}px" viewBox="{view_box}" 
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
      <g id="{group_id}">
        <rect fill="none" x="{bbox['left']}" y="{bbox['top']}" width="{bbox['width']}" 
        height="{bbox['height']}" stroke="{rect_stroke_color}" stroke-width="2"/>
      </g>
    </svg>
    '''

    svg_file_path = os.path.join(output_directory,
                                 f"Execution_Assets/Image_Annotation/Cropped_{rid}.svg")
    os.makedirs(os.path.dirname(svg_file_path), exist_ok=True)
    with open(svg_file_path, "w") as file:
        file.write(svg_content)


def preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_path, model_path,
                        annotation_type_rid, annotation_type_name, cropped_image):
    model = load_model(model_path)
    # Template creation
    template = np.ones((50, 50), dtype="uint8") * 0
    template = cv.circle(template, (25, 25), 24, 255, -1)

    if not cv.imwrite(template_path, template):
        logging.error("Error: Template could not be saved.")
        exit()

    def imgResize_primary(img):
        h = img.shape[0]
        w = img.shape[1]
        perc = 500 / w
        w1 = 500
        h1 = int(h * perc)
        img_rs = cv.resize(img, (w1, h1))
        return img_rs

    def imgResize_secondary(img):
        img_rs = cv.resize(img, (600, 600))
        return img_rs

    def getImage(directory_path, img_name):
        full_path = directory_path + img_name
        image = cv.imread(full_path, -1)
        if image is None:
            logging.error("Error: Image could not be read.")
            return None
        return image

    def kmeansclust(img, k):
        img_rsp = img.reshape((-1, 1))
        img_rsp = img_rsp.astype('float32')
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 400, 0.99)
        _, labels, (centers) = cv.kmeans(img_rsp, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = centers.astype('uint8')
        labels = labels.flatten()
        seg_img = centers[labels.flatten()]
        seg_img = seg_img.reshape(img.shape)
        return seg_img

    def imgResizeToFixedSize(img):
        desired_dim = (300, 300)
        img_rs = cv.resize(img, desired_dim)
        return img_rs

    def predict_optic_disc_center(image, model):
        IMG_SIZE = (300, 300)
        image = cv2.resize(image, IMG_SIZE)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        prediction = model.predict(image)
        return "Proper" if prediction > 0.5 else "Not Proper"

    def crop_to_eye(im):
        mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_im = im[y:y + h, x:x + w]
        return cropped_im

    # def crop_to_eye_with_scaling_to_raw_image(im):
    #     mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     max_contour = max(contours, key=cv2.contourArea)
    #     x,y,w,h = cv2.boundingRect(max_contour)
    #     cropped_im = im[y:y+h, x:x+w]
    #     return cropped_im, x, y  # Return the cropped image and the top-left corner coordinates

    # Function to find the bounding box of the eye on the original image
    def find_eye_bbox(im):
        mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return x, y, w, h

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    start_time = time.time()

    if not os.path.exists(directory_path):
        logging.error(f'Error: Directory {directory_path} does not exist.')
        exit()

    image_files = os.listdir(directory_path)  # List all files in the directory
    csv_data = pd.read_csv(csv_path)

    crop_success = False  # Initialize the flag
    results_data = []  # Initialize the results data list

    for index, row in csv_data.iterrows():
        img_name = row['Filename']
        rid = row['RID']  #
        image_vocab = row['Image_Tag']

        if img_name not in image_files:
            logging.error(f'Image {img_name} does not exist in directory.')
            continue

        img = getImage(directory_path, img_name)
        if img is None:
            continue
        raw_image_size = {"width": img.shape[1], "height": img.shape[0]}
        resize_functions = [imgResize_primary, imgResize_secondary]  # Put resizing functions in a list
        for crop_size in range(95, 116, 10):
            for resize_function in resize_functions:  # Iterate over resizing functions
                img = crop_to_eye(img)  # First, crop to eye
                img_rs = resize_function(img)
                original_scale = img.shape[0] / img_rs.shape[0]  # Calculate the scale of the original image
                for trial, color_channel in enumerate(["grey", "green", "red", "blue"], 1):
                    logging.info(
                        f"Processing image {img_name} : , trial {trial}, color channel {color_channel}, resize function {resize_function.__name__}"
                    )
                    if color_channel == "grey":
                        img_k = kmeansclust(cv.cvtColor(img_rs, cv.COLOR_BGR2GRAY), 7)
                    else:
                        img_k = kmeansclust(cv.split(img_rs)[{"red": 2, "green": 1, "blue": 0}[color_channel]], 7)

                    temp = cv.imread(template_path, -1)
                    if temp is None:
                        logging.error("Template could not be read.")
                        continue

                    metd = cv.TM_CCOEFF_NORMED
                    temp_mat = cv.matchTemplate(img_k, temp, metd)

                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(temp_mat)
                    x = max_loc[0] + 45
                    y = max_loc[1] + 45

                    top = max(0, y - crop_size)
                    bottom = min(img_rs.shape[0], y + crop_size)
                    left = max(0, x - crop_size)
                    right = min(img_rs.shape[1], x + crop_size)

                    # Scale the coordinates back to the original image size
                    top1 = int(top * original_scale)
                    bottom1 = int(bottom * original_scale)
                    left1 = int(left * original_scale)
                    right1 = int(right * original_scale)

                    height1, width1 = bottom1 - top1, right1 - left1
                    if height1 < 600:
                        diff1 = 600 - height1
                        top1 = max(0, top1 - diff1 // 2)
                        bottom1 = min(img.shape[0], bottom1 + (diff1 - diff1 // 2))

                    if width1 < 600:
                        diff1 = 600 - width1
                        left1 = max(0, left1 - diff1 // 2)
                        right1 = min(img.shape[1], right1 + (diff1 - diff1 // 2))

                    cropped_img1 = img[top1:bottom1, left1:right1]
                    resized_img1 = imgResizeToFixedSize(cropped_img1)

                    prediction = predict_optic_disc_center(resized_img1, model)  # Replace with your function
                    if prediction == "Proper":

                        # Get the bounding box of the eye on the original image before resizing
                        original_image = getImage(directory_path, img_name)
                        x, y, w, h = find_eye_bbox(original_image)

                        # Calculate the original bounding box coordinates based on the scale
                        original_left = x + int(left * original_scale)
                        original_top = y + int(top * original_scale)
                        original_right = x + int(right * original_scale)
                        original_bottom = y + int(bottom * original_scale)
                        original_width = original_right - original_left
                        original_height = original_bottom - original_top
                        
                        if cropped_image:
                            img_path1 = f'{output_path}{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}'
                            if not cv.imwrite(img_path1, cropped_img1):
                                logging.error(f"Error: Image could not be saved at: {img_path1}")

                        # Save the SVG file   
                        bbox = {
                            "left": original_left,
                            "top": original_top,
                            "width": original_width,
                            "height": original_height
                        }

                        # save_svg(output_path, execution_rid, annotation_type_rid, rid, raw_image_size, bbox, annotation_type_name)
                        save_svg(output_path, annotation_type_rid, rid, raw_image_size, bbox, annotation_type_name)
                        logging.info(f"SVG for {rid} saved.")

                        # print(f"Image {img_name} ({color_channel}) cropped and saved at {img_path1}.")
                        crop_success = True  # Set the flag

                        # Append the information to the results data list
                        results_data.append(
                            {"Image Name": img_name,
                             'Image RID': rid,
                             "Saved Image Name": f'{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                             "Saved SVG Name": f'Cropped_{rid}.svg',
                             "Worked Color Channel": color_channel,
                             "Cumulative Trials": trial,
                             "Worked Crop Size": crop_size,
                             "Cropped to Eye Bounding Box Top": top1,
                             "Cropped to Eye Bounding Box Bottom": bottom1,
                             "Cropped to Eye Bounding Box Left": left1,
                             "Cropped to Eye Bounding Box Right": right1,
                             "Original Bounding Box Top": original_top,
                             "Original Bounding Box Bottom": original_bottom,
                             "Original Bounding Box Left": original_left,
                             "Original Bounding Box Right": original_right,
                             "Worked Image Cropping Function": resize_function.__name__})
                        break
                    else:
                        continue

                if crop_success:
                    break  # Break from the resize_functions loop
            if crop_success:
                break  # Break the loop over crop sizes

        # If all fails, save the raw cropped image
        if not crop_success:

            img_rs = crop_to_eye(img)

            # Get the bounding box of the eye on the original image before resizing
            original_image = getImage(directory_path, img_name)
            bbox_x, bbox_y, bbox_w, bbox_h = find_eye_bbox(original_image)

            if cropped_image:
                # Define the image paths
                raw_img_path = f'{output_path}{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}'  # f'{dir_path}{"Cropped_High_Res"}_{img_name.split(".")[0]}.{img_name.split(".")[1]}'

                if not cv.imwrite(raw_img_path, img_rs):
                    logging.error(f"Error: Raw image could not be saved at: {raw_img_path}")
                logging.info(f"Raw Image {img_name} saved  at {raw_img_path}.")

            # Save the SVG file for the raw cropped image
            bbox = {
                "left": bbox_x,
                "top": bbox_y,
                "width": bbox_w,
                "height": bbox_h
            }

            # save_svg(output_path, execution_rid, annotation_type_rid, rid, raw_image_size, bbox, annotation_type_name)
            save_svg(output_path, annotation_type_rid, rid, raw_image_size, bbox, annotation_type_name)
            logging.info(f"SVG for {rid} saved.")

            # Append the information to the DataFrame
            results_data.append({"Image Name": img_name,
                                 'Image RID': rid,
                                 "Saved Image Name": f'{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                                 "Saved SVG Name": f'Cropped_{rid}.svg',
                                 "Cropped to Eye left": bbox_x,
                                 "Cropped to Eye top": bbox_y,
                                 "Cropped to Eye width": bbox_w,
                                 "Cropped to Eye height": bbox_h,
                                 "Cropped to Eye right": bbox_x + bbox_w,
                                 "Cropped to Eye bottom": bbox_y + bbox_h,
                                 "Worked Color Channel": "None",
                                 "Worked Image Cropping Function": "Raw_Cropped_to_Eye", }, )

        crop_success = False  # Reset the flag for the next image

    # delete the template image
    try:
        os.remove(template_path)
    except:
        pass

    # Create DataFrame from results data
    results_df = pd.DataFrame(results_data)
    # output_csv_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_results.csv")
    results_df.to_csv(output_csv_path, index=False)

    logging.info(f"Number of images in CSV: {results_df.shape[0]}")
    logging.info(f"Number of images in directory: {len(image_files)}")
    logging.info(f"Number of images in output directory: {len(os.listdir(output_path))}")
    logging.info(f"Number of cropped images: {len(results_data)}")

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', type=str, required=True, help='Path to the bag')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV')
    parser.add_argument('--template_path', type=str, required=False, help='Path to the template')
    parser.add_argument('--output_csv_path', type=str, required=True, help='Path to the output CSV')
    parser.add_argument('--output_directory', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model')
    parser.add_argument('--annotation_type_rid', type=str, required=True, help='Annotation Tag RID')
    parser.add_argument('--annotation_type_name', type=str, required=True, help='Annotation Type Name')
    parser.add_argument('--cropped_image', type=bool, required=True, help='Out put cropped image')

    # parse the arguments
    args = parser.parse_args()

    sys.exit(preprocess_and_crop(
        args.bag_path,
        args.csv_path,
        args.output_csv_path,
        args.template_path,
        args.output_directory,
        args.model_path,
        args.annotation_type_rid,
        args.annotation_type_name,
        args.cropped_image))
