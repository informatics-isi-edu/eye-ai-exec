import argparse
import os
import numpy as np
import pandas as pd
from eye_ai import EyeAI
from pathlib import PurePath
from eye_ai_ml.glaucoma.optic_disk_crop import preprocess_and_crop
import logging

def filter_angle2(bag_path: str) -> str:
    Dataset_Path = os.path.join(bag_path, 'data/Image.csv')
    Dataset = pd.read_csv(Dataset_Path)
    Dataset_Field_2 = Dataset[Dataset['Image_Angle_Vocab'] == "2SK6"]
    file2_csv_path = f"{bag_path}_Field_2.csv"
    Dataset_Field_2.to_csv(file2_csv_path, index=False)
    return file2_csv_path

def main(hostname: str, catalog_id:str, configuration_rid: str, data_dir: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    EA = EyeAI(hostname = hostname, catalog_id = catalog_id )
    # Initiate an Execution
    configuration_records = EA.execution_init(data_dir=data_dir, configuration_rid=configuration_rid)
    # Data Preprocessing (Selecting image of angle 2 (Field 2) -- Image angle vocab 2SK6;)
    file2_csv_path = filter_angle2(configuration_records['bag_paths'][0])
    # Execute Proecss algorithm (Cropping)
    with EA.start_execution(execution_rid=configuration_records['execution']) as exec:
        preprocess_and_crop(
            configuration_records['bag_paths'][0]+"/data/assets/Image/",
            file2_csv_path,
            './output.csv',
            'template.jpg',
            './',
            './'+configuration_records['model_paths'][0],
            configuration_records['Annotation_Type'][0]["RID"],
            configuration_records['Annotation_Type'][0]["name"],
            False
            )
    # Insert metadata into image_annotation  
    annot_metadata = pd.read_csv('output.csv')
    EA.insert_image_annotation(exec.uploaded_assets['Execution_Assets/Image_Annotation'], annot_metadata)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type = str, required=True)
    parser.add_argument('--catalog_id', type = str, required=True)
    parser.add_argument('--configuration_rid', type = str, required=True)
    parser.add_argument('--data_dir', type = str, required=True)
    args = parser.parse_args()
    main(args.hostname, args.catalog_id, args.configuration_rid, args.data_dir)
