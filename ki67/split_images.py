import os
import shutil
import pathlib
from utils import read_json

def split_images(json_annotation_file, base_folder, target_folder):
    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True) 

    annotations = read_json(json_annotation_file)
    
    for image in annotations['images']:
        init_path = os.path.join(base_folder, image['file_name'])
        target_path = os.path.join(target_folder, image['file_name'])
        shutil.copyfile(init_path, target_path)
        
for set_name in ['train', 'val', 'test']:
    split_images(f'data/instances_{set_name}.json', 'data/ori', f'data/images/{set_name}')