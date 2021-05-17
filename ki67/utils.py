# Reference
# https://linuxhint.com/python_xml_to_dictionary/

import os
import xmltodict
import math
import random
from copy import deepcopy

# Open an xml file and convert it into the dictionary format
def xml_to_dict(base, filename):
    with open(os.path.join(base, filename), "r") as xml_obj:
        my_dict = xmltodict.parse(xml_obj.read())
        xml_obj.close()
    return my_dict

def convert2coco(base):    
    # Get filenames of xml files
    filenames = [filename for filename in os.listdir(base) if os.path.splitext(filename)[-1] == '.xml']
            
    # Get all classes in the dataset
    cls_dict = {}
    for filename in filenames:
        ann  = xml_to_dict(base, filename)['annotation']
        for obj in ann['object']:
            cls_name = obj['name']
            if cls_name not in cls_dict:
                cls_dict[cls_name] = len(cls_dict) + 1
    
    # Initialize the annotations dictionary
    annotations = {
        "type": "instances",
        "images": [],
        "categories": [],
        "annotations": []
    }
    img_id = 1; ann_id = 1
    
    # Convert to the COCO Object Detection format
    for filename in filenames:
        # Get annotations of an xml file
        ann  = xml_to_dict(base, filename)['annotation']
            
        for ann_obj in ann['object']:
            bbx = ann_obj['bndbox']
            xmin, ymin = int(bbx['xmin']), int(bbx['ymin'])
            xmax, ymax = int(bbx['xmax']), int(bbx['ymax'])
            dx = xmax - xmin
            dy = ymax - ymin
            
            annot = {
                "id": ann_id,
                "bbox": [xmin, ymin, dx, dy],
                "image_id": img_id,
                "category_id": cls_dict[ann_obj['name']],
                "segmentation": [],
                "area": dx*dy,
                "iscrowd": 0
            }
            annotations["annotations"].append(annot)
            ann_id = ann_id + 1
            
        size = ann['size']
        image = {
            "file_name": ann['filename'],
            "height":size['height'] ,
            "width": size['width'],
            "id": img_id
        }
        annotations["images"].append(image)
        img_id = img_id + 1
            
    for cls_name, cls_id in cls_dict.items():
        annotations["categories"].append({
            "supercategory": "none",
            "name": cls_name,
            "id": cls_id
        })
        
    return annotations

def coco_to_img2annots(annotations):
    img2annots = {}

    num_obj_init = {category['id']: 0 for category in annotations['categories']}
    for image in annotations['images']:
        image_id = image['id']
        if image_id in img2annots:
            print('Error: There are duplicate image ids, please check your dataset.')
            return None
        img2annots[image_id] = {
            'description': deepcopy(image),
            'annotations': [],
            'num_objects': deepcopy(num_obj_init)
        }

    for annotation in annotations['annotations']:
        if annotation['image_id'] not in img2annots:
            print('Error: There exists an annotation where image_id does not exist in annotations["images"]')
            return None
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        img2annots[image_id]['annotations'].append(annotation)
        img2annots[image_id]['num_objects'][category_id] = img2annots[image_id]['num_objects'][category_id] + 1

    return img2annots
    return {
        'type': annotations['type'],
        'categories': annotations['categories'],
        'img2annots': img2annots
    }

def dataset_split(input_annotations, split_dictionary, max_iter=100):    
    total = sum([val for _, val in split_dictionary.items()])
    split_dict = {key: val/total for key, val in split_dictionary.items()}
    
    # Get the number of objects
    categories = [category['id'] for category in input_annotations['categories']]
    total_objects = {cat_id: 0 for cat_id in categories}
    
    # Mapping from an image into its annotations
    img2annots = coco_to_img2annots(input_annotations)['img2annots']
    
    for key, val in img2annots.items():
        for cat_id, cat_n in val['num_objects'].items():
            total_objects[cat_id] = total_objects[cat_id] + cat_n

    # Get the spit_size for every set
    total_img = len(img2annots.keys())
    split_size_img = {}
    for key1, val1 in split_dict.items():
        split_size_img[key1] = math.ceil(val1*total_img)
            
    # Get the index_mapping for each set
    split_img_dict = {}
    start_idx = 0
    for key, val in split_size_img.items():
        split_img_dict[key] = [start_idx, min(start_idx + val, total_img)]
        start_idx = start_idx + val
        
    # Calculate the percentage of objects w.r.t to total objects
    def calculate_object(data_dict, total_objects):
        count = {key: 0 for key, _ in total_objects.items()}
        for key, val in data_dict.items():
            for ann in val['annotations']:
                category_id = ann['category_id']
                count[category_id] = count[category_id] + 1
        for key, val in total_objects.items():
            count[key] = count[key] / val
        return count
            
    # Optimization
    img_name = list(img2annots.keys())
    obj_counts = {}
    best_error = 1.
    for i in range(max_iter):
        random.shuffle(img_name)
        
        for key, val in split_img_dict.items():
            obj_dict = {name: img2annots[name] for name in img_name[val[0]:val[1]]}
            obj_counts[key] = calculate_object(obj_dict, total_objects)
            
        error = 0
        for key1, val1 in split_dictionary.items():
            for key2, val2 in obj_counts[key1].items():
                error = error + (val1-val2)**2
        
        if error < best_error:
            best_error = deepcopy(error)
            best_img_name_seq = deepcopy(img_name)
            print(f'The best error: {best_error}')
    
    # Split the dataset
    annotations = {}
    for key1, val1 in split_img_dict.items():
        obj_dict = {name: img2annots[name] for name in best_img_name_seq[val1[0]:val1[1]]}
        annotations[key1] = {
            'type': input_annotations['type'],
            'categories': input_annotations['categories'],
            'images': [],
            'annotations': []
        }
        for key2, val2 in obj_dict.items():
            annotations[key1]['images'].append(val2['description'])
            annotations[key1]['annotations'].extend(val2['annotations'])
    return annotations

def dataset_analysis(annotations, display=True):
    # Get the classes details
    set_names = list(annotations.keys())
    categories = annotations[set_names[0]]['categories']
    num_objects = 0
    num_images = 0
    
    results = {set_name: {
        'num_images': 0,
        'num_objects': 0,
        'objects': {category['id']: 0 for category in categories}
    } for set_name in set_names}
    
    for set_name in set_names:
        anns = annotations[set_name]
        for image in anns['images']:
            num_images = num_images + 1
            results[set_name]['num_images'] = results[set_name]['num_images'] + 1
        
        for objs in anns['annotations']:
            cat_id = objs['category_id']
            num_objects = num_objects + 1
            results[set_name]['num_objects'] = results[set_name]['num_objects'] + 1
            results[set_name]['objects'][cat_id] = results[set_name]['objects'][cat_id] + 1
    
    if display:
        print('-----------------------------------')
        print('num_images', ' '*(20 - len(f'num_images{num_images}')), num_images)
        print('num_objects', ' '*(20 - len(f'num_objects{num_objects}')), num_objects)

        print('-----------------------------------')
        print('num_images on each set')
        print('')
        total = sum([results[set_name]['num_images'] for set_name in set_names])
        for set_name in set_names:
            nimgs = results[set_name]['num_images']
            pct = nimgs / total
            print(set_name, ' '*(15-len(f'{set_name}{nimgs}')), nimgs, ' '*2, "{:.3f}".format(pct))

        print('-----------------------------------')
        print('num_objects on each set')
        print('')
        total = sum([results[set_name]['num_objects'] for set_name in set_names])
        for set_name in set_names:
            nobjs = results[set_name]['num_objects']
            pct = nobjs / total
            print(set_name, ' '*(15-len(f'{set_name}{nobjs}')), nobjs, ' '*2, "{:.3f}".format(pct))

        for category in categories:
            cat_id = category['id']
            print('-----------------------------------')
            print(f'Category: {cat_id}')
            print('')
            total = sum([results[set_name]['objects'][cat_id] for set_name in set_names])
            for set_name in set_names:
                nobjs = results[set_name]['objects'][cat_id]
                pct = nobjs / total
                print(set_name, ' '*(15-len(f'{set_name}{nobjs}')), nobjs, ' '*2, "{:.3f}".format(pct))
        print('-----------------------------------')
            
    return results