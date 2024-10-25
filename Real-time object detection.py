import numpy as np 
import os
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Set the paths to the train, test, and valid directories
train_dir = 'C:/Users/User/Downloads/train'
test_dir = 'C:/Users/User/Downloads/test'
valid_dir = 'C:/Users/User/Downloads/valid'

# Function to collect image paths and annotation paths
def collect_image_and_annotation_paths(data_dir):
    image_paths = []
    annotation_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files):
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

                # Check if the corresponding annotation file exists
                annotation_path = os.path.join(root, os.path.splitext(file)[0] + '.xml')
                annotation_paths.append(annotation_path if os.path.exists(annotation_path) else None)
    return image_paths, annotation_paths

# Collect paths for train, test, and validation sets
train_image_paths, train_annotation_paths = collect_image_and_annotation_paths(train_dir)
test_image_paths, test_annotation_paths = collect_image_and_annotation_paths(test_dir)
valid_image_paths, valid_annotation_paths = collect_image_and_annotation_paths(valid_dir)

print(f"Number of train images: {len(train_image_paths)}")
print(f"Number of test images: {len(test_image_paths)}")
print(f"Number of valid images: {len(valid_image_paths)}")

# Function to preprocess the image and get the original size
def preprocess_image_with_size(image_path, target_size=(224, 224)):
    try:
        # Open image and resize
        image = Image.open(image_path)
        original_size = image.size  # (width, height)
        image = image.resize(target_size)
        # Convert to array and normalize
        image = img_to_array(image) / 255.0
        return image, original_size
    except Exception as e:
        print(f"Error processing image: {image_path}, {e}")
        return None, None

# Function to extract bounding boxes from an XML annotation file
def extract_bounding_boxes_from_xml(annotation_path):
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bboxes = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append([label, xmin, ymin, xmax, ymax])
        return bboxes
    except Exception as e:
        print(f"Error processing annotation: {annotation_path}, {e}")
        return None

# Function to adjust bounding boxes according to image resizing
def adjust_bounding_boxes(bboxes, original_size, new_size):
    x_scale = new_size[0] / original_size[0]
    y_scale = new_size[1] / original_size[1]
    adjusted_bboxes = []
    for bbox in bboxes:
        label, xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)
        adjusted_bboxes.append([label, xmin, ymin, xmax, ymax])
    return adjusted_bboxes

# Function to preprocess images and annotations for a given dataset
def preprocess_dataset(image_paths, annotation_paths, target_size=(224, 224)):
    images_preprocessed = []
    annotations_preprocessed = []
    
    for image_path, annotation_path in zip(image_paths, annotation_paths):
        img, original_size = preprocess_image_with_size(image_path, target_size)
        if img is not None:
            images_preprocessed.append(img)
            if annotation_path:
                bboxes = extract_bounding_boxes_from_xml(annotation_path)
                if bboxes:
                    adjusted_bboxes = adjust_bounding_boxes(bboxes, original_size, target_size)
                    annotations_preprocessed.append(adjusted_bboxes)
                else:
                    annotations_preprocessed.append(None)
            else:
                annotations_preprocessed.append(None)
    
    return images_preprocessed, annotations_preprocessed

# Preprocess the datasets
target_size = (224, 224)
train_images_preprocessed, train_annotations_preprocessed = preprocess_dataset(train_image_paths, train_annotation_paths, target_size)
val_images_preprocessed, val_annotations_preprocessed = preprocess_dataset(valid_image_paths, valid_annotation_paths, target_size)
test_images_preprocessed, test_annotations_preprocessed = preprocess_dataset(test_image_paths, test_annotation_paths, target_size)

# Save dataset.yaml
dataset_yaml = """train: C:/Users/User/Downloads/train
val: C:/Users/User/Downloads/valid
test: C:/Users/User/Downloads/test

nc: 6 
names: ['Car', 'SUV', 'Bus', 'Truck', 'Pedestrian', 'Lawn']
"""

with open('dataset.yaml', 'w') as f:
    f.write(dataset_yaml)

# Function to convert VOC annotations to YOLO format
def convert_voc_to_yolo(xml_file, img_size):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    w, h = img_size
    
    yolo_annotations = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        class_id = class_to_id(label)
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # YOLO format requires normalized center x, center y, width, and height
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

# Map your class labels to numeric IDs
def class_to_id(label):
    class_map = {"Car": 0, "SUV": 1, "Bus": 2, "Truck": 3, "Pedestrian": 4, "Lawn":5}  
    return class_map.get(label, -1)

# Function to save YOLO annotations
def save_yolo_annotations(yolo_annotations, output_file):
    with open(output_file, 'w') as f:
        for ann in yolo_annotations:
            f.write(f"{ann}\n")

# Loop over all XML files and convert to YOLO format
def convert_annotations_to_yolo(annotation_folder, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    if os.path.exists(annotation_folder):
        for xml_file in os.listdir(annotation_folder):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotation_folder, xml_file)
                image_file = os.path.join(image_folder, xml_file.replace('.xml', '.jpg'))

                try:
                    img = Image.open(image_file)
                    img_size = img.size  # Get width and height
                    yolo_ann = convert_voc_to_yolo(xml_path, img_size)

                    # Save YOLO annotations as .txt files
                    output_file = os.path.join(output_folder, xml_file.replace('.xml', '.txt'))
                    save_yolo_annotations(yolo_ann, output_file)
                except Exception as e:
                    print(f"Error processing {xml_path}: {e}")
    else:
        print(f"Annotation folder does not exist: {annotation_folder}")

# Define paths
image_folder = "C:/Users/User/Downloads/train/images"
annotation_folder = "C:/Users/User/Downloads/train/annotations"
output_folder = "C:/Users/User/Downloads/train/labels"

# Convert VOC annotations to YOLO format
convert_annotations_to_yolo(annotation_folder, image_folder, output_folder)
