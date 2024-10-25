# Realtime Object detection for parking model for parking spaces
# Description
This project implements a real-time object detection model using the YOLO (You Only Look Once) algorithm, trained on a parking dataset to detect and classify vehicles in parking spaces. The model is optimized for efficient and accurate detection, allowing it to identify cars and other vehicle types with high precision.

#Key Features:

Dataset: A parking dataset gotten from Kaggle (PKLot dataset), with labeled images of vehicles in various parking lot scenarios

Model: Utilizes the YOLOv8 architecture for fast and accurate real-time object detection

Classes: 6 different classes of vehicles to cover a wide range of parking situations


# Tech Stack
•Programming Language: Python (core language for model development and deployment)

•Deep Learning Framework: PyTorch - (For model training and fine-tuning, especially useful with YOLOv8), 
                          Ultralytics YOLOv8 - For YOLOv8 model implementation, leveraging pretrained weights and architecture

•Data Labeling Tools: LabelImg- For manual labeling of objects within the dataset

•CUDA and cuDNN: For GPU support 

•Visualization: Matplotlib / Seaborn - For analyzing and visualizing model performance metrics



# Application
Suitable for monitoring parking spaces in real time, enhancing parking management systems.


This project serves as a foundation for advanced parking space management and monitoring solutions, utilizing YOLOv8 for high performance on edge devices with limited GPU resources.
