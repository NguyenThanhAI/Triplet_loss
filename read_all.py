import os

for dir, _, files in os.walk('laplacian'):
    for file in files:
        image_path = os.path.join(os.getcwd(), dir, file)
        class_label = image_path.split('/')[-2]
        print(image_path, class_label)