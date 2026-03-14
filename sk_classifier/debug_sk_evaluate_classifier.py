import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt




def main():

    base_directory = "/path/to/base/directory"

    image_results = glob.glob(base_directory)

    superclass_dict = {
        "PE": "Phospholipids",
        "PC": "Phospholipids",
        "PS": "Phospholipids",
        "PI": "Phospholipids",
        ""
        "Cardiolipin": "Phospholipids",
        ""
    }

    csv_dict = {}

    for image in image_results:

        image_name = os.path.basename(image).split(".")[0]

        csv_path = glob.glob(os.path.join(image, "*.csv"))[0]

        if os.path.exists(csv_path):

            df = pd.read_csv(csv_path, header=None)
            print(f"Loaded CSV for {image_name} with shape {df.shape}")

 
            unique_classes, counts = np.unique(df.values(), return_counts=True)
            class_distribution = dict(zip(unique_classes, counts))

            superclas_distribution = 
            
            csv_dict[image_name] = {
                "data": df,
                "class_distribution": class_distribution,
                "superclass_distribution": superclass_distribution
            }
            

