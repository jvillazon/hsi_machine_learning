import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():

    base_directory = r"/Volumes/ADATA SE880/Molecule Identification/molecule_data"
    
    # Get all averaged spectra files
    averaged_spectra_files = glob.glob(os.path.join(base_directory, "**/*averaged.txt"), recursive=True)
    
    # Create a dataframe to store the data
    df = pd.DataFrame()
    
    # Iterate over the files and add the data to the dataframe, adding a blank column in between
    file_names = []
    for file in tqdm(averaged_spectra_files, desc="Processing files"):
        # Read the .txt file and make column vectors
        array = np.loadtxt(file)
        wavenumbers = array[:, 0]
        intensities = array[:, 1]
        blank_column = np.full((len(wavenumbers), 1), np.nan)

        # Get the name of the file without the extension and make it the column name above wavenumbers
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name = file_name.replace("_averaged", " ")
        file_name = file_name.replace("_", " ")
        file_name = file_name.replace(",", ":")
        
        # Add the data to the dataframe as new columns
        df = pd.concat([df, pd.DataFrame(wavenumbers), pd.DataFrame(intensities), pd.DataFrame(blank_column)], axis=1)

        file_names.append(file_name)
        file_names.append("")
        file_names.append("")
        
    df.columns = file_names
    
    # Save the dataframe to a csv file
    df.to_excel(os.path.join(base_directory, "averaged_spectra.xlsx"), index=False)
    
if __name__ == "__main__":
    main()