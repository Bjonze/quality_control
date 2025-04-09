#This file takes as input a .txt file with file names,
#and convert them to a .vtk LAA mesh surface file.
from plot_meshes import nifisdf2vtk
import os
from tqdm import tqdm

def main(file_list, segmentation_dir, output_dir):
    #read file list 
    with open(file_list, 'r') as f:
        files = f.readlines()
    files = [line.strip() for line in files]
    for file in tqdm(files):
        #create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #create input file path
        input_file = os.path.join(segmentation_dir, file+"_labels.nii.gz")
        #create output file path
        output_file = os.path.join(output_dir, file+".vtk")
        #convert to vtk
        if not os.path.exists(output_file):
            nifisdf2vtk(input_file, output_file)


if __name__ == "__main__":
    file_list = "/storage/code/quality_control/outlier_filenames.txt"
    segmentation_dir = "/data/Data/LAAAnalysis-08-04-2025/labelmaps"
    output_dir = "/data/Data/vtk_meshes"
    main(file_list, segmentation_dir, output_dir)
