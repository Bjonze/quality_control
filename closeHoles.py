import SimpleITK as sitk
import numpy as np 
import os 
from skimage.measure import marching_cubes
import edt
from plot_meshes import nifisdf2vtk

def main(images_txt, image_dir, output_dir):
    #load images_txt file and append names to list
    print(f"Reading images from {images_txt}")
    with open(images_txt, 'r') as f:
        images = f.readlines()
    images = [image.strip() for image in images]

    #loop over images and use sitk BinaryFillholeImageFilter to close holes in binary images
    for image in images:
        print(f"Processing {image}")
        image_path = os.path.join(image_dir, image+"_sdf.nii")
        img_itk = sitk.ReadImage(image_path)
        #we need to convert the image to binary image, pixels below 0 are set to 1, rest to 0
        img = sitk.BinaryThreshold(img_itk, lowerThreshold=-100, upperThreshold=0, insideValue=1, outsideValue=0)
        pad_size = [10, 10, 10]  # Example: pad 10 voxels in each dimension
        padded_mask = sitk.ConstantPad(img, pad_size, pad_size, constant=1)

        img = sitk.BinaryFillhole(padded_mask, fullyConnected=True)

        start_index = pad_size  # because we padded equally on lower and upper sides
        filled_mask = sitk.RegionOfInterest(img, size=(128,128,128), index=start_index)

        #convert image to numpy array
        img = sitk.GetArrayFromImage(filled_mask)
        inside_sdf = edt.edt(img>0.5,anisotropy=(1, 1, 1))
        outside_sdf = edt.edt(img<0.5, anisotropy=(1, 1, 1))
        out_sitk = sitk.GetImageFromArray(outside_sdf-inside_sdf)
        out_sitk.CopyInformation(img_itk)
        sitk.WriteImage(out_sitk, os.path.join(output_dir, image+"_sdf.nii"))
        #we need to convert the image to a SDF using edt. 
        nifisdf2vtk(os.path.join(output_dir, image+"_sdf.nii"), os.path.join(output_dir, image+".vtk"))


    


if __name__ == "__main__":
    images_txt = "/work3/bmsha/quality_control/holes.txt"
    image_dir = "/work3/bmsha/sdf_lq/"
    output_dir = "/work3/bmsha/quality_control/fix_holes/"
    main(images_txt, image_dir, output_dir)