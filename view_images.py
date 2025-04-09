import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Quick image viewer for manual quality check and annotation."
    )
    parser.add_argument("folder", type=str, help="Folder containing .png images", default="/work3/bmsha/quality_control/figures/outlier_visual/images/")
    parser.add_argument("--output", type=str, default="/work3/bmsha/quality_control/figures/outlier_visual/outliers.txt",
                        help="Output text file for selected image names")
    args = parser.parse_args()
    
    # List and sort all .png files in the folder
    folder = args.folder
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    
    if not image_files:
        print(f"No .png files found in folder: {folder}")
        return

    num_images = len(image_files)
    current_index = 0

    # Open the output file in append mode
    with open(args.output, 'a') as f_out:
        while True:
            # Build full path and load the image
            image_path = os.path.join(folder, image_files[current_index])
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                current_index = (current_index + 1) % num_images
                continue

            # Optionally, add an overlay with the current index and filename
            display_img = img.copy()
            overlay_text = f"[{current_index+1}/{num_images}] {image_files[current_index]}"
            cv2.putText(display_img, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the image in a window
            cv2.imshow("Image Viewer", display_img)
            
            # Wait for a key press
            key = cv2.waitKey(0) & 0xFF
            
            # Key mappings:
            #   ESC (27) -> exit
            #   Left arrow -> previous image
            #   Right arrow -> next image
            #   Space bar (32) -> select current image and append its name (without .png) to output file
            
            if key == 27:  # ESC key
                break
            elif key in [81, 2424832]:  # left arrow key (different codes may appear depending on OS)
                current_index = (current_index - 1) % num_images
            elif key in [83, 2555904]:  # right arrow key
                current_index = (current_index + 1) % num_images
            elif key == 32:  # space bar
                # Write filename without extension
                filename_no_ext = os.path.splitext(image_files[current_index])[0]
                f_out.write(filename_no_ext + "\n")
                f_out.flush()
                print(f"Selected: {filename_no_ext}")
                # Optionally, you can automatically move to the next image after selection:
                current_index = (current_index + 1) % num_images

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()