# Automatic License Plate Recognition (ALPR) with YOLOv11 and EasyOCR

This repository contains an implementation of an Automatic License Plate Recognition (ALPR) system using the YOLOv11 model for license plate detection and EasyOCR for text recognition. The project processes video streams and test images to detect and recognize license plates, saving annotated images and JSON metadata. The dataset is a merged collection of Pakistani license plate datasets, preprocessed using custom Python scripts, and trained in a Google Colab environment.

## Overview

The ALPR system is designed to:
- Detect license plates in video frames or images using a pre-trained YOLOv11 model.
- Extract text from detected plates using EasyOCR.
- Annotate frames or images with recognized text and save them to specified directories.

The project is optimized for use in Google Colab with GPU support, leveraging open-source tools for scalability and accessibility. The training, testing, and OCR processes are detailed in the `License_plate_detection.ipynb` notebook.

## Dataset

The merged dataset, named "Pakistani License Plates (Merged) - YOLOv11," was created by combining the following publicly available datasets from Roboflow:

- [Pakistani License Plates](https://universe.roboflow.com/shaheryar-ekgdy/pakistani-license-plates/dataset/2)
- [Numberplate](https://universe.roboflow.com/yolov5-hvpdd/numberplate-x0cle/dataset/1/images)
- [Car Number Plate Detection](https://universe.roboflow.com/nouman-khan-on-ai/car-number-plate-detection-3om4v/dataset/1)
- [Pakistani Number Plates](https://universe.roboflow.com/malik-kashif-saeed-aswwf/pakistani-number-plates/dataset/1/images)
- [Model](https://universe.roboflow.com/khurram-iayhn/model-ubglw/dataset/1)

These datasets were preprocessed to ensure compatibility with YOLOv11 format, including image and label organization into `train`, `valid`, and `test` splits.

## Prerequisites

- **Google Colab**: A free cloud-based Python environment with GPU support.
- **Python 3.x**: Ensure compatibility with required libraries.
- **Google Drive**: For storing annotated images and JSON outputs (if used).
- **Required Libraries**:
  - `ultralytics` (for YOLOv11)
  - `easyocr`
  - `opencv-python`
  - `numpy`
  - `json`
  - `os`
  - `pathlib`
  - `datetime`
  - `shutil`
  - `math`
  - `re`
  - `subprocess` (for video compression)

## Setup

1. **Mount Google Drive** (if using video processing variant):
   - Run the Colab notebook and authorize Google Drive access when prompted.

2. **Install Dependencies**:
   - Execute the installation commands provided in the notebook or scripts (e.g., `!pip install ultralytics easyocr`) to install required libraries.
   - Ensure `ffmpeg` is available for video compression (pre-installed in Colab).

3. **Upload Files**:
   - Upload your dataset to `/content/Pakistani License Plates (Merged) - YOLOv11/`.
   - Upload your video file (e.g., `carLicence1.mp4`) to `/content/data/` or test images to the appropriate split directory.
   - Upload your pre-trained YOLOv11 model (e.g., `best.pt`) to `/content/runs/detect/train/weights/` or use the `yolo_backup_runs` directory.

4. **Set GPU Runtime**:
   - Go to `Runtime > Change runtime type > Hardware accelerator > GPU` in Colab.

## Usage

### Running the ALPR System

1. **Training and Testing**:
   - Open `License_plate_detection.ipynb` in Colab.
   - Follow the notebook to train the YOLOv11 model on the merged dataset, test on the `test` split, and apply EasyOCR to the `predicted_images_easyocr` directory.
   - The notebook includes the full pipeline for model training, validation, and OCR application.

2. **Processing Test Images**:
   - Run the notebook to process test images from `Pakistani License Plates (Merged) - YOLOv11/images/test`.
   - Annotated images with OCR results are saved to `predicted_images_easyocr`.

3. **Source Inference**:
   - Use the `infer_from_source` function to process videos or images.
   - Example usage with tested videos:
     - [Lahore Roads Traffic 1](https://www.pexels.com/video/lahore-roads-traffic-17814090/)
     - [Lahore Roads Traffic 2](https://www.pexels.com/video/lahore-roads-traffic-17814089/)
     - [Islamabad Expressway](https://www.pexels.com/video/islamabad-expressway-19549786/)
   - Download the videos and upload to `/content/data/`, then call `infer_from_source("/content/data/video.mp4")` in a Colab cell.

### Output
- **Annotated Images**: Saved in `predicted_images_easyocr` for test images, with filenames reflecting the original image names and OCR text.
- **YOLO Runs**: Saved in `yolo_backup_runs` for reference, containing training and validation results.
- **Annotated Videos**: Saved in `predicted_output` as compressed `.mp4` files (e.g., `video_annotated_compressed.mp4`) after processing with `infer_from_source`.

## Preprocessing Scripts

The dataset was preprocessed using the following custom scripts, included in this repository:

1. **count_files.py**:
   - Counts the total number of files in a directory.
   - Example: `print(f"📦 Total files in '{path}':", count_files(path))`

2. **imgs_per_split.py**:
   - Calculates the number of images in `train`, `valid`, and `test` splits, with percentages.
   - Example output: `train → 700 images (70.0%)`

3. **merge_yolo_datasets.py**:
   - Merges multiple YOLO datasets into a single dataset with prefixed filenames to avoid conflicts.
   - Merges images and labels from specified dataset directories.

4. **remove_non_licenseplate_labels.py**:
   - Filters YOLO label files to keep only specified class IDs (e.g., `1` for license plates) and renames them to class `0`.

5. **split_dataset.py**:
   - Rebalances the dataset into `train` (70%), `valid` (20%), and `test` (10%) splits based on target proportions.

6. **validate_and_clean_yolo_labels.py**:
   - Validates YOLO label files for correctness (e.g., class ID range, coordinate validity) and optionally deletes invalid pairs.

These scripts ensure the dataset is properly formatted and validated for YOLOv11 training and inference.

## Inference Function

The `infer_from_source` function processes various input sources:
- **Videos**: Processes `.mp4` or `.avi` files, annotates frames, and saves compressed output videos.
- **Webcam**: Supports live webcam feed (source = `0` or device index).
- **Image Directories**: Annotates all images in a directory.
- **Single Images**: Annotates a single image file.

### Example Usage
```python
infer_from_source("/content/data/lahore_roads_17814090.mp4")  # Process a downloaded video
```

### Tested Videos
The function was tested on the following Pexels videos:
- [Lahore Roads Traffic 1](https://www.pexels.com/video/lahore-roads-traffic-17814090/)
- [Lahore Roads Traffic 2](https://www.pexels.com/video/lahore-roads-traffic-17814089/)
- [Islamabad Expressway](https://www.pexels.com/video/islamabad-expressway-19549786/)

Download the videos, upload to `/content/data/`, and run the function to generate annotated outputs.

## Example Output

An example of an annotated image is included in the repository. The license plate "LEE 5559" is detected and recognized, with the bounding box and text annotation applied.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m "Description of changes"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure any changes are tested in Colab and documented.

## License

This project is open-source under the [MIT License](LICENSE). Feel free to use, modify, and distribute, but please include the original license.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11.
- [JaidedAI](https://github.com/JaidedAI/EasyOCR) for EasyOCR.
- Roboflow community for providing the base datasets.
- [Pexels](https://www.pexels.com/) for providing test video sources.
