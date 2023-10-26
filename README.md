# Search_by_summary

Project for segmentation of blood cells in medical images. Based on the U-net model architecture. The model was trained on data from the BCCD dataset. The BCCD dataset is a small dataset for detecting blood cells.
Segmentation involves dividing into 3 classes of cells and background
We have three kind of labels:

RBC (Red Blood Cell)
WBC (White Blood Cell)
Platelets

To obtain a labeled dataset, I used the utility Fiji. Fiji  is an image processing package—a “batteries-included” distribution of ImageJ2, bundling a lot of plugins which facilitate scientific image analysis.

To speed up the process, I used the Labkit plugin with the ability to generate an automatic classification model based on classical ML on several images. So it didn’t take me much time at all (I processed part of the dataset).

*In near future I will make awesome readme and docs...but still...*

## Implemented Tasks

1. Semantic Segmentation

## How to start? Running the app

Run the following commands and open the local host web address.

```shell
streamlit run app.py
```

An example of the expected terminal messages are shown below:

```shell
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.20.0.4:8501

## Project Structure

Segmentation_module/
├── experiments/
│   └── segmentation.ipynb
├── dataset.py  --loading datasets and data proccessing, making DataLoader
├── model.py  --architecture Of U-net model
├── train.py  --a process Of rtaining of U-net model
├── app.py  --main file
└──  data/
    ├── images_test
    ├── annotations_test
    ├── annotations_train
    ├── images_train
    └── images_for_TEST  --a folder for additional images for segmentation testing and visualization
