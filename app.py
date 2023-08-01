#Import the Streamlit library, which is used to create interactive web applications with Python.
import streamlit as st
#Import the PyTorch library, which is used for deep learning and neural networks.
import torch
#Import the detect module, which presumably contains custom functions related to the detection process.
import detect
#Import the Image class from the Python Imaging Library (PIL) to work with images.
from PIL import Image
#Import necessary input-output functions.
from io import *
#Import the glob module, which is used to retrieve filenames that match specified patterns.
import glob
#Import the datetime class to work with date and time.
from datetime import datetime
#Import the os module to interact with the operating system (e.g., file path operations).
import os
#Import the wget library, which allows downloading files from the internet.
import wget
#Import the time module to work with time-related operations.
import time

#Define a function imageInput that takes a single argument src representing the source of the image (either 'Upload your own Image' or 'From test Images').
def imageInput(src): 
  if src == 'Upload your own Image':       # Check if the selected image source is 'Upload your own Image'.
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])     #Create a file uploader widget to allow the user to upload an image. It only accepts images of type PNG, JPEG, or JPG. The uploaded image will be stored in the variable image_file.
    col1, col2 = st.columns(2)       #Create two columns for layout purposes using Streamlit's columns function.    
    if image_file is not None:     #Check if an image has been uploaded.
      img = Image.open(image_file)     #Open the uploaded image using the PIL Image.open method and store it in the variable img.
      with col1:     #Enter the first column context.
        st.image(img, caption='Uploaded Component Image', use_column_width=True)#Display the uploaded image with the specified caption and adjust the width of the image to fit the column.
    ts = datetime.timestamp(datetime.now()) #Get the current timestamp using the datetime.now() function and then convert it to a timestamp using datetime.timestamp(). This timestamp is used as part of the image filename for saving purposes.
    imgpath = os.path.join('data/uploads', str(ts)+image_file.name)#Create a file path where the uploaded image will be saved. The image will be saved in the 'data/uploads' directory with the timestamp and the original filename combined.
    outputpath = os.path.join('data/outputs', os.path.basename(imgpath)) #Create a file path where the predicted image will be saved. The predicted image will be saved in the 'data/outputs' directory with the original filename.
    with open(imgpath, mode="wb") as f: #Open the image path in binary write mode.
      f.write(image_file.getbuffer()) #Write the image data to the file to save it.

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='ecomp_1st/weights/best.pt', force_reload=True) #Load the custom YOLOv5 model using PyTorch's torch.hub.load method from the 'ultralytics/yolov5' repository. The model weights are loaded from the 'ecomp_1st/weights/best.pt' file.
    pred = model(imgpath) # Perform prediction on the uploaded image using the YOLOv5 model and store the results in the variable pred.
    pred.render() #Render the predicted bounding boxes on the image.
    for im in pred.ims: #Iterate through the images with rendered bounding boxes.
      im_base64 = Image.fromarray(im) #Convert each image array to a PIL Image.
      im_base64.save(outputpath) #Save the PIL Image with bounding boxes to the output path.
    img_ = Image.open(outputpath) #Open the predicted image from the output path and store it in the variable img_.
    
    with col2: #Enter the second column context.
      st.image(img_, caption='AI Electronic Component Recogniser', use_column_width=True) # Display the predicted image with the specified caption.
  elif src == 'From test Images':# If the selected image source is 'From test Images', execute the following block.
    imgpath = glob.glob('images/*') # Get the file paths of all images in the 'images' directory using the glob function and store them in the variable imgpath.
    imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) # Create a slider widget to allow the user to select a random image from the test set. The minimum value of the slider is 1, the maximum value is the total number of images, and the step is 1.
    image_file = imgpath[imgsel-1]# Get the selected image file path based on the value of the slider.
    submit = st.button("Predict Electronic Component Name!")#Create a button widget labeled "Predict Electronic Component Name!" that the user can click to trigger the prediction process.
    col1, col2 = st.columns(2) #Create two columns for layout purposes using Streamlit's columns function.
    with col1: # Enter the first column context.
      img = Image.open(image_file) # Open the selected test image using the PIL Image.open method and store it in the variable img.
      st.image(img, caption='Selected Image', use_column_width='always') #Display the selected test image with the specified caption and adjust the width of the image to fit the column.
    with col2: #Enter the second column context.
        if image_file is not None and submit: #Check if a test image has been selected and the "Predict Electronic Component Name!" button has been clicked.
          model = torch.hub.load('ultralytics/yolov5','custom',path='ecomp_1st/weights/best.pt', force_reload=True) #Load the custom YOLOv5 model using PyTorch's torch.hub.load method from the 'ultralytics/yolov5' repository. The model weights are loaded from the 'ecomp_1st/weights/best.pt' file.
          pred = model(image_file) #Perform prediction on the selected test image using the YOLOv5 model and store the results in the variable pred.
          pred.render() #Render the predicted bounding boxes on the image.
          for im in pred.ims: #Iterate through the images with rendered bounding boxes.
            im_base64 = Image.fromarray(im)# Convert each image array to a PIL Image.
            im_base64.save(os.path.join('data/outputs', os.path.basename(image_file))) #Save the PIL Image with bounding boxes to the output directory with the original filename.
            img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))#Open the predicted image from the output directory with the original filename and store it in the variable img_.
            st.image(img_, caption='AI Electronic Component Predictions')# Display the predicted image with the specified caption.
def main():#Define the main function, which acts as the entry point of the application.
    st.image("logo.JPG", width = 500)# Display an image (presumably the logo) with a specified width of 500 pixels.
    st.title("Coresys Limited")# Display the title "Coresys Limited."
    st.header("AI Tool for Electronic Component Recognition")# Display a header with the text "AI Tool for Electronic Component Recognition."
    st.header("👈🏽 Select the Image Source options")# Display a header with the text "👈🏽 Select the Image Source options."
    st.sidebar.title('⚙️Options')# Display a title in the sidebar with the text "⚙️Options."
    src = st.sidebar.radio("Select input source.", ['From test Images', 'Upload your own Image'])# Create a radio button widget in the sidebar to allow the user to select the image source. The available options are 'From test Images' and 'Upload your own Image'. The selected option will be stored in the variable src.
    imageInput(src)# Call the imageInput function with the selected image source (src) as an argument.
if __name__ == '__main__': #Check if the script is being run as the main module.
  main()# Call the main function to start the Streamlit web application.
loadModel()#Call the loadModel function to start the model download process when the script is run. Note that this line is currently commented out (using #@st.cache), so it won't be executed. The function loads a pre-trained YOLOv5 model by downloading it from the provided URL.
