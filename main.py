import os
import cvzone
import h5py
import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
# Model configuration (unchanged)
with h5py.File('Resources/Model/keras_model.h5', 'r+') as f:
    model_config = f.attrs['model_config']
    model_config = model_config.replace('"groups": 1,', '')
    f.attrs['model_config'] = model_config

cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

# Load the arrow image
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
if imgArrow is None:
    print("Failed to load arrow image")
    exit(1)

# Import all the waste images
imgWasteList = []
pathFolderWaste = 'Resources/Waste'
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.shape[2] == 3:  # If the image is RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert to BGRA
        imgWasteList.append(img)
    else:
        print(f"Failed to load image: {path}")

# Create a blank image for initialization
blank_image = np.zeros((720, 1280, 4), dtype=np.uint8)
blank_image[:, :, 3] = 255  # Set alpha channel to fully opaque

# Import all the bin images
imgBinsList = []
pathFolderBins = 'Resources/Bins'
pathList = os.listdir(pathFolderBins)
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.shape[2] == 3:  # If the image is RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert to BGRA
        imgBinsList.append(img)
    else:
        print(f"Failed to load image: {path}")

# Updated classification dictionary
classDic = {
    0: None,  # Assuming 0 is for unclassified
    1: 4,  # Metal (Recyclable)
    2: 4,  # Plastic (Recyclable)
    3: 4,  # Paper (Recyclable)
    4: 2,  # E-waste (Hazardous)
    5: 3,  # Food (Food)
    6: 1,  # Wood (Residual)
    7: 3,  # Leaf (Food)
    8: 2,  # Hazardous
    9: 1,  # Fabric (Residual)
}

# Bin names for display
bin_names = {
    1: "Residual",
    2: "Hazadous",
    3: "Food",
    4: "Recyclable"
}

while True:
    success, img = cap.read()  # Capture frame-by-frame
    if not success:
        print("Failed to capture frame")
        continue

    imgResize = cv2.resize(img, (454, 340))
    imgBackground = cv2.imread('Resources/background.png')
    if imgBackground is None:
        print("Failed to load background image")
        break

    prediction = classifier.getPrediction(img)
    print(prediction)  # Print prediction result

    classID = prediction[1]

    # Initialize with blank image
    waste_image = blank_image.copy()

    if 0 <= classID - 1 < len(imgWasteList):  # Check if the index is valid
        waste_image = imgWasteList[classID - 1]
        #imgBackground = cvzone.overlayPNG(imgBackground, waste_image, (820, 100))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        binID = classDic[classID]

        # Add text to display bin name (in black, aligned above each bin)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (0, 0, 0)  # Black color

        if binID is not None and 0 <= binID - 1 < len(imgBinsList):
            imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[binID - 1], (895, 374))

            # Calculate text size and position
            text = bin_names[binID]
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            if binID == 1:  # Hazardous (top-left)
                text_x = 1095 + (200 - text_size[0]) // 2
                cv2.putText(imgBackground, text, (text_x, 355), font, font_scale, font_color, font_thickness,
                            cv2.LINE_AA)
            elif binID == 2:  # Residual (top-right)
                text_x = 1095 + (200 - text_size[0]) // 2
                cv2.putText(imgBackground, text, (text_x, 355), font, font_scale, font_color, font_thickness,
                            cv2.LINE_AA)
            elif binID == 3:  # Food (bottom-left)
                text_x = 895 + (200 - text_size[0]) // 2
                cv2.putText(imgBackground, text, (text_x, 555), font, font_scale, font_color, font_thickness,
                            cv2.LINE_AA)
            elif binID == 4:  # Recyclable (bottom-right)
                text_x = 1095 + (200 - text_size[0]) // 2
                cv2.putText(imgBackground, text, (text_x, 555), font, font_scale, font_color, font_thickness,
                            cv2.LINE_AA)

    else:
        print(f"Invalid classID: {classID}")

    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Displays
    cv2.imshow("Output", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

