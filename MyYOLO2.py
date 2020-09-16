# import packages
import numpy as np
import argparse
import time
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the YOLO class labels
labelsPath = "data/obj.names"
LABELS = open(labelsPath).read().strip().split("\n")

# paths to the YOLO weights and model configuration
weightsPath = "newweight/small-larva_2000.weights"
configPath = "cfg/small-larva.cfg"

# initialize a list of colors to represent each possible class label
COLORS = (103, 220, 225)
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
# 	dtype="uint8")

# load our YOLO object detector
print("Processing...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load input image and grab its spatial dimensions
image = cv2.imread(args["image"])
# image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("Nilai blob: {}".format(blob.shape))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) form Image
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# print ("Nilai Confidence dari object ialah :", confidence)
		# print ("Nilai ID dari Class ialah :", classID)

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the size of the image
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# print ("Nilai dari B.Box ialah :", centerX," ", centerY, " ", width, " ", height)

			# use the center (x, y)-coordinates to get the top and and left
			#corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			print ("Nilai x dan y dari B.Box ialah :", x," ", y)

			# update our list of bounding box coordinates, confidences, and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		# color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), COLORS, 8)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		print(text)
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,2, COLORS, 8)
		# cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.2, COLORS, 2)

# Font type
font = cv2.FONT_HERSHEY_SIMPLEX
# Font Coordinate
org = (40, 480)
# Font Size
fontScale = 10
# Font color with format (B,G,R)
color = (206, 0, 0)
# Font Thickness
thickness = 20

image = cv2.putText(image, '{} larva '.format(len(idxs)), org, font, fontScale, color, thickness, cv2.LINE_AA)

# Write total of larva in the frame into larvaNumber.txt
with open("jumlahlarva.txt", "a") as myfile:
	myfile.write("Ada {} Larva yang terdeteksi\n".format(len(idxs)))

# show timing information on YOLO
print("YOLO took {:.6f} seconds".format(end - start))

# Save the output image
# cv2.imshow("Image.jpg", image)
cv2.imwrite("Image.jpg", image)
print ("ada", len(idxs), "Larva")
cv2.waitKey(0)
