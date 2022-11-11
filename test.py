# Import statements
from mpi4py import MPI
import time
import numpy as np
import cv2
import os

comm= MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Paths to load the model
DIR = r"C:/Users/rogha/OneDrive/Documents/GitHub/PDC_Colorize"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

images= ["building", "einstein", "nature", "tiger"]
s= len(images)

if rank==0:

    for i in range(s):    
        # Load the input image
        image = cv2.imread('./images/'+images[i]+'.jpg')
        # cv2.imshow("Original", image)
        # cv2.waitKey()
        send_1= i
        comm.send(send_1, dest=1)

elif rank==1:
    for i in range(s):
        print("Worker1 working")
        data= comm.recv(source=0)
        print("Printing: ",data)

elif rank==2:
    print("Worker2 working")

