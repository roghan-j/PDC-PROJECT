# mpirun -np 5 python D2N2Accel.py
# Import statements
from mpi4py import MPI
import time
import numpy as np
import cv2
import os

comm= MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start = time.time()
# Paths to load the model
DIR = r"C:/Users/rogha/OneDrive/Documents/GitHub/PDC_Colorize"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the Model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

s= 40

if rank==0:

    for i in range(s):    
        # Load the input image
        image = cv2.imread('./images/image'+str(i)+'.jpg')
        # cv2.imshow("Original", image)

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        data_1= [image,L]
        comm.send(data_1, dest=1)

        data_2= [lab]
        comm.send(data_2, dest=2)


elif rank==1:
    for i in range(s):
        data_0= comm.recv(source=0)
        
        net.setInput(cv2.dnn.blobFromImage(data_0[1]))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (data_0[0].shape[1], data_0[0].shape[0]))

        data_2= [ab]
        comm.send(data_2, dest=2)

elif rank==2:
    for i in range(s):
        data_0= comm.recv(source=0)
        data_1= comm.recv(source=1)
        
        L = cv2.split(data_0[0])[0]
        colorized = np.concatenate((L[:, :, np.newaxis], data_1[0]), axis=2)

        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        colorized = (255 * colorized).astype("uint8")

        # cv2.imshow("Colorized", colorized)
        # cv2.waitKey()
        end = time.time()

    print(end-start)
