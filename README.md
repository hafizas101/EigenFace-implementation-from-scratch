# EigenFace-implementation-from-scratch
This repository provides Python implementation of EigenFace method proposed by M.Turk and A.Pentland in 1991 for the task of face recognition. The method has been implemented from scratch and only three libraries namely Numpy, MatPlotLib and OpenCV have been used. Place all your training images in the folder train_images. For more accurate and fast implementation, images should be grayscale and must be cropped at the face such as present in the folder. This decreases redundant information. The file main.py implements the desired function which takes image path as argument and returns the integer person ID. <br/>
width, height = the sizes of training face image and this will be the size of test region of interest as well.<br/>
k = Number of largest eigenvectors

## Explanation for train portion of code
Coming towards face_rec function, first of all we train our algorithm and give the location of folder train_images. The variable file name contains names of all training images which will be helpful for us later to get person id. Then we create a G numpy array that stores all train image vectors. Then we calculate the mean face vector and plot it.
<p align="center">
  <img width="436" height="424" src="https://github.com/hafizas101/EigenFace-implementation-from-scratch/blob/master/mean_face.png">
</p>
Then we subtract mean face from G array to calculate normalized train image vectors. Then we calculate covariance matrix of normalized images. Then we calculate eigen-values and eigen-vectors of covariance matrix. These eigen values and eigen vectors are sorted in descending order. The k best eigen vectors are chosen and image space is projected on the k dimensions. So we calculate projected data by taking dot product of train image vectors G with k largest eigenvectors. The project data is multiplied with normalized image vectors to calculate train image weights (w). <br/>

## Explanation for test portion of code
Now the testing process starts, where we first of all resize the given test image to dimension of 893 X 1190 so that whatever smartphone camera has been used, we resize the test image to desired dimension. Then we use pre-trained haar cascade frontal default xml file to detect faces. There may be more than one faces in the test image. So we chose the face that has highest height and width and this should be our desired face that we want to recognize. Then we resize this desired face region to the dimension of width X height (400 X 400 in our case). Then we calculate flattened test image vector, normalized vector and finally weight of test image. Then we compare the test weight to the weights of all train images to create diff of size number of images X k (495 X 5) in our case. Then we normalize it along the column to calculate errors with respect to all train images. Then we find the location of minimum value of the error which corresponds to the index of the person. Then we pass this index to the filenames which gives us the name of file and from the characters of this name, we obtain the person_id which is returned by the person.
