from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2


width  = 400
height = 400
k=5

def face_rec(image):
    # ---------------------- Training --------------------------


    filenames = glob.glob("train_images/*.jpg")
    filenames.sort()
    # print(filenames)
    images = [cv2.imread(img, 0) for img in filenames]
    M = len(images)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    G = np.ndarray(shape=(M, height * width), dtype=np.float64)
    for i in range(M):
        G[i, :] = np.array(images[i], dtype='float64').flatten()

    mean_face = np.zeros((1, height * width))

    for i in G:
        mean_face = np.add(mean_face, i)

    mean_face = np.divide(mean_face, float(M)).flatten()

    normalised_images = np.ndarray(shape=(M, height * width))

    for i in range(M):
        normalised_images[i] = np.subtract(G[i], mean_face)

    cov_matrix = np.cov(normalised_images)
    cov_matrix = np.divide(cov_matrix, M)

    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

    reduced_data = np.array(eigvectors_sort[:k]).transpose()
    proj_data = np.dot(G.transpose(), reduced_data)
    proj_data = proj_data.transpose()

    w = np.array([np.dot(proj_data, i) for i in normalised_images])



    # --------------------------- Testing ---------------------------




    test = cv2.resize(image, (893,1190), interpolation=cv2.INTER_CUBIC)
    faces = face_classifier.detectMultiScale(test, 1.3, 5)
    if len(faces) == 0:
        print("TEST Face not detected !")

    max_loc = np.where(faces[:, 3] == np.amax(faces[:, 3]))
    for (x, y, w1, h1) in faces[max_loc[0]]:
        unknown_face = test[y:y + w1, x: x + h1]
        unknown_face = cv2.resize(unknown_face, (width, height), interpolation=cv2.INTER_CUBIC)

    # plt.imshow(unknown_face)
    # plt.show()

    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()

    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    w_unknown = np.dot(proj_data, normalised_uface_vector)

    diff = w - w_unknown
    # print(diff.shape)
    errors = np.linalg.norm(diff, axis=1)
    min_err = np.amin(errors)
    # if error[0,i] > 900000000:
    #   print("Unknown person !")
    #    continue
    index = np.argmin(errors)
    person_id = filenames[index][8:10]
    return person_id


# -------- Replace test.jpg with the image you want to test -------

img = cv2.imread('test.jpg',0)
person_id = face_rec(img)
print("Person ID: {}".format(person_id))





