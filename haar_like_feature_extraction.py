import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize 
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

#resizing image
img = imread('images/2.jpg')
img = resize(img, (100, 100))

feature_types = ['type-2-y']

feat_t = feature_types[0]
coord, _ = haar_like_feature_coord(img.shape[0], img.shape[1], feat_t)

haar_feature = draw_haar_like_feature(img, 0, 0,
                                        img.shape[0],
                                        img.shape[1],
                                        coord,
                                        max_n_features=20,
                                        random_state=0)

#plt.imshow(haar_feature)
plt.imsave("outputs/haar_features.jpg", haar_feature, cmap="gray")
#plt.show()