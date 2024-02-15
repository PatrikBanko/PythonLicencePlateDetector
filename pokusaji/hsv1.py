import cv2
import numpy as np
import time

from hsv2 import *

img = cv2.imread("slike/5.jpg")
start = time.time()


detection, crops = detect(img)

i = 1
for crop in crops:

    crop = process(crop)

    cv2.imwrite('temp/crop' + str(i) + '.jpg', crop)
    recognise('temp/crop' + str(i) + '.jpg', 'temp/crop'+str(i))
    post_process('temp/crop' + str(i) + '.txt')
    i += 1
cv2.imwrite('temp/detection.jpg', detection)
finish = time.time()
print('Time processing >>>>>>  '+ str(finish-start))

# cv2.imshow("thresh", img_changed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()