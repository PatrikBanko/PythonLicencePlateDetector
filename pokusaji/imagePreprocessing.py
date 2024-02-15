import math
import cv2
import numpy as np
import io

from PIL import Image
from matplotlib import pyplot as plt


def preprocesiraj(img):
    img_gray_lp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(
        img_gray_lp,
        200,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # img_gray_lp,
        # 64,
        # 255,
        # cv2.THRESH_BINARY_INV,
    )
    return img_binary_lp


def segment_characters(image):
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(
        img_gray_lp,
        200,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # img_gray_lp,
        # 64,
        # 255,
        # cv2.THRESH_BINARY_INV,
    )

    # img_binary_lp = cv2.adaptiveThreshold(
    #     img_gray_lp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    # )

    # img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    # img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:1, :] = 255
    img_binary_lp[:, 0:1] = 255
    img_binary_lp[72:73, :] = 255
    img_binary_lp[:, 330:331] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH / 10, LP_WIDTH / 2, LP_HEIGHT / 20, 3 * LP_HEIGHT / 4]
    # dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
    
    # dimensions = [LP_WIDTH / 10, LP_WIDTH / 2, LP_HEIGHT / 15, 2 * LP_HEIGHT / 3]
    # plt.imshow(img_binary_lp, cmap="gray")
    # plt.title("Contour")
    # plt.show()
    cv2.imwrite("contour.jpg", img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread("contour.jpg")

    x_cntr_list = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if (
            intWidth > lower_width
            and intWidth < upper_width
            and intHeight > lower_height
            and intHeight < upper_height
        ):
            x_cntr_list.append(
                intX
            )  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((100, 55))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY : intY + intHeight, intX : intX + intWidth]
            char = cv2.resize(char, (45, 90))

            cv2.rectangle(
                ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2
            )
            plt.imshow(ii, cmap="gray")
            plt.title("Predict Segments")

            # Make result formatted for classification: invert colors
            # char = cv2.subtract(255, char)
            # cv2.imshow("char", char)

            # Resize the image to 55x100 with black border
            char_copy[5:95, 5:50] = char
            char_copy[0:5, :] = 255
            char_copy[:, 0:5] = 255
            char_copy[95:100, :] = 255
            char_copy[:, 50:55] = 255

            img_res.append(
                char_copy
            )  # List that stores the character's binary image (unsorted)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    plt.savefig("segmented.jpg")
    # plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(
            img_res[idx]
        )  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def compute_skew(src_img):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print("upsupported image type")

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(
        img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True
    )
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, 30, minLineLength=w / 4.0, maxLineGap=h / 4.0
    )
    angle = 0.0
    nlines = lines.size

    # print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        # print(ang)
        if math.fabs(ang) <= 30:  # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi


def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))


def compress_image(image_path, quality):
    # Open the image
    original_image = Image.open(image_path)

    # Create a buffer to hold the compressed image without saving to disk
    buffer = io.BytesIO()

    # Save the image to the buffer with the desired quality
    original_image.save(buffer, format="JPEG", quality=quality)

    # Rewind the buffer to the beginning
    buffer.seek(0)

    # Re-open the compressed image from the buffer
    compressed_image = Image.open(buffer)

    # Access pixel values using getdata
    pixel_data = compressed_image.getdata()

    return compressed_image, pixel_data
