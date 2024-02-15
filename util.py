import easyocr
import re
import cv2
import math
import numpy as np


# Initialize the OCR reader
reader = easyocr.Reader(["hr"])

# Mapping dictionaries for character conversion
dict_char_to_int = {
    "O": "0",
    "I": "1",
    "J": "3",
    "A": "4",
    "G": "6",
    "S": "5",
    "L": "1",
}

dict_int_to_char = {
    "0": "O",
    "1": "I",
    "3": "J",
    "4": "A",
    "6": "G",
    "5": "S",
    "2": "Z",
    "7": "Z",
    "8": "B",
}


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) < 6 or len(text) > 8:
        return False
    else:
        return True

    # if (
    #     (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
    #     (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
    #     (text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] or text[2] in dict_char_to_int.keys()) and
    #     (text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] or text[3] in dict_char_to_int.keys()) and
    #     (text[4] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] or text[4] in dict_char_to_int.keys()) and
    #     (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys() or text[5] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] or text[5] in dict_char_to_int.keys()) or
    #     (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) or
    #     (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())
    # ):
    #     return True
    # else:
    #     return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ""

    if len(text) == 6:
        mapping = {
            0: dict_int_to_char,
            1: dict_int_to_char,
            2: dict_char_to_int,
            3: dict_char_to_int,
            4: dict_char_to_int,
            5: dict_int_to_char,
        }
        for j in [0, 1, 2, 3, 4, 5]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_
    elif len(text) == 7:
        mapping = {
            0: dict_int_to_char,
            1: dict_int_to_char,
            2: dict_char_to_int,
            3: dict_char_to_int,
            4: dict_char_to_int,
            5: dict_int_to_char,
            6: dict_int_to_char,
        }
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_
    else:
        mapping = {
            0: dict_int_to_char,
            1: dict_int_to_char,
            2: dict_char_to_int,
            3: dict_char_to_int,
            4: dict_char_to_int,
            5: dict_char_to_int,
            6: dict_int_to_char,
            7: dict_int_to_char,
        }
        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_


# def recognise(src_path, out_path):

#     pytesseract.pytesseract.tesseract_cmd = (
#         r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#     )
#     text = pytesseract.image_to_string(src_path)

#     if license_complies_format(text):
#         return format_license(text)

#     return None


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    plate_concat = ""
    for detection in detections:
        bbox, text_dirty, score = detection

        plate_concat += text_dirty

        plate_concat = plate_concat.upper().replace(" ", "")

        text = re.sub(r"[^a-zA-Z0-9]", "", plate_concat)
        print(text, score)

        if license_complies_format(text):
            return format_license(text), score

    return None, None


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
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:  # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi


def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta
