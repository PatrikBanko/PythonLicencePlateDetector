import cv2
import csv
import shutil
import os
import cv2

from util import *
from ultralytics import YOLO

from util import read_license_plate


results = {}
images = []
vehicle_class = [2]
detections_vehicles = []
detections_licence_plates = []

source = "image_list.csv"
dst = "image_list_predicted.csv"
shutil.copy(source, dst)

saved_thresh_directory = "./extracted_plates/"
shutil.rmtree(saved_thresh_directory)

new_dir = "extracted_plates"
os.mkdir(new_dir)


if __name__ == "__main__":

    # load models
    coco_model = YOLO("./models/yolov8n.pt")
    license_plate_detector = YOLO("./models/license_plate_detector.pt")

    images_directory = "mini_test_1/"
    for filename in os.listdir(images_directory):
        f = os.path.join(images_directory, filename)
        images.append(f)

    # * run detection model
    extracted_licence_plates = []
    for img in images:
        image_ = cv2.imread(img)
        x = os.path.basename(img)

        vehicles = [2, 3, 5, 7]

        detections = coco_model(image_)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) not in vehicles:
                continue
            detections_.append(image_)
            break

        # detect license plates
        if len(detections_) == 0:
            continue

        license_plates = license_plate_detector(detections_[0])[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # crop license plate
            license_plate_crop = image_[int(y1) : int(y2), int(x1) : int(x2), :]
            cv2.imwrite(
                f"./extracted_plates/{os.path.basename(img)}",
                license_plate_crop,
            )

            if license_plate_crop is None:
                continue

            corrected_lighting, _, __ = automatic_brightness_and_contrast(
                license_plate_crop
            )

            denoise = cv2.fastNlMeansDenoisingColored(
                corrected_lighting, None, 10, 10, 7, 15
            )

            # read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(denoise)

            if license_plate_text is not None:
                True
            else:
                continue

            rows = []
            with open(dst, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)

            for row in rows:
                if row[0] == os.path.basename(img):
                    row.append(license_plate_text)
                    break

            with open(dst, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(rows)

                break
