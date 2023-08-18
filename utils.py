import numpy as np
import json
import os
import cv2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_image(save_results, page_number, cropped_image, bbox):
    # Save cropped images
    if save_results:
        print("Saving Cropped Images ...")
        save_cropped_images = f"{save_results}/page_number_{page_number}/cropped_image/"
        if not os.path.exists(save_cropped_images):
            os.makedirs(save_cropped_images)
        cv2.imwrite(
            f"{save_cropped_images}/croped_images_{bbox}.png",
            cropped_image,
        )
