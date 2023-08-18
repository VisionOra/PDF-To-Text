import json
import os
import re
import uuid

import cv2
import imgproc
import numpy as np
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path

from BERT_Embeddings.embedings import get_embeddings
from text_detector.imgproc import PIL2array
from text_detector.text_detector import detector, load_default_model
from translation.chinese_to_english import translate
from utils import NumpyEncoder, save_image

load_dotenv()
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
# Loading text detector model
load_default_model()


def main(pdf_path="pdf/maintenance manual_Aion LX.pdf", save_results="test/"):
    """
    Main function to read and parse a PDF user manual.
    """
    global TESSERACT_LANG

    # Parsing PDF
    images = convert_from_path(pdf_path)
    name = re.findall(r"pdf/(.*).pdf", pdf_path)[0]
    print(images[1], type(images[1]), len(images))

    if images:
        for page_number, image in enumerate(images):
            results, text_to_ignore = [], []
            print("Detecting text in images...")
            image = PIL2array(image)
            bboxes, polys, score_texts = detector(image)
            print("Extracting text from images...")
            bboxes = np.asarray(bboxes, dtype=np.int32)
            # Convert boxes to x1, x2, x3, x4
            bboxes = [[box[0][0], box[0][1], box[2][0], box[2][1]] for box in bboxes]
            # Sort bbox by index 1 (y1)
            bboxes = sorted(bboxes, key=lambda x: x[1])
            # Getting headers and Footers (top 3 and bottom 3)
            headers = bboxes[:3]
            footers = bboxes[-3:]
            # Remove headers and footers
            bboxes = bboxes[3:-3]

            # Extracting text from headers and footers
            for bbox in headers + footers:
                x1, y1, x2, y2 = bbox
                cropped_image = image[y1:y2, x1:x2]

                # Extract text from croped images
                text = pytesseract.image_to_string(
                    cropped_image, lang=TESSERACT_LANG
                ).strip("\n")
                print("Extracted text: ", text)
                text_to_ignore.append(text)
                save_image(
                    save_results, str(page_number) + "_Ignore_text", cropped_image, bbox
                )

            # Extracting text from images
            for bbox, score_text in zip(bboxes, score_texts):
                x1, y1, x2, y2 = bbox
                cropped_image = image[y1:y2, x1:x2]

                # Extract text from croped images
                text = pytesseract.image_to_string(
                    cropped_image, lang=TESSERACT_LANG
                ).strip("\n")

                # Translate to english
                translated_text = translate(text)
                (
                    quantised_ch_embeddings,
                    normal_ch_embeddings,
                    min_max_array_per_column_ch,
                ) = get_embeddings(text, "ch")
                (
                    quantised_en_embeddings,
                    normal_en_embeddings,
                    min_max_array_per_column_en,
                ) = get_embeddings(translated_text, "ch")

                print("Extracted text: ", TESSERACT_LANG)

                result = {
                    "id": str(uuid.uuid4()),
                    "display": "Picture cloud storage Path",
                    "bbox": [str(i) for i in [x1, y1, x2, y2]],
                    "text": text,
                    "text_to_ignore": text_to_ignore,
                    "text_en": translated_text,
                    "score_text": str(score_text),
                    "text_ch_bert": normal_ch_embeddings,
                    "text_ch_bert_qq": quantised_ch_embeddings,
                    "text_en_bert": normal_en_embeddings,
                    "text_en_bert_qq": quantised_en_embeddings,
                    "text_en_bert_qq_min_max": min_max_array_per_column_en,
                    "text_ch_bert_qq_min_max": min_max_array_per_column_ch,
                }
                results.append(result)
                save_image(save_results, page_number, cropped_image, bbox)
            # Saving Results in Json
            save_json_at = f"{save_results}/page_number_{page_number}/"
            print("Saving results...")
            if not os.path.exists(save_json_at):
                os.makedirs(save_json_at)
            with open(f"{save_json_at}/results.json", "w") as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
