# PDFParser

# User Manual Search

This Python script is used to parse a PDF user manual, extract text from it (even if it's a scanned document), and remove headers and footers. It uses the Tesseract OCR engine for text extraction and PyPDF2 for PDF handling.

# Prerequisites

Before running this script, you need to install some dependencies:

1. Python 3.10 or higher
   ```
   brew install python
   ```
2. Install any virtual environment manager (e.g. virtualenv, conda, etc.)
   1. I am using conda
   2. Install conda from https://docs.conda.io/en/latest/miniconda.html
   3. Create a virtual environment
      ```
      conda create --name pdf_parser python=3.8
      ```
   4. Activate the virtual environment
      ```
      conda activate pdf_parser
      ```
   5. Install the required packages
      ```
      pip install -r requirements.txt
      ```
3. Install the required packages
4. For Mac Users
   1. Install Tesseract
      ```
      brew install tesseract
      ```
   2. Install Poppler
      ```
      brew install poppler
      ```
5. For Linux (Ubuntu-18.04) Users
   1. Install tesseract
      ```
      sudo apt-get install tesseract-ocr
      ```
   2. Install poppler
      ```
      sudo apt-get install poppler-utils
      ```
6. Download the trained data file for Simplified Chinese
   ```
   sudo mkdir -p /usr/local/share/tessdata/
   sudo curl -L -o /usr/local/share/tessdata/chi_sim.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata
   ```
7. Download the text detector model
   ```
   sudo wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/1n/p')&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ" -O craft_mlt_25k.pth && rm -rf /tmp/cookies.txt
   ```
8. Run the script
   ```
   python main.py
   ```
