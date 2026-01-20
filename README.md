# French-To-English-Translation-NLP
# NLP: Sequence-to-Sequence Translation and OCR

demonstrating two key concepts:

1.  **Sequence-to-Sequence (Seq2Seq) Model for Neural Machine Translation:** Building and training a character-level Seq2Seq model using Keras to translate French sentences into English.
2.  **Optical Character Recognition (OCR):** Utilizing `pytesseract` to extract text from images.

## Table of Contents

*   [Project Overview](#project-overview)
*   [Dataset](#dataset)
*   [Dependencies](#dependencies)
*   [Neural Machine Translation (Seq2Seq)](#neural-machine-translation-seq2seq)
    *   [Data Loading & Preparation](#data-loading--preparation)
    *   [Data Cleaning](#data-cleaning)
    *   [Model Architecture](#model-architecture)
    *   [Training](#training)
    *   [Inference](#inference)
*   [Optical Character Recognition (OCR)](#optical-character-recognition-ocr)
*   [Results & Observations](#results--observations)

## Project Overview

The primary goal of this notebook is to illustrate the process of creating a simple neural machine translation system and integrating OCR functionality. The translation model is a character-level sequence-to-sequence model, which is a foundational architecture in NLP for tasks like translation, chatbots, and text summarization. The OCR section provides a quick demonstration of extracting text from an image.

## Dataset

The neural machine translation part of this project uses a bilingual dataset `bilingual_pairs.txt`, consisting of French-English sentence pairs. For practical purposes and faster training, the dataset is limited to the first 140,000 entries.

## Dependencies

To run this notebook, you will need the following Python libraries:

*   `pandas`
*   `numpy`
*   `unicodedata`
*   `re`
*   `io`
*   `keras`
*   `tensorflow`
*   `Pillow` (PIL - Python Imaging Library)
*   `pytesseract`
*   `cv2` (OpenCV - for image processing in OCR)

Additionally, for OCR, the `tesseract-ocr` engine needs to be installed on the system.

## Neural Machine Translation (Seq2Seq)

### Data Loading & Preparation

The `bilingual_pairs.txt` file is loaded, and sentences are split into English and French components. The notebook then truncates the data to 140,000 pairs.

### Data Cleaning

A `clean_sentences` function is applied to both English and French sentences. This function performs several cleaning steps:

*   Converts text to lowercase.
*   Removes punctuation.
*   Removes non-alphabetic characters.
*   Removes non-printable characters.

After cleaning, the data is prepared for the Seq2Seq model by creating `input_dataset` (French sentences) and `target_dataset` (English sentences, prefixed with `\t` and suffixed with `\n`). Unique characters for both input and target languages are identified, and character-to-index and index-to-character mappings are created.

### Model Architecture

The Seq2Seq model consists of an Encoder and a Decoder:

*   **Encoder:** An LSTM layer that processes the input (French) sequence and returns its final hidden and cell states. These states summarize the input sequence.
*   **Decoder:** Another LSTM layer that takes the final states of the encoder as its initial states. It processes the target (English) sequence one character at a time. The output of the decoder LSTM is passed through a Dense layer with a `softmax` activation to predict the probability distribution over the target vocabulary.

Key hyperparameters include:

*   `batch_size`: 256
*   `epochs`: 100
*   `latent_dim`: 256 (dimensionality of the LSTM output space)

### Training

The model is compiled with `rmsprop` optimizer and `categorical_crossentropy` loss. It is trained for 100 epochs with a validation split of 0.2.

### Inference

Separate encoder and decoder models are built for inference. The `decode_sequence` function is used to translate new French sentences into English by:

1.  Encoding the input sentence to get its state vectors.
2.  Starting the decoder with a `\t` (start-of-sequence) token and the encoder's state vectors.
3.  Iteratively predicting the next character until a `\n` (end-of-sequence) token is generated or the maximum sequence length is reached.

## Optical Character Recognition (OCR)

This section demonstrates how to perform OCR using the `pytesseract` library. The necessary `tesseract-ocr` engine is installed, and `pytesseract` is set up. An example shows reading an image and extracting text from it.

**Note on `Pillow`:** An `AttributeError: module 'PIL.Image' has no attribute 'Resampling'` might occur due to a version mismatch. The notebook includes a step to explicitly install `Pillow==9.0.0` to resolve this.

## Results & Observations

*   The Seq2Seq model, after 100 epochs, shows a decreasing training loss (`0.6347`) and validation loss (`1.2871`), indicating learning. 
*   Translation examples show that the model can capture some meaning, but often produces literal or partially correct translations, demonstrating the complexity of character-level Seq2Seq for full-fledged machine translation. For instance, 'je me sens affreusement mal' translates to 'i feel like such an idiot', which is a reasonable output, while 'hier etait une bonne journee' translates to 'stop still the store', indicating areas for improvement.
*   The OCR functionality successfully extracts text from an example image, highlighting its utility for digitizing text from visual sources.

This notebook provides a good foundation for understanding and experimenting with Seq2Seq models and OCR techniques.
