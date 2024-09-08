# TextPredict: Deep Learning-based Next Word Prediction

This project uses an LSTM (Long Short-Term Memory) neural network model to generate text sequences based on a given seed input. The model is trained on the **Sherlock Holmes stories** text dataset. It uses the **Keras** library in TensorFlow for text processing and model training.

## Overview

The script performs the following tasks:

1. **Data Preprocessing**: 
   - Reads the text from a file (`sherlock-holm.es_stories_plain-text_advs.txt`).
   - Tokenizes the text, converts it into sequences of integers, and creates input sequences for training the model.
   - Pads the sequences to ensure uniform length for training.
   
2. **Model Creation**: 
   - Builds a Sequential model with an **Embedding** layer, an **LSTM** layer, and a **Dense** layer with a softmax activation function for predicting the next word.
   
3. **Training**: 
   - The model is trained using categorical cross-entropy loss and the Adam optimizer for 100 epochs.

4. **Text Generation**: 
   - After training, the model generates a sequence of words given a seed text input using the trained model.
  
## Model Training Results

The model was trained for 100 epochs using the **Sherlock Holmes stories** dataset. The final training accuracy and loss are as follows:

- **Accuracy**: 87.40% (0.8740)
- **Loss**: 0.4759

### Training Time

Each epoch took approximately 66 seconds, with each batch taking around 22 milliseconds to process. The training was run with 3010 batches per epoch.

## Dependencies

The following libraries are required to run the script:

- `numpy`
- `tensorflow`
- `keras` (Keras is included in TensorFlow 2.x)

You can install the necessary libraries with the following command:

```bash
pip install numpy tensorflow
