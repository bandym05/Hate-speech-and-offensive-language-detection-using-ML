import streamlit as st
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils  import pad_sequences
import pickle
from keras.models import load_model
import transformers
from PIL import Image

# Load the saved model
model = load_model('AI_model.h5')

# Load the tokenizer

tokenizer = model.layers[0].get_weights()[0]

# Function to preprocess test data
def preprocess(text):
    sequences = tokenizer.texts_to_sequences(text)
    X_test = pad_sequences(sequences, maxlen=2000)
    return X_test

# Function to predict the class labels
def predict(model, test_data):
    X_test = preprocess(test_data)
    y_pred = model.predict(X_test)
    y_classes = np.argmax(y_pred, axis=1)
    labels = {0:'Hate', 1:'Offensive', 2:'Neither'}
    y_pred = [labels[idx] for idx in y_classes]
    return y_pred

# Function to paraphrase the input statement
def paraphrase(text):
    t5 = transformers.BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    generated_text = t5.generate(input_ids=input_ids, max_length=200, do_sample=True, temperature=0.7)
    paraphrase = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return paraphrase


# Create a Streamlit app
st.title('Social Media App')

# Add a photo of the post
image = Image.open('123.jpg')
st.image(image)

# Add a like button
if st.button('Like'):
    st.write('You liked the post!')

# Add a share comment button
if st.button('Share Comment'):
    comment = st.text_input('Enter your comment:')

    # Predict the class of the comment
    y_pred = predict(model, comment)

    # If the comment is hate speech or offensive, prevent sharing
    if y_pred[0] == 'Hate' or y_pred[0] == 'Offensive':
        st.write('Your comment contains hate speech or offensive language.')
        st.write('Please do not share this comment.')
    else:
        # Paraphrase the comment
        paraphrase = paraphrase(comment)

        # Show the comment and the paraphrased sentence
        st.write('Your comment:', comment)
        st.write('Paraphrased sentence:', paraphrase)
        
