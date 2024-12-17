import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import myapp as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprication.showfileUploaderEncoding', False)
st.title("Image Classifier")
st.text("Provide URL of image for classification")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/MLOps/models/')
    return model

with st.spinner('Loading Model Into Memory....'):
    model = load_model()

classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge',
 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, [150,150])
    return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify...')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying....'):
        label = np.argmax(model.predict(decode_img(content)), axis=1)
        st.write(classes[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Image', use_columnwidth=True)