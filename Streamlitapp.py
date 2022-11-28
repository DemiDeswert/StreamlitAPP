import streamlit as st
import pickle
from PIL import Image
import io

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        return uploaded_file
    else:
        return None
def main():
    st.title('Pretrained model demo')
    image = load_image()
    result = st.button('Run on image')
    model = pickle.load(open('model.pkl', 'rb'))
    if result:
        st.write('Calculating results...')
        model.predict(image, confidence=40, overlap=30).save("prediction.jpg")
        prediction = Image.open('prediction.jpg')
        st.image(prediction)

if __name__ == '__main__':
    main()
