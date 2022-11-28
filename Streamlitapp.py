import streamlit as st
import pickle
from PIL import Image
import io
import os
from PIL import Image

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.title('Pretrained model demo')
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        st.write(file_details)
        img = load_image(image_file)
        with open(os.path.join("tempDir",image_file.name),"wb") as f: 
            f.write(image_file.getbuffer())         
    st.success("Saved File")
    image =load_image(image_file)
    result = st.button('Run on image')
    model = pickle.load(open('model.pkl', 'rb'))
    if result:
        st.write('Calculating results...')
        model.predict(image, confidence=40, overlap=30).save("./prediction/prediction.jpg")
        prediction = Image.open("./prediction/prediction.jpg")
        st.image(prediction)

if __name__ == '__main__':
    main()
