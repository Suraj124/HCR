import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageOps
import cv2
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt

st.set_option('deprecation.showfileUploaderEncoding', False)

# Set the Page Tite
st.set_page_config(
    page_title="Hindi Character Recognition",
    page_icon= "🔎"
)

# Load the Model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

hindi_character = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'फ', 'ब', 'भ', 'म',\
                    'य', 'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'ॠ', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ',\
                    'ज', 'झ', '0', '१', '२', '३', '४', '५', '६', '७', '८', '९']

with st.spinner("Model is being loaded..."):
    model = load_model()

# Side Navigation
with st.sidebar:
    sel = option_menu(
        menu_title="Navigation",
        options=["Home", "Prediction","Get test images"],
        icons='house book download'.split(),
        menu_icon='cast',
        default_index=0,
        orientation='vertical'
    )

def load_and_prep(file):

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)  # Return ndarray
    img = tf.image.resize(opencv_image,size=[32,32])  # Resize the to 32 by 32

    return img

def get_n_predictions(pred_prob,n):

    pred_prob = np.squeeze(pred_prob)
    
    top_n_max_idx = np.argsort(pred_prob)[::-1][:n]
    top_n_max_val = list(pred_prob[top_n_max_idx])
    
    top_n_class_name=[]
    for i in top_n_max_idx:
        top_n_class_name.append(hindi_character[i])
    
    return top_n_class_name,top_n_max_val



# Show Animated image
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if sel =='Home':
    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_4asnmi1e.json") # Link for animated image
    st_lottie(lottie_hello)
    st.title("Hindi Character Recognition")

if sel =='Prediction':
    # lottie_pred = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_jz4fqbbk.json")
    # st_lottie(lottie_pred)

    file = st.file_uploader("Upload an Image")

    if file is None:
        st.write("## Please Upload an Image")
    else:

        img = load_and_prep(file)

        fig,ax = plt.subplots()
        ax.imshow(img.numpy().astype('uint8'))
        ax.axis(False)
        st.pyplot(fig)

        pred_prob = model.predict(tf.expand_dims(img,axis=0))
        st.write("### Select the top n predictions")
        n=st.slider('n',min_value=1,max_value=5,value=3,step=1)
        class_name , confidense = get_n_predictions(pred_prob,n)

        if st.button("Predict"):


            st.header(f"Top {n} Prediction for given image")

            fig = go.Figure()

            fig.add_trace(go.Bar(
                    x=confidense[::-1],
                    y=class_name[::-1],
                    orientation='h'))
            fig.update_layout(height = 500 , width = 900, 
                        xaxis_title='Probability' , yaxis_title=f'Top {n} Class Name')
            
            st.plotly_chart(fig,use_container_width=True)


            st.success(f"The image is classified as \t  \'{class_name[0]}\' \t with {confidense[0]*100:.1f} % probability")


        # st.success(hindi_character[pred_prob.argmax()]+str(tf.reduce_max(pred_prob).numpy()))


if sel =='Get test images':
    st.write('# Download the test images')
    lottie_download = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_szdrhwiq.json") # Link for animated image
    st_lottie(lottie_download)

    # https://stackoverflow.com/questions/71589690/streamlit-image-download-button
    # For download button, code is take from above link
    st.download_button(
        label='Download the image of क',
        data = open('img/ka.png', 'rb').read(),
        file_name="ka.png",
        mime='image/png'

    )
    st.download_button(
        label='Download the image of ख',
        data = open('img/kha.png', 'rb').read(),
        file_name="kha.png",
        mime='image/png'

    )
    st.download_button(
        label='Download the image of ग',
        data = open('img/ga.png', 'rb').read(),
        file_name="ga.png",
        mime='image/png'

    )
    st.download_button(
        label='Download the image of ध',
        data = open('img/dha.png', 'rb').read(),
        file_name="dha.png",
        mime='image/png'

    )
    st.download_button(
        label='Download the image of ङ',
        data = open('img/da.png', 'rb').read(),
        file_name="da.png",
        mime='image/png'

    )
