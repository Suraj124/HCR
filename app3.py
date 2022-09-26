import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageOps
import cv2
import plotly.graph_objects as go



@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

hindi_character = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'फ', 'ब', 'भ', 'म',\
                    'य', 'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'ॠ', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ',\
                    'ज', 'झ', '0', '१', '२', '३', '४', '५', '६', '७', '८', '९']



with st.spinner("Model is being loaded..."):
    model = load_model()

st.title("Hindi Character Recognition")

file = st.file_uploader("Upload an Image")
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_and_predict(uploaded_image,model,size=(32,32)):
    image = ImageOps.fit(uploaded_image,size,Image.LANCZOS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize =  cv2.resize(image, dsize=(32,32),interpolation=cv2.INTER_CUBIC)
    # image_reshape = image_resize[np.newaxis,...]
    pred_prob = model.predict(tf.expand_dims(image_resize,axis=0))
    pred_prob = np.squeeze(pred_prob)
    
    top_5_max_idx = np.argsort(pred_prob)[::-1][:5]
    top_5_max_val = list(pred_prob[top_5_max_idx])
    
    top_5_class_name=[]
    for i in top_5_max_idx:
        top_5_class_name.append(hindi_character[i])
    

    # return hindi_character[pred_prob.argmax()] , tf.reduce_max(pred_prob)
    return top_5_class_name,top_5_max_val

if file is None:
    st.text("Please Upload an Image")
else:
    
    image = Image.open(file)
    st.image(image,use_column_width=True)
    
    st.header("Top 5 Prediction for given image")
    class_name , confidense = upload_and_predict(image,model)


    fig = go.Figure()

    fig.add_trace(go.Bar(
            x=confidense[::-1],
            y=class_name[::-1],
            orientation='h'))
    fig.update_layout(height = 500 , width = 900, 
                  xaxis_title='Probability' , yaxis_title='Top 5 Class Name')
    
    st.plotly_chart(fig,use_container_width=True)


    st.success(f"The image is classified as \t  \'{class_name[0]}\' \t with {confidense[0]*100:.1f} %")