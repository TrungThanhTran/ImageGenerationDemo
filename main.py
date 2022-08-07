import time
import torch
import os, psutil
from PIL import Image
import streamlit as st
from time import gmtime, strftime
from diffusers import DiffusionPipeline

EXAMPLE = 'A painting of a monkey eating a banana'
MODEL_ID = "CompVis/ldm-text2im-large-256"
    
st.set_page_config(layout="wide")
st.title('Welcome To AI ART Creation!')
st.markdown(
	"""
	<style>
	[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
		width: 350px;
	}
	[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
		width: 350px;
		margin-left: -350px;
	}
	</style>
	""",
	unsafe_allow_html=True,)
    
instructions = """
        This app is using lib diffusers published by Hugging Face to generate
        photos/images when inputing a prompt. 
        """
st.write(instructions)
col_sidebar, _ = st.columns((2, 5))

with col_sidebar: # Sidebar
    st.sidebar.title('Configuration for the diffusion model')
    st.sidebar.subheader('Parameters')

    st.sidebar.markdown('---')
    st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 400px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 400px;
                margin-left: -400px;
            }
            </style>
            """,
            unsafe_allow_html=True)

    inference_steps = st.sidebar.slider('Inference Steps', min_value =1, max_value = 100, value = 50)
    eta = st.sidebar.slider('Eta', min_value = 0.0,max_value = 2.0,value = 0.3)
    guidance_scale = st.sidebar.slider('Guidance Scale - how much the prompt will influence the results',
                                            min_value =1, max_value = 20, value = 6)

    st.sidebar.markdown('---')

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None}, allow_output_mutation=True)
def load_model():
    start = time.time()
    ldm = DiffusionPipeline.from_pretrained(MODEL_ID)
    print(f'loading model time = {int(time.time() - start)} seconds')
    return ldm

def byte2megabyte(num_byte):
    MB = 9.537e-7
    GB = 1e+9
    return num_byte * MB, int(num_byte / GB)

if __name__=="__main__":
    ldm = load_model()
    print('start app')

    user_prompt = st.text_input("Prompt", '') # require key for an identical input
    if st.button('Generate photo'):
        if user_prompt != EXAMPLE or user_prompt != '' or user_prompt is not None:
            # load text input to diffusion pipeline
            tme = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            start = time.time()
            # run inference (sample random noise and denoise)
            with st.spinner('Generating...'):
                images = ldm([user_prompt], num_inference_steps=inference_steps, eta=eta, guidance_scale=guidance_scale)["sample"]
                if images is not None:
                    # save images
                    for idx, image in enumerate(images):
                        process = psutil.Process(os.getpid())
                        u_mb, u_gb = byte2megabyte(process.memory_info().rss) # in bytes 
                        cap = f'Generat an image in {int(time.time() - start)} seconds, memory using is {u_gb} GB({u_mb} MB)'
                        st.image(image, caption=cap)
                        image.save(f"data/{user_prompt}-{tme}.png")
                        
                process = psutil.Process(os.getpid())
        else:
            st.warning(f'Please fill in a prompt. For example: {EXAMPLE}')
