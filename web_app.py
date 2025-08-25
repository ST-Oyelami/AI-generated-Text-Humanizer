from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import torch
import tensorflow
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from gramformer import Gramformer


#Loading all models to cache
@st.cache_resource
def load_all_models():
    #Initializing the model and its tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("alykassem/FLAN-T5-Paraphraser")
    model = model = TFAutoModelForSeq2SeqLM.from_pretrained("alykassem/FLAN-T5-Paraphraser", from_pt=True)
    # Handling gramformer specially
    gf = Gramformer(models=1)
    return device, tokenizer, model, gf

device, tokenizer, model, gf = load_all_models()


# Capitalize the first letter
def fix(sentence):
    # Remove leading/trailing whitespace
    sentence = sentence.strip()
    if not sentence:
        return ""
    
    sentence = sentence[0].upper() + sentence[1:]
    return sentence



#Calling the model and other functions for humanizing task
def paraphrase(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    paraphrased_sentences = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="tf")
        outputs = model.generate(**inputs)
        paraphrased_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_text = gf.correct(paraphrased_result)
        paraphrased_sentences.append(fix(list(corrected_text)[0])) 

    # Join the paraphrased sentences back together
    paraphrased_paragraph = " ".join(paraphrased_sentences)
    return paraphrased_paragraph



# Streamlit app layout
# Style and Layout: Set the background and header colors for a blue hue aesthetic
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;  /* Alice Blue */
        }
        h1, h2, h3 {
            color: #007acc;  /* Soft Blue for titles */
        }
        .reportview-container {
            background: #f0f8ff;
        }
        textarea {
            background-color: #e0f7fa;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
        }
        .stTextArea>div>div>textarea {
            color: #007acc;
        }
    </style>
    """, unsafe_allow_html=True)
# Add the banner image at the top
#st.image("Your_banner.png Here, if any", use_column_width=True)


st.title("Text Humanizing App")
st.markdown("Enter your text on the left, and see the paraphrased version on the right!")

# Split the page into two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Original Text")
    input_text = st.text_area("Enter your text here:", height=500)

with col2:
    st.header("Paraphrased Text")
    if st.button("Humanize"):
        if input_text:
            
            # Paraphrase the input text
            paraphrased_text = paraphrase(input_text)

            st.write(paraphrased_text)
        else:
            st.write("Please enter text in the left column.")
