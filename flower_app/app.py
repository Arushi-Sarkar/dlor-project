import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import pandas as pd
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FloraVision AI",layout="wide",initial_sidebar_state="expanded")

#global CSS styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --sage:   darkseagreen;
    --forest: darkgreen;
    --cream:  linen;
    --sand:   wheat;
    --petal:  rosybrown;
    --night:  darkslategray;
}

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: darkslategray !important;
}

/*background shapes*/
.stApp::before {
    content: '';
    position: fixed;
    top: -30%;
    right: -20%;
    width: 70vw;
    height: 70vw;
    background: radial-gradient(ellipse, rgba(143,188,143,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -20%;
    left: -15%;
    width: 60vw;
    height: 60vw;
    background: radial-gradient(ellipse, rgba(188,143,143,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    position: relative;
    z-index: 1;
}

section[data-testid="stAppViewContainer"] {
    background: darkslategray !important;
}

/*typography*/
h1 { font-family: 'Playfair Display', serif !important; color: linen !important; }
h2 { font-family: 'Playfair Display', serif !important; color: linen !important; }
h3 { font-family: 'DM Sans', sans-serif !important; color: linen !important; }
p, label, .stMarkdown { color: linen !important; }

/*sidebar*/
[data-testid="stSidebar"] {
    background: black !important;
    border-right: 1px solid rgba(143,188,143,0.2);
}
[data-testid="stSidebar"] * { color: linen !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: linen !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: silver !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/*file uploader*/
[data-testid="stFileUploader"] {
    background: rgba(143,188,143,0.08);
    border: 1.5px dashed darkseagreen;
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.35s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: darkseagreen;
    background: rgba(143,188,143,0.13);
}
[data-testid="stFileUploader"] * { color: linen !important; }

/*buttons*/
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, darkseagreen, darkgreen);
    color: linen !important;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,100,0,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0,100,0,0.5);
}

/*progress bar*/
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, darkseagreen, rosybrown);
    border-radius: 99px;
}
.stProgress > div > div {
    background: rgba(143,188,143,0.15);
    border-radius: 99px;
}

/*expander*/
[data-testid="stExpander"] {
    background: rgba(255,239,213,0.04);
    border: 1px solid rgba(143,188,143,0.2);
    border-radius: 14px;
}
[data-testid="stExpander"] summary { color: linen !important; }

/*alerts*/
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 12px !important;
}

/*bar chart*/
[data-testid="stVegaLiteChart"] {
    background: rgba(255,239,213,0.04);
    border-radius: 12px;
    padding: 0.5rem;
}

/*select and slider labels*/
.stSelectbox > label, .stSlider > label {
    color: silver !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
</style>
""", unsafe_allow_html=True)

#loading the model
@st.cache_resource
def load_all_models():
    save_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        models = {
            'custom_cnn':load_model(os.path.join(save_dir,'custom_cnn_final.keras')),
            'resnet50':load_model(os.path.join(save_dir,'resnet50_final.keras')),
            'efficientnet':load_model(os.path.join(save_dir,'efficientnet_final.keras'))
        }
        with open(os.path.join(save_dir, 'model_metadata.json'),'r') as f:
            metadata=json.load(f)
        return models,metadata
    except FileNotFoundError:
        st.error(f"Could not find model files in: {save_dir}")
        st.error("need to create a 'models' folder and keep the downloaded the files there")
        st.stop()

with st.spinner('Waking up the neural networks…'):
    models_dict, metadata = load_all_models()
    CLASS_NAMES=metadata['class_names']
    WEIGHTS= metadata.get('ensemble_weights', {
        'custom_cnn': 0.260,'resnet50': 0.370,'efficientnet': 0.370
    })

FLOWER_INFO = {
    'daisy': {
        'emoji': '🌼','family': 'Asteraceae','origin': 'Europe & North America',
        'season': 'Spring – summer',
        'fun_fact': 'Daisies were used in Victorian era floriography to mean innocence.',
        'color': 'gold',},
    'dandelion': {
        'emoji': '❋','family': 'Asteraceae','origin': 'Eurasia','season': 'Year round',
        'fun_fact': 'Every part of a dandelion is edible, even its roots, leaves and flowers.',
        'color': 'goldenrod',},
    'rose': {
        'emoji': '🌹','family': 'Rosaceae','origin': 'Asia','season': 'Late spring – autumn',
        'fun_fact': 'Roses have been cultivated for over 5000 years with 30000+ known varieties.',
        'color': 'crimson',},
    'sunflower': {
        'emoji': '🌻',
        'family': 'Asteraceae','origin': 'North America','season': 'Summer – autumn',
        'fun_fact': 'Young sunflowers track the sun daily (heliotropism) but stop when mature.',
        'color': 'darkorange',},
    'tulip': {
        'emoji': '🌷','family': 'Liliaceae', 'origin': 'Central Asia','season': 'Spring',
        'fun_fact': 'Tulip bulbs were used as a currency substitute during 17th century Netherlands.',
        'color': 'mediumvioletred',},}

def confidence_tier(conf):
    if conf>0.85:
        return "High","darkseagreen", "✦"
    elif conf >0.60:
        return "Moderate", "wheat","◈"
    else:
        return "Low", "rosybrown","◇"

def predict_ensemble(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array= keras_image.img_to_array(img_resized)
    img_batch= np.expand_dims(img_array, axis=0)

    input_custom = img_batch.copy() / 255.0
    input_resnet= resnet_preprocess(img_batch.copy())
    input_effnet=effnet_preprocess(img_batch.copy())

    preds = {}
    preds['custom_cnn']= models_dict['custom_cnn'].predict(input_custom, verbose=0)[0]
    preds['resnet50']= models_dict['resnet50'].predict(input_resnet,verbose=0)[0]
    preds['efficientnet']= models_dict['efficientnet'].predict(input_effnet, verbose=0)[0]

    ensemble_prob = (
        preds['custom_cnn']* WEIGHTS['custom_cnn'] +
        preds['resnet50']* WEIGHTS['resnet50'] +
        preds['efficientnet']* WEIGHTS['efficientnet'])
    return ensemble_prob,preds,img

def apply_image_enhancement(img, mode):
    if mode =="Original":
        return img
    elif mode =="Vivid":
        return ImageEnhance.Color(img).enhance(1.6)
    elif mode=="Warm":
        r,g,b = img.split()
        r = ImageEnhance.Brightness(r).enhance(1.15)
        return Image.merge('RGB',(r,g, b))
    elif mode == "Cool":
        r,g,b = img.split()
        b= ImageEnhance.Brightness(b).enhance(1.2)
        return Image.merge('RGB', (r, g, b))
    elif mode== "Grayscale":
        return img.convert('L').convert('RGB')
    return img

#creating the sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem;'>
        <p style='font-family:Playfair Display,serif; font-size:1.6rem; font-style:italic;
                  color:linen; margin:0; line-height:1.2;'>Flora<strong style='font-style:normal;'>Vision</strong></p>
        <p style='font-size:0.75rem; color:silver; letter-spacing:0.2em;
                  text-transform:uppercase; margin:0.3rem 0 1.5rem;'>AI · Flower intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:silver;margin-bottom:0.5rem;'>Ensemble weights</p>", unsafe_allow_html=True)

    model_labels = {
        'resnet50':('ResNet50','cornflowerblue'),
        'efficientnet':('EfficientNet','darkseagreen'),
        'custom_cnn':('Custom CNN','rosybrown'),
    }
    for key,(label,color) in model_labels.items():
        w = WEIGHTS[key]
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.7rem;margin:0.4rem 0;'>
            <div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;'></div>
            <span style='font-size:0.88rem;color:linen;flex:1;'>{label}</span>
            <span style='font-family:DM Mono,monospace;font-size:0.82rem;color:{color};'>{w:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:darkseagreen;opacity:0.2;margin:1.2rem 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:silver;margin-bottom:0.7rem;'>Supported species</p>", unsafe_allow_html=True)
    for name, info in FLOWER_INFO.items():
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;border-bottom:1px solid rgba(143,188,143,0.08);'>
            <span style='font-size:1.1rem;'>{info['emoji']}</span>
            <span style='font-size:0.9rem;color:linen;'>{name.capitalize()}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:darkseagreen;opacity:0.2;margin:1.2rem 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:silver;margin-bottom:0.7rem;'>Image filter</p>", unsafe_allow_html=True)
    enhancement_mode = st.selectbox(
        "Enhancement",
        ["Original","Vivid","Warm", "Cool","Grayscale"],
        label_visibility="collapsed")

    show_individual = st.checkbox("Show per model breakdown",value=True)
    show_radar = st.checkbox("Show probability chart",value=True)

    st.markdown("<hr style='border-color:darkseagreen;opacity:0.2;margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.75rem;color:darkgray;line-height:1.6;'>
    FloraVision combines three neural architectures called ResNet50, EfficientNetB0 and a custom CNN via weighted ensemble voting for robust classification.
    </p>
    """,unsafe_allow_html=True)

col_h1,col_h2 =st.columns([3,1])
with col_h1:
    st.markdown("""
    <div style='padding-bottom:0.3rem;'>
        <h1 style='font-size:3rem;margin:0;letter-spacing:-0.02em;'>
            Flora<em style='color:darkseagreen;'>Vision</em>
        </h1>
        <p style='color:darkgray;font-size:1rem;margin:0.4rem 0 0;letter-spacing:0.02em;'>
            Ensemble deep learning · Flower species identification
        </p>
    </div>
    """,unsafe_allow_html=True)

st.markdown("<hr style='border-color:darkseagreen;opacity:0.2;margin:0.8rem 0 1.5rem;'>", unsafe_allow_html=True)

#uploading the file
uploaded_file = st.file_uploader(
    "Drop a flower image here or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG and PNG. Best results with clear, well lit flower photos.")

#the results after the file is uploaded
if uploaded_file is not None:
    t0 = time.time()
    with st.spinner('Analysing with 3 neural networks…'):
        ensemble_probs, individual_preds, raw_img = predict_ensemble(uploaded_file)
    elapsed=time.time()-t0

    pred_idx= np.argmax(ensemble_probs)
    pred_class=CLASS_NAMES[pred_idx]
    confidence = ensemble_probs[pred_idx]
    tier_label,tier_color, tier_icon = confidence_tier(confidence)
    flower_info =FLOWER_INFO.get(pred_class.lower(),{})

    enhanced_img=apply_image_enhancement(raw_img, enhancement_mode)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.55], gap="large")

    #left column with image and predictions
    with left:
        st.image(enhanced_img, use_container_width=True,
                 caption=f"Filter: {enhancement_mode}")

        #top 3 predictions
        st.markdown("""
        <p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                  color:darkgray;margin:1.2rem 0 0.6rem;'>Top predictions</p>
        """, unsafe_allow_html=True)

        top_3_idx=np.argsort(ensemble_probs)[-3:][::-1]
        for rank, idx in enumerate(top_3_idx):
            sname= CLASS_NAMES[idx]
            prob= ensemble_probs[idx]
            finfo= FLOWER_INFO.get(sname.lower(), {})
            emoji= finfo.get('emoji','🌸')
            bar_w= int(prob * 100)
            accent= finfo.get('color','darkseagreen')
            bold= "font-weight:700;" if rank== 0 else ""
            border_alpha = "0.35" if rank==0 else "0.10"
            st.markdown(f"""
            <div style='background:rgba(255,239,213,0.04);border-radius:10px;
                        padding:0.75rem 1rem;margin-bottom:0.5rem;
                        border:1px solid rgba(143,188,143,{border_alpha});'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.45rem;'>
                    <span style='font-size:0.95rem;color:linen;{bold}'>{emoji} {sname.capitalize()}</span>
                    <span style='font-family:DM Mono,monospace;font-size:0.92rem;color:{accent};{bold}'>{prob*100:.1f}%</span>
                </div>
                <div style='height:3px;background:rgba(255,239,213,0.07);border-radius:99px;'>
                    <div style='width:{bar_w}%;height:100%;background:{accent};border-radius:99px;'></div>
                </div>
            </div>
            """,unsafe_allow_html=True)

        st.markdown(f"""
        <div style='display:flex;gap:0.5rem;margin-top:0.8rem;flex-wrap:wrap;'>
            <span style='font-size:0.75rem;padding:0.25rem 0.6rem;background:rgba(143,188,143,0.12);
                         border-radius:99px;color:darkgray;'>⏱ {elapsed:.2f}s</span>
            <span style='font-size:0.75rem;padding:0.25rem 0.6rem;background:rgba(143,188,143,0.12);
                         border-radius:99px;color:darkgray;'>3 models</span>
            <span style='font-size:0.75rem;padding:0.25rem 0.6rem;background:rgba(143,188,143,0.12);
                         border-radius:99px;color:darkgray;'>224 × 224 px</span>
        </div>
        """, unsafe_allow_html=True)

    #right column with main prediction and details
    with right:
        accent_c=flower_info.get('color','darkseagreen')
        st.markdown(f"""
        <div style='background:rgba(255,239,213,0.04);border:1px solid rgba(143,188,143,0.2);
                    border-radius:16px;padding:2rem;margin-bottom:1rem;position:relative;overflow:hidden;'>
            <p style='font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;
                      color:darkgray;margin:0 0 0.4rem;'>Identified as</p>
            <h1 style='font-size:3.2rem;margin:0;letter-spacing:-0.02em;
                       color:{accent_c};font-family:Playfair Display,serif;'>{flower_info.get('emoji','🌸')} {pred_class.capitalize()}</h1>
            <div style='margin-top:1rem;display:flex;align-items:center;gap:0.6rem;'>
                <span style='font-family:DM Mono,monospace;font-size:1.6rem;color:linen;font-weight:700;'>{confidence*100:.1f}%</span>
                <span style='font-size:0.85rem;padding:0.2rem 0.7rem;
                             background:transparent;border:1px solid {tier_color};
                             border-radius:99px;color:{tier_color};'>
                    {tier_icon} {tier_label} confidence
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence))

        #botanical info card
        if flower_info:
            st.markdown("""
            <p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                      color:darkgray;margin:1.2rem 0 0.6rem;'>Botanical profile</p>
            """, unsafe_allow_html=True)

            info_items = [
                ('Family',flower_info.get('family','—')),
                ('Origin',flower_info.get('origin','—')),
                ('Season',flower_info.get('season','—')),]
            cols=st.columns(3)
            for col,(label,val) in zip(cols, info_items):
                with col:
                    st.markdown(f"""
                    <div style='background:rgba(255,239,213,0.04);border-radius:10px;
                                padding:0.9rem;border:1px solid rgba(143,188,143,0.1);text-align:center;'>
                        <p style='font-size:0.7rem;color:darkgray;
                                  text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.3rem;'>{label}</p>
                        <p style='font-size:0.88rem;color:linen;margin:0;font-weight:500;'>{val}</p>
                    </div>
                    """,unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:rgba(255,239,213,0.03);border-left:3px solid {accent_c};
                        border-radius:0 10px 10px 0;padding:0.9rem 1rem;margin-top:0.8rem;'>
                <p style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                          color:darkgray;margin:0 0 0.3rem;'>Did you know?</p>
                <p style='font-size:0.9rem;color:silver;margin:0;
                          font-style:italic;line-height:1.5;'>{flower_info.get('fun_fact','')}</p>
            </div>
            """,unsafe_allow_html=True)

        #probability chart
        if show_radar:
            st.markdown("""
            <p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                      color:darkgray;margin:1.3rem 0 0.4rem;'>Probability distribution</p>
            """, unsafe_allow_html=True)
            chart_data =pd.DataFrame({
                'Species':[c.capitalize() for c in CLASS_NAMES],
                'Probability':ensemble_probs*100
            })
            st.bar_chart(chart_data.set_index('Species'),height=220,color="#8fbc8f")

    #model breakdown
    if show_individual:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        with st.expander("Neural network ensemble breakdown", expanded=False):
            st.markdown("""
            <p style='color:darkgray;font-size:0.9rem;margin-bottom:1.2rem;'>
            Each model votes independently and their outputs are merged using weighted averaging.
            </p>
            """, unsafe_allow_html=True)

            vcols = st.columns(3)
            model_styles = {
                'custom_cnn':('Custom CNN','linear-gradient(135deg, mediumvioletred, darkviolet)'),
                'resnet50':('ResNet-50','linear-gradient(135deg, cornflowerblue, deepskyblue)'),
                'efficientnet':('EfficientNet','linear-gradient(135deg, mediumseagreen, teal)'),}

            for col,(mkey,(mlabel, mgrad)) in zip(vcols, model_styles.items()):
                preds = individual_preds[mkey]
                winner_name=CLASS_NAMES[np.argmax(preds)]
                winner_conf =np.max(preds)
                winner_info =FLOWER_INFO.get(winner_name.lower(),{})

                with col:
                    st.markdown(f"""
                    <div style='background:{mgrad};border-radius:14px;padding:1.4rem;text-align:center;'>
                        <p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                                  color:silver;margin:0 0 0.3rem;'>{mlabel}</p>
                        <p style='font-size:1.7rem;margin:0;'>{winner_info.get('emoji','🌸')}</p>
                        <p style='font-size:1.15rem;font-weight:700;color:white;margin:0.3rem 0 0.1rem;'>
                            {winner_name.capitalize()}
                        </p>
                        <p style='font-family:DM Mono,monospace;font-size:1.1rem;color:ghostwhite;margin:0;'>
                            {winner_conf*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:0.8rem;'>",unsafe_allow_html=True)
                    for ci,cname in enumerate(CLASS_NAMES):
                        p=preds[ci]
                        bw=int(p*100)
                        fi =FLOWER_INFO.get(cname.lower(), {})
                        col_bar = fi.get('color','darkseagreen')
                        st.markdown(f"""
                        <div style='display:flex;align-items:center;gap:0.4rem;margin-bottom:0.3rem;'>
                            <span style='font-size:0.75rem;color:darkgray;width:70px;'>{cname.capitalize()}</span>
                            <div style='flex:1;height:4px;background:rgba(255,239,213,0.1);border-radius:99px;'>
                                <div style='width:{bw}%;height:100%;background:{col_bar};border-radius:99px;'></div>
                            </div>
                            <span style='font-family:DM Mono,monospace;font-size:0.7rem;
                                         color:darkgray;width:36px;text-align:right;'>{p*100:.0f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    #exporting the results
    with st.expander("Export results",expanded=False):
        result_df = pd.DataFrame({
            'Species':[c.capitalize() for c in CLASS_NAMES],
            'Ensemble (%)':(ensemble_probs*100).round(2),
            'Custom CNN (%)':(individual_preds['custom_cnn']*100).round(2),
            'ResNet50 (%)':(individual_preds['resnet50']*100).round(2),
            'EfficientNet (%)':(individual_preds['efficientnet']*100).round(2),
        })
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",data=csv,
            file_name=f"floravision_{pred_class}_{int(confidence*100)}pct.csv",
            mime="text/csv")

else:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;padding:3rem 2rem;background:rgba(255,239,213,0.03);
                border:1px dashed rgba(143,188,143,0.25);border-radius:20px;margin:1.5rem 0;'>
        <p style='font-size:2.5rem;margin:0 0 0.8rem;'>🌿</p>
        <h2 style='color:linen;font-family:Playfair Display,serif;
                   font-size:1.6rem;margin:0 0 0.6rem;'>Drop a flower image to begin</h2>
        <p style='color:darkgray;font-size:0.95rem;max-width:420px;
                  margin:0 auto;line-height:1.6;'>
            FloraVision will identify the species using a 3 model neural ensemble trained on thousands of flower photographs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
              color:darkgray;margin:2rem 0 0.8rem;'>Supported species</p>
    """,unsafe_allow_html=True)

    scols = st.columns(5)
    for col, (name,info) in zip(scols, FLOWER_INFO.items()):
        with col:
            st.markdown(f"""
            <div style='text-align:center;padding:1.2rem 0.5rem;
                        background:rgba(255,239,213,0.03);border-radius:12px;
                        border:1px solid rgba(143,188,143,0.12);'>
                <p style='font-size:2rem;margin:0 0 0.4rem;'>{info['emoji']}</p>
                <p style='font-size:0.88rem;color:linen;margin:0;font-weight:500;'>
                    {name.capitalize()}
                </p>
                <p style='font-size:0.72rem;color:darkgray;margin:0.2rem 0 0;'>
                    {info['season']}
                </p>
            </div>
            """,unsafe_allow_html=True)