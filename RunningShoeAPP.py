from fastai.vision.widgets import *
import streamlit as st
import PIL.Image
from fastai.vision.all import Path,load_learner,Image
from pathlib import Path
import fastbook
from fastbook import *
from fastai.vision.widgets import *

learn_inf = load_learner(Path()/'export.pkl', cpu=True)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
def on_click(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:0.04f}'
btn_upload.observe(on_click, names=['data'])
display(VBox([widgets.Label('Select your bear!'), btn_upload, out_pl, lbl_pred]))
