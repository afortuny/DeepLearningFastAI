from IPython.display import Image
Image(filename="/content/Brands.JPG")
import ipywidgets as widgets
from fastai.vision.all import *
from fastai.vision.widgets import * 
learn_inf = load_learner('export.pkl')
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
def on_data_change(change):
    lbl_pred.value = ''
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
btn_upload.observe(on_data_change, names=['data'])
display(VBox([widgets.Label('Upload your running shoe!'), btn_upload, out_pl, lbl_pred]))

