import numpy as np
import plotly.graph_objects as go
from taipy.gui import Gui, Markdown, notify
import taipy.gui.builder as tgb
import pandas as pd
from datetime import datetime
import random
import xgboost as xgb

layout = {
    "dragmode": "select",
    "margin": {"l":0,"r":0,"b":0,"t":0}
}

config = {
    "displayModeBar": True,
    "displaylogo":False
}

eda_chart_properties = {
    "layout": {
    "title":"Electrodermal Activity",
    "showlegend":False
    }
}

hr_chart_properties = {
    "layout": {
    "title":"Heart Rate",
    "showlegend":False
    }
}

pred_chart_properties = {
    "layout": {
    "title":"Emotional State",
    "showlegend":True
    }
}

hr_visible = False
eda_visible = False
eda_data = None
hr_data = None
eda_data_chart = None
hr_data_chart = None
file_content = None
pred_visible = False
pred_data_chart = None

model_stress = xgb.Booster()
model_stress.load_model("Wearables/model_stress.xgb")
model_attention = xgb.Booster()
model_attention.load_model("Wearables/model_attention.xgb")
model_valence = xgb.Booster()
model_valence.load_model("Wearables/model_valence.xgb")
model_arousal = xgb.Booster()
model_arousal.load_model("Wearables/model_arousal.xgb")

def on_upload_file(state):
    files = state.file_content.split(";")
    for file in files:
        if "eda" in file:
            try:
                state.eda_data = pd.read_csv(file)
                state.eda_data["datetime"] = pd.to_datetime(state.eda_data["time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                state.eda_data_chart = {"Period": state.eda_data["datetime"].tolist(), "EDA": state.eda_data["eda"].tolist()}
                state.eda_visible = True
            except:
                state.eda_data_chart = None
                state.eda_visible = False
                state.eda_data = None
                notify(state, "error", "Error processing EDA data...")
        if "hr" in file:
            try:
                state.hr_data = pd.read_csv(file, header=0)
                state.hr_data.columns = ["hr"]
                state.hr_data["period"] = range(0, state.hr_data.shape[0])
                state.hr_data_chart = {"Period": state.hr_data["period"].tolist(), "HR": state.hr_data["hr"].tolist()}
                state.hr_visible = True
            except:
                state.hr_data_chart = None
                state.hr_visible = False
                state.hr_data = None
                notify(state, "error", "Error processing HR data...")
    if state.hr_data is not None and state.eda_data is not None:
        hr = state.hr_data["hr"].tolist()
        eda = state.eda_data['eda'].tolist()
        eda = [np.mean(eda[i:i+4]) for i in range(0, len(eda), 4)]
        if len(eda) < len(hr):
            hr = hr[0:len(eda)]
        else:
            eda = eda[0:len(hr)]
        temp = pd.DataFrame({'hr': hr,'eda': eda})
        stress = model_stress.predict(xgb.DMatrix(temp[["hr","eda"]]))
        attention = model_attention.predict(xgb.DMatrix(temp[["hr","eda"]]))
        arousal = model_arousal.predict(xgb.DMatrix(temp[["hr","eda"]]))
        valence = model_valence.predict(xgb.DMatrix(temp[["hr","eda"]]))
        temp = pd.DataFrame({'period': range(0, len(stress)),\
                             'stress': stress, \
                             "attention": attention, \
                             "arousal": arousal, \
                             "valance": valence})
        state.pred_data_chart = {"Period": temp["period"].tolist(), \
                                 "Stress": temp["stress"].tolist(), \
                                 "Attention": temp["attention"].tolist(), \
                                 "Valence": temp["valance"].tolist(),
                                 "Arousal": temp["arousal"].tolist()}
        state.pred_visible = True



with tgb.Page() as page:
    with tgb.layout("1 1 1"):
        with tgb.part():
            tgb.file_selector("{file_content}", label="Upload File", on_action=on_upload_file, extensions=".csv", drop_message="Drop Message", multiple=True)

    with tgb.layout("1 1fs"):
        with tgb.part(render="{hr_visible}"):
            tgb.chart("{hr_data_chart}", height="300px", rebuild=True, layout="{layout}", plot_config="{config}", properties="{hr_chart_properties}")
        with tgb.part(render="{eda_visible}"):
            tgb.chart("{eda_data_chart}", height="300px", rebuild=True, layout="{layout}", plot_config="{config}", properties="{eda_chart_properties}")
        with tgb.part(render="{pred_visible}"):
            tgb.chart("{pred_data_chart}", x="Period", y__1="Stress", y__2="Arousal", y__3="Valence", y__4="Attention", height="300px", rebuild=True, layout="{layout}", plot_config="{config}", properties="{pred_chart_properties}")

if __name__ == "__main__":
    Gui(page=page).run(title="Wearables")

# tabs using buttons and render
# first tab: eda, hr, etc. as line plots
# overlay our predictions on top of them
# second tab: correlation, histograms side by side
# third tab: eeg - correlation, other plots