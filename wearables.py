import math
import numpy as np
import pandas as pd
import xgboost as xgb
from taipy.gui import Gui, notify
import taipy.gui.builder as tgb
from datetime import datetime, timedelta

# region Chart Configuration

chart_layout = {
    "dragmode": "select",
    "margin": {"l":0,"r":0,"b":0,"t":0}
}

chart_config = {
    "displayModeBar": True,
    "displaylogo":False
}

chart_properties = {
    "layout": {
    "showlegend":False
    }
}

pred_chart_properties = {
    "layout": {
    "showlegend":True
    }
}

correlation_chart_options = {
    "colorscale": "Bluered",
    "mode":"text"
}

correlation_chart_layout = {
    "annotations": [],
    "xaxis": {
        "visible": True,
        "title":None
    },
    "yaxis": {
        "visible": True,
        "title":None
    }
}

hist_chart_layout = {
    "xaxis": {
        "visible": True,
        "title":None
    },
    "yaxis": {
        "visible": True,
        "title":None
    }
}

#endregion

# region ML Models

model_stress = xgb.Booster()
model_stress.load_model("Wearables/model_stress.xgb")
model_attention = xgb.Booster()
model_attention.load_model("Wearables/model_attention.xgb")
model_valence = xgb.Booster()
model_valence.load_model("Wearables/model_valence.xgb")
model_arousal = xgb.Booster()
model_arousal.load_model("Wearables/model_arousal.xgb")

#endregion

# region Data and Charts

hr_visible = False
eda_visible = False
bvp_visible = False
ibi_visible = False
temp_visible = False
acc_visible = False
eda_data = None
hr_data = None
bvp_data = None
ibi_data = None
temp_data = None
acc_data = None
eda_data_chart = pd.DataFrame(columns=["Period", "EDA"])
hr_data_chart = pd.DataFrame(columns=["Period", "HR"])
bvp_data_chart = pd.DataFrame(columns=["Period", "BVP"])
ibi_data_chart = pd.DataFrame(columns=["Period", "IBI"])
temp_data_chart = pd.DataFrame(columns=["Period", "TEMP"])
acc_data_chart = pd.DataFrame(columns=["Period", "Magnitude"])

stat_visible = False
correlation_data_chart = pd.DataFrame(columns=["x","y","z"])
hr_hist_data_chart = []
eda_hist_data_chart = []
temp_hist_data_chart = []

pred_visible = False
pred_data_chart = pd.DataFrame(columns=["Period", "Stress", "Attention", "Valence", "Arousal"])

# endregion

# region Empatica Upload/Download

file_content = None
show_embrace_dialog = False
aws_path = ""
aws_key = ""
aws_secret = ""

def on_upload_embrace(state):
    state.show_embrace_dialog = True

def embrace_action(state, id, payload):
    if payload["args"][0] == 0:
        print("val")
    state.show_embrace_dialog = False

def on_upload_e4(state):
    files = None
    if type(state.file_content) is str:
        files = state.file_content.split(";")
    else:
        files = state.file_content
    for file in files:
        if "ACC" in file.upper():
            state.acc_data = pd.read_csv(file, header=None)
            start_time = state.acc_data.values[0]
            sampling_rate = state.acc_data.values[1]
            x_data = state.acc_data.iloc[2:][0].tolist()
            y_data = state.acc_data.iloc[2:][1].tolist()
            z_data = state.acc_data.iloc[2:][2].tolist()
            g_magnitude = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_data, y_data, z_data)]
            g_magnitude = [np.mean(g_magnitude[i:i+2]) for i in range(0, len(g_magnitude), int(sampling_rate[0]))]
            timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(g_magnitude))]
            state.acc_data = pd.DataFrame(columns=["datetime", "magnitude"])
            state.acc_data["datetime"] = timestamps
            state.acc_data["magnitude"] = g_magnitude
            state.acc_data_chart = {"Period": state.acc_data["datetime"].tolist(), "Magnitude": state.acc_data["magnitude"].tolist()}
            state.acc_visible = True
        if "IBI" in file.upper():
            try:
                state.ibi_data = pd.read_csv(file, header=None)
                start_time = state.ibi_data.values[0][0]
                intervals = state.ibi_data.values[1:][:,0].tolist()
                ibi = state.ibi_data.values[1:][:,1].tolist()
                ibi = [float(i) for i in ibi]
                timestamps = [datetime.fromtimestamp(start_time) + timedelta(seconds=i) for i in intervals]
                state.ibi_data = pd.DataFrame(columns=["datetime", "ibi"])
                state.ibi_data["datetime"] = timestamps
                state.ibi_data["ibi"] = ibi
                state.ibi_data_chart = {"Period": state.ibi_data["datetime"].tolist(), "IBI": state.ibi_data["ibi"].tolist()}
                state.ibi_visible = True
            except:
                state.ibi_data_chart = None
                state.ibi_visible = False
                state.ibi_data = None
                notify(state, "error", "Error processing IBI data...")
        if "TEMP" in file.upper():
            try:
                state.temp_data = pd.read_csv(file, header=None)
                start_time = state.temp_data.values[0]
                sampling_rate = state.temp_data.values[1]
                temp = state.temp_data.iloc[2:][0].tolist()
                # temp sensor takes time to settle, so average first 10 seconds
                avg_temp = np.mean(temp[10:])
                temp[0:9] = [avg_temp] * 10
                temp = [np.mean(temp[i:i+2]) for i in range(0, len(temp), int(sampling_rate[0]))]
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(temp))]
                state.temp_data = pd.DataFrame(columns=["datetime", "temp"])
                state.temp_data["datetime"] = timestamps
                state.temp_data["temp"] = temp
                state.temp_hist_data_chart = state.temp_data["temp"].tolist()
                state.temp_data_chart = {"Period": state.temp_data["datetime"].tolist(), "TEMP": state.temp_data["temp"].tolist()}
                state.temp_visible = True
            except:
                state.temp_data_chart = None
                state.temp_visible = False
                state.temp_data = None
                notify(state, "error", "Error processing TEMP data...")
        if "BVP" in file.upper():
            try:
                state.bvp_data = pd.read_csv(file, header=None)
                start_time = state.bvp_data.values[0]
                sampling_rate = state.bvp_data.values[1]
                bvp = state.bvp_data.iloc[2:][0].tolist()
                bvp = [np.mean(bvp[i:i+2]) for i in range(0, len(bvp), int(sampling_rate[0]))]
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(bvp))]
                state.bvp_data = pd.DataFrame(columns=["datetime", "bvp"])
                state.bvp_data["datetime"] = timestamps
                state.bvp_data["bvp"] = bvp
                state.bvp_data_chart = {"Period": state.bvp_data["datetime"].tolist(), "BVP": state.bvp_data["bvp"].tolist()}
                state.bvp_visible = True
            except:
                state.bvp_data_chart = None
                state.bvp_visible = False
                state.bvp_data = None
                notify(state, "error", "Error processing BVP data...")
        if "EDA" in file.upper():
            try:
                state.eda_data = pd.read_csv(file, header=None)
                start_time = state.eda_data.values[0]
                sampling_rate = state.eda_data.values[1]
                eda = state.eda_data.iloc[2:][0].tolist()
                eda = [np.mean(eda[i:i+4]) for i in range(0, len(eda), int(sampling_rate[0]))]
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(eda))]
                state.eda_data = pd.DataFrame(columns=["datetime", "eda"])
                state.eda_data["datetime"] = timestamps
                state.eda_data["eda"] = eda
                state.eda_hist_data_chart = state.eda_data["eda"].tolist()
                state.eda_data_chart = {"Period": state.eda_data["datetime"].tolist(), "EDA": state.eda_data["eda"].tolist()}
                state.eda_visible = True
            except:
                state.eda_data_chart = None
                state.eda_visible = False
                state.eda_data = None
                notify(state, "error", "Error processing EDA data...")
        if "HR" in file.upper():
            try:
                state.hr_data = pd.read_csv(file, header=None)
                start_time = state.hr_data.values[0]
                hr = state.hr_data.iloc[2:][0].tolist()
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(hr))]
                state.hr_data = pd.DataFrame(columns=["datetime", "hr"])
                state.hr_data["datetime"] = timestamps
                state.hr_data["hr"] = hr
                state.hr_hist_data_chart = state.hr_data["hr"].tolist()
                state.hr_data_chart = {"Period": state.hr_data["datetime"].tolist(), "HR": state.hr_data["hr"].tolist()}
                state.hr_visible = True
            except:
                state.hr_data_chart = None
                state.hr_visible = False
                state.hr_data = None
                notify(state, "error", "Error processing HR data...")
    # correlation
    tables = []
    if state.hr_data.shape[0] > 0:
        tables.append(state.hr_data)
    if state.eda_data.shape[0] > 0:
        tables.append(state.eda_data)
    if state.bvp_data.shape[0] > 0:
        tables.append(state.bvp_data)
    if state.ibi_data.shape[0] > 0:
        tables.append(state.ibi_data)
    if state.temp_data.shape[0] > 0:
        tables.append(state.temp_data)
    if state.acc_data.shape[0] > 0:
        tables.append(state.acc_data)
    if len(tables) > 1:
        state.stat_visible = True
        second_columns = [df.iloc[:, 1] for df in tables]
        combined_df = pd.concat(second_columns, axis=1)
        if "hr" in combined_df.columns:
            combined_df.rename(columns={"hr": "HR"}, inplace=True)
        if "eda" in combined_df.columns:
            combined_df.rename(columns={"eda": "EDA"}, inplace=True)
        if "bvp" in combined_df.columns:
            combined_df.rename(columns={"bvp": "BVP"}, inplace=True)
        if "ibi" in combined_df.columns:
            combined_df.rename(columns={"ibi": "IBI"}, inplace=True)
        if "temp" in combined_df.columns:
            combined_df.rename(columns={"temp": "TEMP"}, inplace=True)
        if "magnitude" in combined_df.columns:
            combined_df.rename(columns={"magnitude": "ACC"}, inplace=True)
        df_corr = combined_df.corr()
        x = df_corr.columns.tolist()
        y = df_corr.index.tolist()
        z = np.array(df_corr).tolist()

        for xx in df_corr.columns:
            for yy in df_corr.index:
                corr_value = df_corr[xx][yy]
                annotation = {
                    "x": xx,
                    "y": yy,
                    "text": round(corr_value, 2),
                    "showarrow": False
                }
                state.correlation_chart_layout["annotations"].append(annotation)
        state.correlation_data_chart = {"x": x, "y": y, "z": z}
    # predictions
    if state.hr_data is not None and state.eda_data is not None: # used for prediction, so we need both
        hr = state.hr_data["hr"].tolist()
        eda = state.eda_data['eda'].tolist()
        if len(eda) < len(hr):
            hr = hr[0:len(eda)]
        else:
            eda = eda[0:len(hr)]
        periods = state.hr_data["datetime"].tolist()[0:len(hr)]
        temp = pd.DataFrame({'hr': hr,'eda': eda})
        stress = model_stress.predict(xgb.DMatrix(temp[["hr","eda"]]))
        attention = model_attention.predict(xgb.DMatrix(temp[["hr","eda"]]))
        arousal = model_arousal.predict(xgb.DMatrix(temp[["hr","eda"]]))
        valence = model_valence.predict(xgb.DMatrix(temp[["hr","eda"]]))
        temp = pd.DataFrame({'period': periods,\
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

# endregion

# region Popup Dialogs

with tgb.Page() as embrace_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.input(value="{aws_path}", label="S3 Path:")
            tgb.input(value="{aws_key}", label="S3 Key:")
            tgb.input(value="{aws_secret}", label="S3 Secret:", password=True)

# endregion

with tgb.Page() as page_main:
    with tgb.layout("0.5 1 0.5"):
        tgb.text("")
        tgb.text("### Wearable Device Analysis", mode="md", class_name="center_text")
        tgb.text("")
    with tgb.layout("1 1"):
        with tgb.part(class_name="buttons"):
            tgb.file_selector("{file_content}", label="Import E4 Data", on_action=on_upload_e4, extensions=".csv", drop_message="Drop To Process", multiple=True)
            tgb.button("Import Embrace Data", on_action=on_upload_embrace)
            tgb.dialog("{show_embrace_dialog}", title="Download Embrace Data", page="embrace_dialog", on_action=embrace_action, labels="Download;Cancel")
    with tgb.layout("1 1fs"):
        with tgb.part(render="{hr_visible}"):
            with tgb.expandable(title="Heart Rate", expanded=True):
                tgb.chart("{hr_data_chart}", height="300px", x="Period", y="HR", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{eda_visible}"):
            with tgb.expandable(title="Electrodermal Activity", expanded=True):
                tgb.chart("{eda_data_chart}", height="300px", x="Period", y="EDA", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{bvp_visible}"):
            with tgb.expandable(title="Blood Volume Pulse", expanded=True):
                tgb.chart("{bvp_data_chart}", height="300px", x="Period", y="BVP", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{temp_visible}"):
            with tgb.expandable(title="Temperature", expanded=True):
                tgb.chart("{temp_data_chart}", height="300px", x="Period", y="TEMP", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{ibi_visible}"):
            with tgb.expandable(title="Interbeat Interval", expanded=True):
                tgb.chart("{ibi_data_chart}", height="300px", x="Period", y="IBI", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{acc_visible}"):
            with tgb.expandable(title="Movement", expanded=True):
                tgb.chart("{acc_data_chart}", height="300px", x="Period", y="Magnitude", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part():
            with tgb.part(render="{stat_visible}"):
                with tgb.expandable(title="Statistics", expanded=False):
                    with tgb.layout("1 1 1 1"):
                        with tgb.part():
                            tgb.chart("{correlation_data_chart}", type="heatmap", title="Correlation", x="x", y="y", z="z", height="300px", rebuild=True, options="{correlation_chart_options}", layout="{correlation_chart_layout}")
                        with tgb.part(render="{hr_visible}"):
                            tgb.chart("{hr_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="HR", layout="{hist_chart_layout}")
                        with tgb.part(render="{eda_visible}"):
                            tgb.chart("{eda_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="EDA", layout="{hist_chart_layout}")
                        with tgb.part(render="{temp_visible}"):
                            tgb.chart("{temp_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="TEMP", layout="{hist_chart_layout}")
        with tgb.part(render="{pred_visible}"):
            with tgb.expandable(title="Emotional State", expanded=False):
                with tgb.part():
                    tgb.chart("{pred_data_chart}", x="Period", y__1="Stress", y__2="Arousal", y__3="Valence", y__4="Attention", height="300px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")

            
if __name__ == "__main__":
    pages = {"page_main": page_main, "embrace_dialog": embrace_dialog}
    Gui(pages=pages).run(title="Empatica Data Viewer")