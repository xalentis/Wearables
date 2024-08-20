import math
import numpy as np
import pandas as pd
import xgboost as xgb
from taipy.gui import Gui, notify
import taipy.gui.builder as tgb
from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None 

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
        "automargin":True,
        "title":None,
        "tickfont": {
            "size":10
        }
    },
    "yaxis": {
        "visible": True,
        "automargin":True,
        "title":None,
        "tickfont": {
            "size":10
        }
    }
}

correlation_chart_config = {
    "displayModeBar": False,
    "displaylogo":False
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

# E4
e4_hr_visible = False
e4_eda_visible = False
e4_bvp_visible = False
e4_ibi_visible = False
e4_temp_visible = False
e4_acc_visible = False
e4_pred_visible = False
e4_stat_visible = False
e4_eda_data = None
e4_hr_data = None
e4_bvp_data = None
e4_ibi_data = None
e4_temp_data = None
e4_acc_data = None
e4_eda_data_chart = pd.DataFrame(columns=["Period", "EDA"])
e4_hr_data_chart = pd.DataFrame(columns=["Period", "HR"])
e4_bvp_data_chart = pd.DataFrame(columns=["Period", "BVP"])
e4_ibi_data_chart = pd.DataFrame(columns=["Period", "IBI"])
e4_temp_data_chart = pd.DataFrame(columns=["Period", "TEMP"])
e4_acc_data_chart = pd.DataFrame(columns=["Period", "Magnitude"])
e4_correlation_data_chart = pd.DataFrame(columns=["x","y","z"])
e4_hr_hist_data_chart = []
e4_eda_hist_data_chart = []
e4_temp_hist_data_chart = []
e4_pred_data_chart = pd.DataFrame(columns=["Period", "Stress", "Attention", "Valence", "Arousal"])

# embrace
emb_activity_counts_data = None
emb_activity_counts_data_chart = pd.DataFrame(columns=["Period", "Activity Counts"])
emb_activity_counts_visible = False
emb_pulse_rate_data = None
emb_pulse_rate_data_chart = pd.DataFrame(columns=["Period", "Pulse Rate"])
emb_pulse_rate_visible = False
emb_eda_data = None
emb_eda_data_chart = pd.DataFrame(columns=["Period", "EDA"])
emb_eda_visible = False
emb_acc_data = None
emb_acc_data_chart = pd.DataFrame(columns=["Period", "Magnitude"])
emb_acc_visible = False
emb_prv_data = None
emb_prv_data_chart = pd.DataFrame(columns=["Period", "PRV"])
emb_prv_visible = False
emb_temp_data = None
emb_temp_data_chart = pd.DataFrame(columns=["Period", "TEMP"])
emb_temp_visible = False
emb_step_counts_data = None
emb_step_counts_data_chart = pd.DataFrame(columns=["Period", "STEPS"])
emb_step_counts_visible = False
emb_respiratory_rate_data = None
emb_respiratory_rate_data_chart = pd.DataFrame(columns=["Period", "RATE"])
emb_respiratory_rate_visible = False
emb_stat_visible = False
emb_hr_hist_data_chart = []
emb_eda_hist_data_chart = []
emb_temp_hist_data_chart = []
emb_pred_visible = False
emb_pred_data_chart = pd.DataFrame(columns=["Period", "Stress", "Attention", "Valence", "Arousal"])
emb_correlation_data_chart = pd.DataFrame(columns=["x","y","z"])

# oura
oura_stat_visible = False
oura_data = None
oura_sleep_visible = False
oura_sleep_data = None
oura_sleep_data_chart = pd.DataFrame(columns=["Date", "Score", "Total Score", "REM Score", "Deep Score", "Tranquility Score", "Latency Score", "Timing Score"])
oura_biomarkers_visible = False
oura_biomarkers_data = None
oura_biomarkers_data_chart = pd.DataFrame(columns=["Date", "Average Resting Heart Rate", "Lowest Resting Heart Rate", "Average HRV", "Respiratory Rate"])
oura_activity_visible = False
oura_activity_data = None
oura_activity_data_chart = pd.DataFrame(columns=["Date", "Activity Score", "Stay Active Score", "Move Every Hour Score", "Meet Daily Targets Score", "Training Frequency Score", "Training Volume Score", "Recovery Time Score"])
oura_calories_visible = False
oura_calories_data = None
oura_calories_data_chart = pd.DataFrame(columns=["Date", "Activity Burn", "Total Burn", "Target Calories"])
oura_movement_visible = False
oura_movement_data = None
oura_movement_data_chart = pd.DataFrame(columns=["Date", "Steps", "Daily Movement", "Inactive Time", "Rest Time", "Low Activity Time", "Medium Activity Time", "High Activity Time", "Non-wear Time"])
oura_stat_visible = False
oura_correlation_data_chart = pd.DataFrame(columns=["x","y","z"])

# endregion

# region File Upload/Download

file_content = None

def on_upload_oura(state):
    files = None
    if type(state.file_content) is str:
        files = state.file_content.split(";")
    else:
        files = state.file_content
    for file in files:
        state.oura_data = pd.read_csv(file)
        state.oura_sleep_data = state.oura_data[["date", "Sleep Score", "Total Sleep Score", "REM Sleep Score", "Deep Sleep Score", "Sleep Tranquility Score", "Sleep Latency Score", "Sleep Timing Score"]] 
        state.oura_sleep_data.dropna(inplace=True)
        state.oura_sleep_data_chart = state.oura_sleep_data.rename(columns={"date": "Date", "Sleep Score": "Score", "Total Sleep Score": "Total Score", "REM Sleep Score": "REM Score", "Deep Sleep Score": "Deep Score", "Sleep Tranquility Score": "Tranquility Score", "Sleep Latency Score": "Latency Score", "Sleep Timing Score": "Timing Score"})
        state.oura_sleep_visible = True
        state.oura_biomarkers_data = state.oura_data[["date", "Average Resting Heart Rate", "Lowest Resting Heart Rate", "Average HRV", "Respiratory Rate"]]
        state.oura_biomarkers_data.dropna(inplace=True)
        state.oura_biomarkers_data_chart = state.oura_biomarkers_data.rename(columns={"date": "Date"})
        state.oura_biomarkers_visible = True
        state.oura_activity_data = state.oura_data[["date", "Activity Score", "Stay Active Score", "Move Every Hour Score", "Meet Daily Targets Score", "Training Frequency Score", "Training Volume Score", "Recovery Time Score"]]
        state.oura_activity_data.dropna(inplace=True)
        state.oura_activity_data_chart = state.oura_activity_data.rename(columns={"date": "Date"})
        state.oura_activity_visible = True
        state.oura_calories_data = state.oura_data[["date", "Activity Burn", "Total Burn", "Target Calories"]]
        state.oura_calories_data.dropna(inplace=True)
        state.oura_calories_data_chart = state.oura_calories_data.rename(columns={"date": "Date"})
        state.oura_calories_visible = True
        state.oura_movement_data = state.oura_data[["date", "Steps", "Daily Movement", "Inactive Time", "Rest Time", "Low Activity Time", "Medium Activity Time", "High Activity Time", "Non-wear Time"]]
        state.oura_movement_data.dropna(inplace=True)
        state.oura_movement_data_chart = state.oura_movement_data.rename(columns={"date": "Date"})
        state.oura_movement_visible = True
        combined_df = pd.merge(state.oura_sleep_data, state.oura_biomarkers_data, on="date", how="inner")
        combined_df = pd.merge(combined_df, state.oura_activity_data, on="date", how="inner")
        combined_df = pd.merge(combined_df, state.oura_calories_data, on="date", how="inner")
        combined_df = pd.merge(combined_df, state.oura_movement_data, on="date", how="inner")
        combined_df.drop("date", axis=1, inplace=True)
        combined_df.drop("Non-wear Time", axis=1, inplace=True)
        combined_df.drop("Sleep Score", axis=1, inplace=True)
        combined_df.drop("Lowest Resting Heart Rate", axis=1, inplace=True)
        combined_df.drop("Stay Active Score", axis=1, inplace=True)
        combined_df.drop("Meet Daily Targets Score", axis=1, inplace=True)
        combined_df.drop("Recovery Time Score", axis=1, inplace=True)
        combined_df.drop("Activity Burn", axis=1, inplace=True)
        combined_df.drop("Target Calories", axis=1, inplace=True)
        combined_df.drop("Steps", axis=1, inplace=True)
        combined_df.drop("Low Activity Time", axis=1, inplace=True)
        combined_df.drop("Medium Activity Time", axis=1, inplace=True)
        combined_df.drop("High Activity Time", axis=1, inplace=True)
        combined_df.columns = combined_df.columns.str.replace("Score", "", regex=False)
        combined_df.columns = combined_df.columns.str.replace("Rate", "", regex=False)
        combined_df.columns = combined_df.columns.str.strip()
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
        state.oura_correlation_data_chart = {"x": x, "y": y, "z": z}
        state.oura_stat_visible = True

def on_upload_embrace(state):
    state.emb_activity_counts_data = None
    files = None
    if type(state.file_content) is str:
        files = state.file_content.split(";")
    else:
        files = state.file_content
    for file in files:
        if "_RESPIRATORY-RATE"  in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_respiratory_rate_data is None:
                state.emb_respiratory_rate_data = temp.copy()
            else:
                state.emb_respiratory_rate_data = pd.concat([state.emb_respiratory_rate_data, temp], axis=0)
            state.emb_respiratory_rate_data = state.emb_respiratory_rate_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_respiratory_rate_data["timestamp_iso"].tolist()
            rate = state.emb_respiratory_rate_data["respiratory_rate_brpm"].tolist()
            state.emb_respiratory_rate_data_chart = {"Period": timestamps, "RATE": rate}
            state.emb_respiratory_rate_visible = True
        if "_STEP-COUNTS"  in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_step_counts_data is None:
                state.emb_step_counts_data = temp.copy()
            else:
                state.emb_step_counts_data = pd.concat([state.emb_step_counts_data, temp], axis=0)
            state.emb_step_counts_data = state.emb_step_counts_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_step_counts_data["timestamp_iso"].tolist()
            steps = state.emb_step_counts_data["step_counts"].tolist()
            state.emb_step_counts_data_chart = {"Period": timestamps, "STEPS": steps}
            state.emb_step_counts_visible = True
        if "_TEMPERATURE"  in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_temp_data is None:
                state.emb_temp_data = temp.copy()
            else:
                state.emb_temp_data = pd.concat([state.emb_temp_data, temp], axis=0)
            state.emb_temp_data = state.emb_temp_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_temp_data["timestamp_iso"].tolist()
            temperature = state.emb_temp_data["temperature_celsius"].tolist()
            state.emb_temp_hist_data_chart = state.emb_temp_data["temperature_celsius"].tolist()
            state.emb_temp_data_chart = {"Period": timestamps, "TEMP": temperature}
            state.emb_temp_visible = True
        if "_PRV"  in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_prv_data is None:
                state.emb_prv_data = temp.copy()
            else:
                state.emb_prv_data = pd.concat([state.emb_prv_data, temp], axis=0)
            state.emb_prv_data = state.emb_prv_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_prv_data["timestamp_iso"].tolist()
            prv = state.emb_prv_data["prv_rmssd_ms"].tolist()
            state.emb_prv_data_chart = {"Period": timestamps, "PRV": prv}
            state.emb_prv_visible = True
        if "_ACCELEROMETERS-STD"  in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_acc_data is None:
                state.emb_acc_data = temp.copy()
            else:
                state.emb_acc_data = pd.concat([state.emb_acc_data, temp], axis=0)
            state.emb_acc_data = state.emb_acc_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_acc_data["timestamp_iso"].tolist()
            magnitude = state.emb_acc_data["accelerometers_std_g"].tolist()
            state.emb_acc_data_chart = {"Period": timestamps, "Magnitude": magnitude}
            state.emb_acc_visible = True
        if "_ACTIVITY-COUNTS" in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_activity_counts_data is None:
                state.emb_activity_counts_data = temp.copy()
            else:
                state.emb_activity_counts_data = pd.concat([state.emb_activity_counts_data, temp], axis=0)
            state.emb_activity_counts_data = state.emb_activity_counts_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_activity_counts_data["timestamp_iso"].tolist()
            activity_counts = state.emb_activity_counts_data["activity_counts"].tolist()
            state.emb_activity_counts_data_chart = {"Period": timestamps, "Activity Counts": activity_counts}
            state.emb_activity_counts_visible = True
        if "_PULSE-RATE" in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_pulse_rate_data is None:
                state.emb_pulse_rate_data = temp.copy()
            else:
                state.emb_pulse_rate_data = pd.concat([state.emb_pulse_rate_data, temp], axis=0)
            state.emb_pulse_rate_data = state.emb_pulse_rate_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_pulse_rate_data["timestamp_iso"].tolist()
            pulse_rate = state.emb_pulse_rate_data["pulse_rate_bpm"].tolist()
            state.emb_hr_hist_data_chart = state.emb_pulse_rate_data["pulse_rate_bpm"].tolist()
            state.emb_pulse_rate_data_chart = {"Period": timestamps, "Pulse Rate": pulse_rate}
            state.emb_pulse_rate_visible = True
        if "_EDA" in file.upper():
            temp = pd.read_csv(file)
            temp = temp[temp["missing_value_reason"].isna()]
            if state.emb_eda_data is None:
                state.emb_eda_data = temp.copy()
            else:
                state.emb_eda_data = pd.concat([state.emb_eda_data, temp], axis=0)
            state.emb_eda_data = state.emb_eda_data.sort_values(by=["timestamp_iso"], ascending=True)
            timestamps = state.emb_eda_data["timestamp_iso"].tolist()
            eda = state.emb_eda_data["eda_scl_usiemens"].tolist()
            state.emb_eda_hist_data_chart = state.emb_eda_data["eda_scl_usiemens"].tolist()
            state.emb_eda_data_chart = {"Period": timestamps, "EDA": eda}
            state.emb_eda_visible = True
    # correlation
    tables = []
    if state.emb_pulse_rate_data.shape[0] > 0:
        tables.append(state.emb_pulse_rate_data[["timestamp_unix", "pulse_rate_bpm"]])
    if state.emb_eda_data.shape[0] > 0:
        tables.append(state.emb_eda_data[["timestamp_unix", "eda_scl_usiemens"]])
    if state.emb_prv_data.shape[0] > 0:
        tables.append(state.emb_prv_data[["timestamp_unix", "prv_rmssd_ms"]])
    if state.emb_temp_data.shape[0] > 0:
        tables.append(state.emb_temp_data[["timestamp_unix", "temperature_celsius"]])
    if state.emb_acc_data.shape[0] > 0:
        tables.append(state.emb_acc_data[["timestamp_unix", "accelerometers_std_g"]])
    if state.emb_activity_counts_data.shape[0] > 0:
        tables.append(state.emb_activity_counts_data[["timestamp_unix", "activity_counts"]])
    if state.emb_step_counts_data.shape[0] > 0:
        tables.append(state.emb_step_counts_data[["timestamp_unix", "step_counts"]])
    if state.emb_respiratory_rate_data.shape[0] > 0:
        tables.append(state.emb_respiratory_rate_data[["timestamp_unix", "respiratory_rate_brpm"]])
    if len(tables) > 1:
        state.emb_stat_visible = True
        combined_df = pd.concat([df.iloc[:, 1].reset_index(drop=True) for df in tables], axis=1)
        if "pulse_rate_bpm" in combined_df.columns:
            combined_df.rename(columns={"pulse_rate_bpm": "PR"}, inplace=True)
        if "eda_scl_usiemens" in combined_df.columns:
            combined_df.rename(columns={"eda_scl_usiemens": "EDA"}, inplace=True)
        if "prv_rmssd_ms" in combined_df.columns:
            combined_df.rename(columns={"prv_rmssd_ms": "PRV"}, inplace=True)
        if "activity_counts" in combined_df.columns:
            combined_df.rename(columns={"activity_counts": "ACT"}, inplace=True)
        if "temperature_celsius" in combined_df.columns:
            combined_df.rename(columns={"temperature_celsius": "TEMP"}, inplace=True)
        if "accelerometers_std_g" in combined_df.columns:
            combined_df.rename(columns={"accelerometers_std_g": "ACC"}, inplace=True)
        if "step_counts" in combined_df.columns:
            combined_df.rename(columns={"step_counts": "STEPS"}, inplace=True)
        if "respiratory_rate_brpm" in combined_df.columns:
            combined_df.rename(columns={"respiratory_rate_brpm": "RESP"}, inplace=True)
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
        state.emb_correlation_data_chart = {"x": x, "y": y, "z": z}
    # predictions
    if state.emb_pulse_rate_data is not None and state.emb_eda_data is not None: # used for prediction, so we need both
        hr = state.emb_pulse_rate_data["pulse_rate_bpm"].tolist()
        eda = state.emb_eda_data["eda_scl_usiemens"].tolist()
        if len(eda) < len(hr):
            hr = hr[0:len(eda)]
        else:
            eda = eda[0:len(hr)]
        periods = state.emb_pulse_rate_data["timestamp_iso"].tolist()[0:len(hr)]
        temp = pd.DataFrame({"hr": hr,"eda": eda})
        stress = model_stress.predict(xgb.DMatrix(temp[["hr","eda"]]))
        attention = model_attention.predict(xgb.DMatrix(temp[["hr","eda"]]))
        arousal = model_arousal.predict(xgb.DMatrix(temp[["hr","eda"]]))
        valence = model_valence.predict(xgb.DMatrix(temp[["hr","eda"]]))
        temp = pd.DataFrame({"period": periods,\
                             "stress": stress, \
                             "attention": attention, \
                             "arousal": arousal, \
                             "valance": valence})
        state.emb_pred_data_chart = {"Period": temp["period"].tolist(), \
                                 "Stress": temp["stress"].tolist(), \
                                 "Attention": temp["attention"].tolist(), \
                                 "Valence": temp["valance"].tolist(),
                                 "Arousal": temp["arousal"].tolist()}
        state.emb_pred_visible = True

def on_upload_e4(state):
    files = None
    if type(state.file_content) is str:
        files = state.file_content.split(";")
    else:
        files = state.file_content
    for file in files:
        if "ACC" in file.upper():
            state.e4_acc_data = pd.read_csv(file, header=None)
            start_time = state.e4_acc_data.values[0]
            sampling_rate = state.e4_acc_data.values[1]
            x_data = state.e4_acc_data.iloc[2:][0].tolist()
            y_data = state.e4_acc_data.iloc[2:][1].tolist()
            z_data = state.e4_acc_data.iloc[2:][2].tolist()
            g_magnitude = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_data, y_data, z_data)]
            g_magnitude = [np.mean(g_magnitude[i:i+2]) for i in range(0, len(g_magnitude), int(sampling_rate[0]))]
            timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(g_magnitude))]
            state.e4_acc_data = pd.DataFrame(columns=["datetime", "magnitude"])
            state.e4_acc_data["datetime"] = timestamps
            state.e4_acc_data["magnitude"] = g_magnitude
            state.e4_acc_data_chart = {"Period": state.e4_acc_data["datetime"].tolist(), "Magnitude": state.e4_acc_data["magnitude"].tolist()}
            state.e4_acc_visible = True
        if "IBI" in file.upper():
            try:
                state.e4_ibi_data = pd.read_csv(file, header=None)
                start_time = state.e4_ibi_data.values[0][0]
                intervals = state.e4_ibi_data.values[1:][:,0].tolist()
                ibi = state.e4_ibi_data.values[1:][:,1].tolist()
                ibi = [float(i) for i in ibi]
                timestamps = [datetime.fromtimestamp(start_time) + timedelta(seconds=i) for i in intervals]
                state.e4_ibi_data = pd.DataFrame(columns=["datetime", "ibi"])
                state.e4_ibi_data["datetime"] = timestamps
                state.e4_ibi_data["ibi"] = ibi
                state.e4_ibi_data_chart = {"Period": state.e4_ibi_data["datetime"].tolist(), "IBI": state.e4_ibi_data["ibi"].tolist()}
                state.e4_ibi_visible = True
            except:
                state.e4_ibi_data_chart = None
                state.e4_ibi_visible = False
                state.e4_ibi_data = None
                notify(state, "error", "Error processing IBI data...")
        if "TEMP" in file.upper():
            try:
                state.e4_temp_data = pd.read_csv(file, header=None)
                start_time = state.e4_temp_data.values[0]
                sampling_rate = state.e4_temp_data.values[1]
                temp = state.e4_temp_data.iloc[2:][0].tolist()
                # temp sensor takes time to settle, so average first 10 seconds
                avg_temp = np.mean(temp[10:])
                temp[0:9] = [avg_temp] * 10
                temp = [np.mean(temp[i:i+2]) for i in range(0, len(temp), int(sampling_rate[0]))]
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(temp))]
                state.e4_temp_data = pd.DataFrame(columns=["datetime", "temp"])
                state.e4_temp_data["datetime"] = timestamps
                state.e4_temp_data["temp"] = temp
                state.e4_temp_hist_data_chart = state.e4_temp_data["temp"].tolist()
                state.e4_temp_data_chart = {"Period": state.e4_temp_data["datetime"].tolist(), "TEMP": state.e4_temp_data["temp"].tolist()}
                state.e4_temp_visible = True
            except:
                state.e4_temp_data_chart = None
                state.e4_temp_visible = False
                state.e4_temp_data = None
                notify(state, "error", "Error processing TEMP data...")
        if "BVP" in file.upper():
            try:
                state.e4_bvp_data = pd.read_csv(file, header=None)
                start_time = state.e4_bvp_data.values[0]
                sampling_rate = state.e4_bvp_data.values[1]
                bvp = state.e4_bvp_data.iloc[2:][0].tolist()
                bvp = [np.mean(bvp[i:i+2]) for i in range(0, len(bvp), int(sampling_rate[0]))]
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(bvp))]
                state.e4_bvp_data = pd.DataFrame(columns=["datetime", "bvp"])
                state.e4_bvp_data["datetime"] = timestamps
                state.e4_bvp_data["bvp"] = bvp
                state.e4_bvp_data_chart = {"Period": state.e4_bvp_data["datetime"].tolist(), "BVP": state.e4_bvp_data["bvp"].tolist()}
                state.e4_bvp_visible = True
            except:
                state.e4_bvp_data_chart = None
                state.e4_bvp_visible = False
                state.e4_bvp_data = None
                notify(state, "error", "Error processing BVP data...")
        if "EDA" in file.upper():
            try:
                state.e4_eda_data = pd.read_csv(file, header=None)
                start_time = state.e4_eda_data.values[0]
                sampling_rate = state.e4_eda_data.values[1]
                eda = state.e4_eda_data.iloc[2:][0].tolist()
                eda = [np.mean(eda[i:i+4]) for i in range(0, len(eda), int(sampling_rate[0]))]
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(eda))]
                state.e4_eda_data = pd.DataFrame(columns=["datetime", "eda"])
                state.e4_eda_data["datetime"] = timestamps
                state.e4_eda_data["eda"] = eda
                state.e4_eda_hist_data_chart = state.e4_eda_data["eda"].tolist()
                state.e4_eda_data_chart = {"Period": state.e4_eda_data["datetime"].tolist(), "EDA": state.e4_eda_data["eda"].tolist()}
                state.e4_eda_visible = True
            except:
                state.e4_eda_data_chart = None
                state.e4_eda_visible = False
                state.e4_eda_data = None
                notify(state, "error", "Error processing EDA data...")
        if "HR" in file.upper():
            try:
                state.e4_hr_data = pd.read_csv(file, header=None)
                start_time = state.e4_hr_data.values[0]
                hr = state.e4_hr_data.iloc[2:][0].tolist()
                timestamps = [datetime.fromtimestamp(start_time[0]) + timedelta(seconds=i) for i in range(len(hr))]
                state.e4_hr_data = pd.DataFrame(columns=["datetime", "hr"])
                state.e4_hr_data["datetime"] = timestamps
                state.e4_hr_data["hr"] = hr
                state.e4_hr_hist_data_chart = state.e4_hr_data["hr"].tolist()
                state.e4_hr_data_chart = {"Period": state.e4_hr_data["datetime"].tolist(), "HR": state.e4_hr_data["hr"].tolist()}
                state.e4_hr_visible = True
            except:
                state.e4_hr_data_chart = None
                state.e4_hr_visible = False
                state.e4_hr_data = None
                notify(state, "error", "Error processing HR data...")
    # correlation
    tables = []
    if state.e4_hr_data.shape[0] > 0:
        tables.append(state.e4_hr_data)
    if state.e4_eda_data.shape[0] > 0:
        tables.append(state.e4_eda_data)
    if state.e4_bvp_data.shape[0] > 0:
        tables.append(state.e4_bvp_data)
    if state.e4_ibi_data.shape[0] > 0:
        tables.append(state.e4_ibi_data)
    if state.e4_temp_data.shape[0] > 0:
        tables.append(state.e4_temp_data)
    if state.e4_acc_data.shape[0] > 0:
        tables.append(state.e4_acc_data)
    if len(tables) > 1:
        state.e4_stat_visible = True
        combined_df = pd.concat([df.iloc[:, 1].reset_index(drop=True) for df in tables], axis=1)
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
        state.e4_correlation_data_chart = {"x": x, "y": y, "z": z}
    # predictions
    if state.e4_hr_data is not None and state.e4_eda_data is not None: # used for prediction, so we need both
        hr = state.e4_hr_data["hr"].tolist()
        eda = state.e4_eda_data["eda"].tolist()
        if len(eda) < len(hr):
            hr = hr[0:len(eda)]
        else:
            eda = eda[0:len(hr)]
        periods = state.e4_hr_data["datetime"].tolist()[0:len(hr)]
        temp = pd.DataFrame({"hr": hr,"eda": eda})
        stress = model_stress.predict(xgb.DMatrix(temp[["hr","eda"]]))
        attention = model_attention.predict(xgb.DMatrix(temp[["hr","eda"]]))
        arousal = model_arousal.predict(xgb.DMatrix(temp[["hr","eda"]]))
        valence = model_valence.predict(xgb.DMatrix(temp[["hr","eda"]]))
        temp = pd.DataFrame({"period": periods,\
                             "stress": stress, \
                             "attention": attention, \
                             "arousal": arousal, \
                             "valance": valence})
        state.e4_pred_data_chart = {"Period": temp["period"].tolist(), \
                                 "Stress": temp["stress"].tolist(), \
                                 "Attention": temp["attention"].tolist(), \
                                 "Valence": temp["valance"].tolist(),
                                 "Arousal": temp["arousal"].tolist()}
        state.e4_pred_visible = True

# endregion

with tgb.Page() as page_main:
    with tgb.layout("0.5 1 0.5"):
        tgb.text("")
        tgb.text("### Wearable Device Analysis", mode="md", class_name="center_text")
        tgb.text("")
    with tgb.layout("1 1"):
        with tgb.part(class_name="buttons"):
            tgb.file_selector("{file_content}", label="Import E4 Data", on_action=on_upload_e4, extensions=".csv", drop_message="Drop To Process", multiple=True)
            tgb.file_selector("{file_content}", label="Import Embrace Data", on_action=on_upload_embrace, extensions=".csv", drop_message="Drop To Process", multiple=True)
            tgb.file_selector("{file_content}", label="Import Oura Data", on_action=on_upload_oura, extensions=".csv", drop_message="Drop To Process", multiple=False)
    with tgb.layout("1 1fs"):
        # embrace
        with tgb.part(render="{emb_activity_counts_visible}"):
            with tgb.expandable(title="Activity Counts", expanded=True):
                tgb.chart("{emb_activity_counts_data_chart}", height="300px", x="Period", y="Activity Counts", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_pulse_rate_visible}"):
            with tgb.expandable(title="Pulse Rate", expanded=True):
                tgb.chart("{emb_pulse_rate_data_chart}", height="300px", x="Period", y="Pulse Rate", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_eda_visible}"):
            with tgb.expandable(title="Electrodermal Activity", expanded=True):
                tgb.chart("{emb_eda_data_chart}", height="300px", x="Period", y="EDA", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_acc_visible}"):
            with tgb.expandable(title="Movement", expanded=True):
                tgb.chart("{emb_acc_data_chart}", height="300px", x="Period", y="Magnitude", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_prv_visible}"):
            with tgb.expandable(title="Pulse Rate Variation", expanded=True):
                tgb.chart("{emb_prv_data_chart}", height="300px", x="Period", y="PRV", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_temp_visible}"):
            with tgb.expandable(title="Temperature", expanded=True):
                tgb.chart("{emb_temp_data_chart}", height="300px", x="Period", y="TEMP", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_step_counts_visible}"):
            with tgb.expandable(title="Step Count", expanded=True):
                tgb.chart("{emb_step_counts_data_chart}", height="300px", x="Period", y="STEPS", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{emb_respiratory_rate_visible}"):
            with tgb.expandable(title="Respiratory Rate", expanded=True):
                tgb.chart("{emb_respiratory_rate_data_chart}", height="300px", x="Period", y="RATE", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part():
            with tgb.part(render="{emb_stat_visible}"):
                with tgb.expandable(title="Statistics", expanded=False):
                    with tgb.layout("1 1 1 1"):
                        with tgb.part():
                            tgb.chart("{emb_correlation_data_chart}", type="heatmap", title="Correlation", x="x", y="y", z="z", height="300px", rebuild=True, options="{correlation_chart_options}", layout="{correlation_chart_layout}")
                        with tgb.part(render="{emb_prv_visible}"):
                            tgb.chart("{emb_hr_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="HR", layout="{hist_chart_layout}")
                        with tgb.part(render="{emb_eda_visible}"):
                            tgb.chart("{emb_eda_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="EDA", layout="{hist_chart_layout}")
                        with tgb.part(render="{emb_temp_visible}"):
                            tgb.chart("{emb_temp_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="TEMP", layout="{hist_chart_layout}")                 
        with tgb.part(render="{emb_pred_visible}"):
            with tgb.expandable(title="Emotional State", expanded=False):
                with tgb.part():
                    tgb.chart("{emb_pred_data_chart}", x="Period", y__1="Stress", y__2="Arousal", y__3="Valence", y__4="Attention", height="300px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
               
        # empatica e4
        with tgb.part(render="{e4_hr_visible}"):
            with tgb.expandable(title="Heart Rate", expanded=True):
                tgb.chart("{e4_hr_data_chart}", height="300px", x="Period", y="HR", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{e4_eda_visible}"):
            with tgb.expandable(title="Electrodermal Activity", expanded=True):
                tgb.chart("{e4_eda_data_chart}", height="300px", x="Period", y="EDA", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{e4_bvp_visible}"):
            with tgb.expandable(title="Blood Volume Pulse", expanded=True):
                tgb.chart("{e4_bvp_data_chart}", height="300px", x="Period", y="BVP", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{e4_temp_visible}"):
            with tgb.expandable(title="Temperature", expanded=True):
                tgb.chart("{e4_temp_data_chart}", height="300px", x="Period", y="TEMP", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{e4_ibi_visible}"):
            with tgb.expandable(title="Interbeat Interval", expanded=True):
                tgb.chart("{e4_ibi_data_chart}", height="300px", x="Period", y="IBI", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part(render="{e4_acc_visible}"):
            with tgb.expandable(title="Movement", expanded=True):
                tgb.chart("{e4_acc_data_chart}", height="300px", x="Period", y="Magnitude", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
        with tgb.part():
            with tgb.part(render="{e4_stat_visible}"):
                with tgb.expandable(title="Statistics", expanded=False):
                    with tgb.layout("1 1 1 1"):
                        with tgb.part():
                            tgb.chart("{e4_correlation_data_chart}", type="heatmap", title="Correlation", x="x", y="y", z="z", height="300px", rebuild=True, options="{correlation_chart_options}", layout="{correlation_chart_layout}")
                        with tgb.part(render="{e4_hr_visible}"):
                            tgb.chart("{e4_hr_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="HR", layout="{hist_chart_layout}")
                        with tgb.part(render="{e4_eda_visible}"):
                            tgb.chart("{e4_eda_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="EDA", layout="{hist_chart_layout}")
                        with tgb.part(render="{e4_temp_visible}"):
                            tgb.chart("{e4_temp_hist_data_chart}", type="histogram", height="300px", rebuild=True, title="TEMP", layout="{hist_chart_layout}")
        with tgb.part(render="{e4_pred_visible}"):
            with tgb.expandable(title="Emotional State", expanded=False):
                with tgb.part():
                    tgb.chart("{e4_pred_data_chart}", x="Period", y__1="Stress", y__2="Arousal", y__3="Valence", y__4="Attention", height="300px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
        # oura
        with tgb.part(render="{oura_sleep_visible}"):
            with tgb.expandable(title="Sleep", expanded=False):
                with tgb.part():
                    tgb.chart("{oura_sleep_data_chart}", x="Date", y__1="Score", y__2="Total Score", y__3="REM Score", y__4="Deep Score", y__5="Tranquility Score", y__6="Latency Score", y__7="Timing Score", height="400px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
        with tgb.part(render="{oura_biomarkers_visible}"):
            with tgb.expandable(title="Biomarkers", expanded=False):
                with tgb.part():
                    tgb.chart("{oura_biomarkers_data_chart}", x="Date", y__1="Average Resting Heart Rate", y__2="Lowest Resting Heart Rate", y__3="Average HRV", y__4="Respiratory Rate", height="400px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
        with tgb.part(render="{oura_activity_visible}"):
            with tgb.expandable(title="Activity", expanded=False):
                with tgb.part():
                    tgb.chart("{oura_activity_data_chart}", x="Date", y__1="Activity Score", y__2="Stay Active Score", y__3="Move Every Hour Score", y__4="Meet Daily Targets Score", y__5="Training Frequency Score", y__6="Training Volume Score", y__7="Recovery Time Score", height="400px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
        with tgb.part(render="{oura_calories_visible}"):
            with tgb.expandable(title="Calories", expanded=False):
                with tgb.part():
                    tgb.chart("{oura_calories_data_chart}", x="Date", y__1="Activity Burn", y__2="Total Burn", y__3="Target Calories", height="400px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
        with tgb.part(render="{oura_movement_visible}"):
            with tgb.expandable(title="Movement", expanded=False):
                with tgb.part():
                    tgb.chart("{oura_movement_data_chart}", x="Date", y__1="Steps", y__2="Daily Movement", y__3="Inactive Time", y__4="Rest Time", y__5="Low Activity Time", y__6="Medium Activity Time", y__7="High Activity Time", y__8="Non-wear Time", height="400px", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{pred_chart_properties}")
        with tgb.part():
            with tgb.part(render="{oura_stat_visible}"):
                with tgb.expandable(title="Correlation", expanded=False):
                    with tgb.part():
                        tgb.chart("{oura_correlation_data_chart}", type="heatmap", x="x", y="y", z="z", rebuild=True, options="{correlation_chart_options}", layout="{correlation_chart_layout}", class_name="center_text", plot_config="{correlation_chart_config}")
if __name__ == "__main__":
    pages = {"page_main": page_main}
    Gui(pages=pages).run(title="Wearable Device Analysis")