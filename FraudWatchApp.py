import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifacts():
    svc_model = joblib.load("svc_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return svc_model, xgb_model, scaler, label_encoders, feature_names


def format_input(user_input):
    formatted_input = pd.DataFrame(index=[0])
    for i in feature_names:
        if i in label_encoders:
            formatted_input[i] = label_encoders[i].transform([user_input[i]])
        else:
            formatted_input[i] = user_input[i]

    return formatted_input


svc_model, xgb_model, scaler, label_encoders, feature_names = load_artifacts()
# test code
# dd = [
#     4209696857872688515,
#     "Adams-Barrows",
#     "health_fitness",
#     44.48,
#     "F",
#     37.8274,
#     -88.6235,
#     1943,
#     "Restaurant manager, fast food",
#     1385998361,
#     37.096297,
#     -89.224870,
#     15,
#     2,
# ]

# new_dd = {}

# for i, item in enumerate(dd):
#     new_dd[feature_names[i]] = item

# print(feature_names)
# new_dd = format_input(new_dd)
# new_dd = scaler.transform(new_dd)
# pred = xgb_model.predict(new_dd)

# st.text(pred)


st.title("FraudWatch Detection")


user_input = st.file_uploader("Upload .csv file of transactions here:", type=[".csv"])


if user_input:
    user_input = pd.read_csv(user_input, index_col=0)
    user_input["merchant"] = user_input["merchant"].apply(
        lambda x: str(x).replace("fraud_", "")
    )
    st.dataframe(user_input.head())
    # Drop unused text/unique string identifiers to prevent data leakage
    drop_cols = ["trans_num", "first", "last", "street", "city", "state", "zip", "dob"]
    df_clean = user_input.drop(
        columns=[col for col in drop_cols if col in user_input.columns]
    )

    if "is_fraud" in df_clean.columns:
        df_clean = df_clean.drop(columns=["is_fraud"])
    # Convert timestamp to temporal features
    if "trans_date_trans_time" in df_clean.columns:
        df_clean["trans_date_trans_time"] = pd.to_datetime(
            df_clean["trans_date_trans_time"]
        )
        df_clean["hour"] = df_clean["trans_date_trans_time"].dt.hour
        df_clean["day_of_week"] = df_clean["trans_date_trans_time"].dt.dayofweek
        df_clean.drop(columns=["trans_date_trans_time"], inplace=True)
    categorical_cols = ["merchant", "category", "gender", "job"]
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = label_encoders[col].transform(df_clean[col].astype(str))
    df_clean = df_clean[feature_names]
    df_clean = scaler.transform(df_clean)
    st.dataframe(xgb_model.predict(df_clean))

    st.text("Total number of malicious transactions detected: {}".format(xgb_model.predict(df_clean).sum()))
