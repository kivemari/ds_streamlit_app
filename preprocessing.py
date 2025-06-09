def add_features(df):
    df = df.copy()
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
    df["cardiovascular_condition"] = (
        (df["heart_disease"] == 1) | (df["hypertension"] == 1)
    ).astype(int)
    return df
