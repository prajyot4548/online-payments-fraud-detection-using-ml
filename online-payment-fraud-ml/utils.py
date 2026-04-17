def create_features(df):
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # 🚨 ADD THESE (YOU MISSED THIS)
    df["errorOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["errorDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    return df