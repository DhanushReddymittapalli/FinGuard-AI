import streamlit as st
import pandas as pd
import model.predictor as predictor
st.title("FinGuard AI")
st.write("Test successful")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("demo_transactions_small.csv")


df = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "An AI-powered machine-learning system for detecting "
    "and analyzing suspicious credit-card transactions."
)


# ============================================================
# DASHBOARD
# ============================================================

st.write("### 📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

fraud_transactions = int(df["Class"].sum())

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with c2:
    st.metric(
        "Fraud Cases",
        f"{fraud_transactions:,}"
    )

with c3:
    st.metric(
        "Legitimate",
        f"{legitimate_transactions:,}"
    )

with c4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

st.write("### 📈 Dataset Distribution")

chart_data = pd.DataFrame(
    {
        "Transaction Type": [
            "Fraud",
            "Legitimate"
        ],
        "Count": [
            fraud_transactions,
            legitimate_transactions
        ]
    }
)

st.bar_chart(
    chart_data.set_index("Transaction Type")
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.write("### ⚙️ Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model", "Random Forest")

with m2:
    st.metric("ROC-AUC", "0.953")

with m3:
    st.metric("PR-AUC", "0.854")

with m4:
    st.metric("Threshold", "0.30")


# ============================================================
# BATCH FRAUD DETECTION
# ============================================================

st.write("### 📁 Batch Fraud Detection")

st.write(
    "Upload a CSV containing transactions and "
    "FinGuard AI will analyze them automatically."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(uploaded_file)

        st.write("### 📄 Uploaded Transactions")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        if "Class" in batch_df.columns:
            prediction_data = batch_df.drop(
                columns=["Class"]
            )
        else:
            prediction_data = batch_df.copy()


        if st.button(
            "🚨 Analyze Uploaded Transactions",
            use_container_width=True
        ):

            probabilities = []

            progress_bar = st.progress(0)

            total_rows = len(prediction_data)

            for i in range(total_rows):

                transaction_row = prediction_data.iloc[
                    [[i]]
                ]

                result = predictor.predict_transaction(
                    transaction_row
                )

                probability = float(
                    result["fraud_probability"]
                )

                probabilities.append(
                    probability
                )

                progress_bar.progress(
                    int(
                        ((i + 1) / total_rows) * 100
                    )
                )


            results = batch_df.copy()

            results["Fraud Probability"] = (
                probabilities
            )

            results["Risk Level"] = (
                results["Fraud Probability"]
                .apply(
                    lambda x:
                    "HIGH RISK 🚨"
                    if x >= 0.30
                    else (
                        "MEDIUM RISK ⚠️"
                        if x >= 0.10
                        else "LOW RISK ✅"
                    )
                )
            )

            results["Recommended Action"] = (
                results["Fraud Probability"]
                .apply(
                    lambda x:
                    "BLOCK"
                    if x >= 0.30
                    else (
                        "MANUAL REVIEW"
                        if x >= 0.10
                        else "APPROVE"
                    )
                )
            )


            # =================================================
            # BATCH SUMMARY
            # =================================================

            st.write(
                "### 📊 Batch Analysis Summary"
            )

            predicted_fraud = int(
                (
                    results["Fraud Probability"]
                    >= 0.30
                ).sum()
            )

            predicted_legitimate = (
                len(results) - predicted_fraud
            )

            batch_fraud_rate = (
                predicted_fraud / len(results) * 100
                if len(results) > 0
                else 0
            )


            b1, b2, b3, b4 = st.columns(4)

            with b1:
                st.metric(
                    "Analyzed",
                    f"{len(results):,}"
                )

            with b2:
                st.metric(
                    "High Risk",
                    f"{predicted_fraud:,}"
                )

            with b3:
                st.metric(
                    "Low Risk",
                    f"{predicted_legitimate:,}"
                )

            with b4:
                st.metric(
                    "Fraud Rate",
                    f"{batch_fraud_rate:.2f}%"
                )


            if predicted_fraud > 0:

                st.error(
                    f"🚨 {predicted_fraud} transaction(s) "
                    "require immediate attention."
                )

            else:

                st.success(
                    "✅ No high-risk transactions detected."
                )


            # =================================================
            # RESULTS
            # =================================================

            st.write(
                "### 🔍 Batch Detection Results"
            )

            display_results = results.copy()

            display_results[
                "Fraud Probability"
            ] = display_results[
                "Fraud Probability"
            ].map(
                lambda x: f"{x:.2%}"
            )

            st.dataframe(
                display_results,
                use_container_width=True
            )


            # =================================================
            # DOWNLOAD RESULTS
            # =================================================

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Analysis Results",
                data=csv_data,
                file_name="finguard_batch_results.csv",
                mime="text/csv",
                use_container_width=True
            )


    except Exception:

        st.error(
            "❌ Unable to analyze this CSV."
        )

        st.info(
            "Please upload a CSV containing the same "
            "transaction features used by the model."
        )


# ============================================================
# INDIVIDUAL TRANSACTION ANALYSIS
# ============================================================

st.write("### 🔍 Analyze Real Transaction")

row = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


selected = df.iloc[
    [[int(row)]]
].copy()


actual_label = int(
    selected["Class"].iloc[0]
)


transaction = selected.drop(
    columns=["Class"]
)


st.dataframe(
    transaction,
    use_container_width=True
)


# ============================================================
# ANALYZE TRANSACTION
# ============================================================

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    try:

        result = predictor.predict_transaction(
            transaction
        )

        probability = float(
            result["fraud_probability"]
        )

        risk = result["risk_level"]


        # ====================================================
        # FRAUD ANALYSIS
        # ====================================================

        st.write("### 📊 Fraud Analysis")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with r2:
            st.metric(
                "Risk Level",
                risk
            )

        risk_score = round(
            probability * 100
        )

        with r3:
            st.metric(
                "Risk Score",
                f"{risk_score}/100"
            )


        # ====================================================
        # RISK SCORE
        # ====================================================

        st.write("### 🎯 Fraud Risk Score")

        st.progress(
            min(max(risk_score, 0), 100)
        )


        if risk_score >= 30:

            st.write(
                f"🔴 **Elevated Risk: "
                f"{risk_score}/100**"
            )

        else:

            st.write(
                f"🟢 **Low Risk: "
                f"{risk_score}/100**"
            )


        # ====================================================
        # EXPLANATION
        # ====================================================

        st.write(
            "### 🔎 Why was this transaction flagged?"
        )

        importance = (
            predictor.get_feature_importance(
                transaction
            )
        )


        if (
            importance is not None
            and not importance.empty
        ):

            st.dataframe(
                importance.head(5),
                use_container_width=True
            )

        else:

            st.info(
                "Feature explanation is not available."
            )


        # ====================================================
        # RISK EXPLANATION
        # ====================================================

        if risk.startswith("HIGH"):

            st.error(
                "🚨 HIGH FRAUD RISK\n\n"
                "The model estimates a high probability "
                "of fraudulent activity."
            )

        elif risk.startswith("MEDIUM"):

            st.warning(
                "⚠️ MEDIUM FRAUD RISK\n\n"
                "Additional verification is recommended."
            )

        else:

            st.success(
                "✅ LOW FRAUD RISK\n\n"
                "The model estimates a low probability "
                "of fraud."
            )


        # ====================================================
        # RECOMMENDED ACTION
        # ====================================================

        st.write("### 🛡️ Recommended Action")


        if risk.startswith("HIGH"):

            st.error(
                "🚫 BLOCK TRANSACTION\n\n"
                "High fraud risk detected. "
                "Block the transaction and investigate "
                "immediately."
            )

        elif risk.startswith("MEDIUM"):

            st.warning(
                "🔎 MANUAL REVIEW REQUIRED\n\n"
                "Suspicious activity detected. "
                "Additional verification is recommended "
                "before approval."
            )

        else:

            st.success(
                "✅ APPROVE TRANSACTION\n\n"
                "Low fraud risk detected. "
                "The transaction can proceed."
            )


        # ====================================================
        # ACTUAL DATASET LABEL
        # ====================================================

        st.write("### 🎯 Actual Dataset Label")

        if actual_label == 1:

            st.error("🚨 FRAUD")

        else:

            st.success("✅ LEGITIMATE")


        # ====================================================
        # MODEL VERIFICATION
        # ====================================================

        st.write("### 🧠 Model Verification")

        predicted_fraud = (
            probability >= 0.30
        )


        if predicted_fraud == bool(
            actual_label
        ):

            st.success(
                "✅ Model prediction matches "
                "the actual dataset label."
            )

        else:

            st.warning(
                "⚠️ Model prediction differs from "
                "the actual dataset label."
            )


    except Exception as e:

        st.error(
            "❌ Transaction analysis failed."
        )

        st.info(
            "Please check that predictor.py, the model "
            "file, and transaction features are compatible."
        )


# ============================================================
# FOOTER
# ============================================================

st.write("---")

st.caption(
    "FinGuard AI | Random Forest Fraud Detection | "
    "Explainable AI | Batch Fraud Detection | "
    "ULB Credit Card Fraud Detection Dataset"
                       )
            
