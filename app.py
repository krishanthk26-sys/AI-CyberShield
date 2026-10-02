import streamlit as st
import pandas as pd
import pickle

from feature_extractor import extract_features


# Load trained AI model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="AI CyberShield",
    page_icon="🛡️",
    layout="centered"
)


# Header
st.title("🛡️ AI CyberShield")
st.subheader("AI-Based Phishing URL Detection System")

st.write(
    "AI CyberShield uses Machine Learning to analyze website URLs "
    "and identify characteristics associated with phishing or "
    "suspicious websites."
)

st.divider()


# URL input
url = st.text_input(
    "🌐 Enter Website URL",
    placeholder="Example: https://www.google.com"
)


# Analyze button
if st.button("🔍 Analyze URL"):

    if not url.strip():
        st.warning("⚠️ Please enter a URL first.")

    else:

        # Extract URL features
        features = extract_features(url)

        # Convert features into DataFrame
        feature_data = pd.DataFrame([features])

        # AI prediction
        prediction = model.predict(feature_data)[0]

        # Model probability
        probability = model.predict_proba(feature_data)[0]

        # Phishing probability
        phishing_probability = probability[1] * 100

        st.divider()

        # Result
        st.subheader("🧠 AI Analysis Result")

        if prediction == 1:

            st.error("🚨 SUSPICIOUS / POSSIBLE PHISHING URL")

            st.metric(
                "Model-Estimated Phishing Probability",
                f"{phishing_probability:.1f}%"
            )

            st.progress(int(phishing_probability))
            if phishing_probability >= 70:
                st.error("🔴 Risk Level: HIGH")
            elif phishing_probability >= 40:
                st.warning("🟠 Risk Level: MEDIUM")
            else:
                st.success("🟢 Risk Level: LOW")
            st.warning(
                "The machine-learning model detected URL "
                "characteristics associated with suspicious websites."
            )

        else:

            st.success("✅ LIKELY LEGITIMATE URL")

            st.metric(
                "Model-Estimated Phishing Probability",
                f"{phishing_probability:.1f}%"
            )

            st.progress(int(phishing_probability))

            st.info(
                "The machine-learning model did not detect "
                "strong phishing-like characteristics."
            )


        # Feature analysis
        st.subheader("🔎 URL Feature Analysis")

        feature_table = pd.DataFrame(
            features.items(),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True
        )


        # Security indicators
        st.subheader("💡 Security Indicators")
        st.info(
            "The AI analyzes multiple characteristics of the URL, "
            "rather than relying on a single rule."
        )
        indicators = []

        if features["has_https"] == 1:
            indicators.append("🔒 HTTPS is present.")
        else:
            indicators.append("⚠️ HTTPS is not present.")

        if features["has_ip"] == 1:
            indicators.append("⚠️ The URL contains an IP address.")

        if features["has_suspicious_words"] == 1:
            indicators.append(
                "⚠️ The URL contains words commonly associated "
                "with suspicious links."
            )

        if features["num_hyphens"] > 2:
            indicators.append(
                "⚠️ The URL contains multiple hyphens."
            )

        if features["num_dots"] > 3:
            indicators.append(
                "⚠️ The URL contains many dots/subdomains."
            )

        if features["url_length"] > 75:
            indicators.append(
                "⚠️ The URL is relatively long."
            )

        if not indicators:
            indicators.append(
                "✅ No major indicators were detected."
            )

        for indicator in indicators:
            st.write(indicator)


# Footer
st.divider()

st.caption(
    "AI CyberShield • Machine Learning Cybersecurity Project"
)

st.caption(
    "Educational prototype — model results should not be treated "
    "as a guarantee that a website is safe."
)