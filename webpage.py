import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np


# loading the models

heart_disease = pickle.load(open("project_model.pkl", "rb"))

lung_cancer = pickle.load(open("lung_cancer_model.pkl", "rb"))


with st.sidebar:

    selected = option_menu(
        "Multiple Disease Prediction System using Machine Learning",
        ["Heart Disese Prediction", "Lung Cancer Prediction"],
        icons=["heart-fill", "apple"],
        default_index=0,
    )

if selected == "Heart Disese Prediction":

    st.title("Heart Attack Risk Prediction")
    st.write(
        "This web app predicts the risk of developing a heart disease based on user input."
    )

    # User Input Fields
    st.header("Enter Your Information:")
    age = st.number_input("Age:")
    sex = st.number_input("Gender: Male = 0, Female =1")
    cp = st.number_input(
        "Chest Pain Type: Typical Angina = 0, Atypical Angina =1, Non-Anginal Pain =2, Asymptomatic =3"
    )
    trtbps = st.number_input("Resting Blood Pressure:")
    chol = st.number_input("Cholesterol Level:")
    fbs = st.number_input("Fasting Blood Sugar: No = 0, Yes =1")
    restecg = st.number_input(
        "Resting Electrocardiographic Results: Normal =0, ST-T Wave Abnormality =1, Left Ventricular Hypertrophy =2"
    )
    thalachh = st.number_input("Maximum Heart Rate Achieved:")
    exang = st.number_input("Exercise Induced Angina: No = 0, Yes =1")
    oldpeak = st.number_input("ST Depression Induced by Exercise:")
    slope = st.number_input(
        "Heart Rate Slope: Upsloping =0, Flatsloping =1, Downsloping =2"
    )
    caa = st.number_input("Number of Major Vessels Colored by Flourosopy:")
    thall = st.number_input(
        "Thalium Stress Test Result:  Null =0, Normal =1, Fixed Defect =2, Reversible Defect =3"
    )

    if st.button("Predict"):
        input_data = np.array(
            [
                [
                    age,
                    sex,
                    cp,
                    trtbps,
                    chol,
                    fbs,
                    restecg,
                    thalachh,
                    exang,
                    oldpeak,
                    slope,
                    caa,
                    thall,
                ]
            ]
        )
        prediction = heart_disease.predict(input_data)[0]
        risk_message = "low" if prediction == 0 else "high"
        st.write(
            f"Based on your information, you have a *{risk_message}* risk of getting a heart disease."
        )

        # Plot predicted risk
        fig, ax = plt.subplots()
        ax.bar(
            ["Low Risk", "High Risk"],
            [1 - prediction, prediction],
            color=["green", "red"],
        )
        ax.set_ylabel("Probability")
        ax.set_title("Predicted Risk of Heart Disease")
        st.pyplot(fig)


# Lung Cancer Prediction Page:

if selected == "Lung Cancer Prediction":

    st.title("Lung Cancer Prediction")
    st.write(
        "This web app predicts the risk of developing a Lung cancer based on user input."
    )

    st.header("Enter Your Information:")

    GENDER = st.number_input("Gender: Male=1, Female=2")

    AGE = st.number_input("AGE")

    SMOKING = st.number_input("Smoking: No smoking=1 , smoking=2")

    YELLOW_FINGERS = st.number_input(
        "Yellow_fingers: yellowish-absent=1, yellowish-present=2"
    )

    ANXIETY = st.number_input("Anxiety: no-anxiety=1, anxiety=2")

    PEER_PRESSURE = st.number_input("Peer_pressure: no-effect=1, effect=2")

    CHRONIC_DISEASE = st.number_input(
        "Chronic Disease: disease which caused more than one times=1, disease which caused more than one times=2"
    )

    FATIGUE = st.number_input("Fatigue: absent=1, present=2")

    ALLERGY = st.number_input("Allergy: absent=1, present=2")

    WHEEZING = st.number_input("Wheezing: ")

    ALCOHOL_CONSUMING = st.number_input("Alcohol Consuming: not-consume=1, consume=2")

    COUGHING = st.number_input("Coughing: absent=1, present=2")

    SHORTNESS_OF_BREATH = st.number_input(
        "Shortness of breath: less-breath=1, normal=2"
    )

    SWALLOWING_DIFFICULTY = st.number_input(
        "Swallowing Difficulty: no-swallowing=1, swallowing=2"
    )

    CHEST_PAIN = st.number_input("Chest Pain: absent=1, present=2")

    if st.button("Predict"):
        input_data = np.array(
            [
                [
                    GENDER,
                    AGE,
                    SMOKING,
                    YELLOW_FINGERS,
                    ANXIETY,
                    PEER_PRESSURE,
                    CHRONIC_DISEASE,
                    FATIGUE,
                    ALLERGY,
                    WHEEZING,
                    ALCOHOL_CONSUMING,
                    COUGHING,
                    SHORTNESS_OF_BREATH,
                    SWALLOWING_DIFFICULTY,
                    CHEST_PAIN,
                ]
            ]
        )
        prediction = lung_cancer.predict(input_data)[0]
        risk_message = "low" if prediction == 0 else "high"
        st.write(
            f"Based on your information, you have a *{risk_message}* risk of getting a heart disease."
        )

        # st.write("hwllo", prediction)
        # Plot predicted risk
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "High Risk"], [1 - prediction, prediction], color=['green', 'red'])
        ax.set_ylabel('Probability')
        ax.set_title('Predicted Risk of Heart Disease')
        st.pyplot(fig)


st.sidebar.subheader("About Heart Attack")
st.sidebar.info(
    "This web app is helps you to find out whether you are at a risk of developing a heart disease.Understanding heart attack risks and using prediction tools can help prevent heart attacks, promoting better heart health for everyone."
)
st.sidebar.subheader("About Lung cancer")
st.sidebar.info(
    "webpages about lung cancer offer crucial information for understanding risks, promoting early detection, and empowering individuals to make informed health decisions."
)
