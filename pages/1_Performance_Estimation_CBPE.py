import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Confidence Based Performance Estimation (CBPE)')

st.markdown("""
- Used for binary and multiclass classification problems
- Leverages confidence scores to estimate confusion matrix
- Estimates any classification performance metric
""")
st.write('Classification model predictions usually come with an associated uncertainty. For example, a binary classification model typically returns two outputs for each prediction - a predicted class (binary) and a class probability estimate (sometimes referred to as score). The score provides information about the confidence of the prediction. A rule of thumb is that the closer the score is to its lower or upper limit (usually 0 and 1), the higher the probability that the classifier’s prediction is correct. When this score is an actual probability, it can be directly used to estimate the probability of making an error. For instance, imagine a high-performing model which, for a large set of observations, returns a prediction of 1 (positive class) with a probability of 0.9. It means that the model is correct for approximately 90% of these observations, while for the other 10%, the model is wrong.')
st.write('Source: https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#confidence-based-performance-estimation-cbpe')

st.header('Example')

with st.spinner('Loading data'):
    # Load real-world data:
    reference_df, analysis_df, _ = nml.load_us_census_ma_employment_data()

st.subheader('Reference')
st.write(reference_df.head())

st.subheader('Analysis')
st.write(analysis_df.head())

st.subheader('Estimated performance')
with st.spinner('Estimating performance'):
    # Choose a chunker or set a chunk size:
    chunk_size = 5000

    # initialize, specify required data columns, fit estimator and estimate:
    estimator = nml.CBPE(
        problem_type='classification_binary',
        y_pred_proba='predicted_probability',
        y_pred='prediction',
        y_true='employed',
        metrics=['roc_auc'],
        chunk_size=chunk_size,
    )
    estimator = estimator.fit(reference_df)
    estimated_performance = estimator.estimate(analysis_df)

st.plotly_chart(estimated_performance.plot())

from utils import display_source_code
display_source_code()
