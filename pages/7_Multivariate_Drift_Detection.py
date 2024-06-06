import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Multivariate Drift Detection')

st.write('Multivariate data drift detection compliments univariate data drift detection methods. It provides one summary number reducing the risk of false alerts, and detects more subtle changes in the data structure that cannot be detected with univariate approaches. The trade off is that multivariate drift results are less explainable compared to univariate drift results.')

st.markdown("""Source:
- https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/multivariate_drift_detection.html
- https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html""")

st.header('Data Reconstruction with PCA')
st.write('The first multivariate drift detection method of NannyML is Data Reconstruction with PCA. For a detailed explanation of the method see https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html#how-multiv-drift')

reference_df, analysis_df, _ = nml.load_synthetic_car_loan_dataset()
feature_column_names = [
    'car_value',
    'salary_range',
    'debt_to_income_ratio',
    'loan_length',
    'repaid_loan_on_prev_car',
    'size_of_downpayment',
    'driver_tenure'
]

st.subheader('Reference')
st.write(reference_df.head())

with st.spinner('Calculating drift'):
    calc = nml.DataReconstructionDriftCalculator(
        column_names=feature_column_names,
        timestamp_column_name='timestamp',
        chunk_size=5000
    )
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)

    st.subheader('Drift result')
    st.write(results.filter(period='analysis').to_df())
    st.plotly_chart(results.plot())

from utils import display_source_code
display_source_code()
