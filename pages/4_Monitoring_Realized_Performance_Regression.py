import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Monitoring Realized Performance - Regression')

st.write('Source: https://nannyml.readthedocs.io/en/stable/tutorials/performance_calculation/regression_performance_calculation.html')

st.header('Example')

with st.spinner('Loading data'):
    reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_price_dataset()
    analysis_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)

st.subheader('Reference')
st.write(reference_df.head())

st.subheader('Analysis')
st.write(analysis_df.head())

st.subheader('Realized performance')
with st.spinner('Calculating performance'):
    calc = nml.PerformanceCalculator(
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        problem_type='regression',
        metrics=['mae', 'mse', 'rmse'],
        chunk_size=6000)
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)
st.plotly_chart(results.plot())

from utils import display_source_code
display_source_code()
