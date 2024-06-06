import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Comparing Estimated and Realized Performance')

st.write('Source: https://nannyml.readthedocs.io/en/stable/tutorials/compare_estimated_and_realized_performance.html')

st.header('Example')

with st.spinner('Loading data'):
    reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
    analysis_with_targets = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)

st.subheader('Reference')
st.write(reference_df.head())

st.subheader('Analysis')
st.write(analysis_df.head())

st.subheader('Estimated vs calculated performance')
with st.spinner('Estimating performance'):
    estimator = nml.CBPE(
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='repaid',
        timestamp_column_name='timestamp',
        metrics=['roc_auc', 'f1'],
        chunk_size=5000,
        problem_type='classification_binary',
    )
    estimator.fit(reference_df)
    results = estimator.estimate(analysis_df)

with st.spinner('Calculating performance'):
    calculator = nml.PerformanceCalculator(
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='repaid',
        timestamp_column_name='timestamp',
        metrics=['roc_auc', 'f1'],
        chunk_size=5000,
        problem_type='classification_binary',
    ).fit(reference_df)
    realized_results = calculator.calculate(analysis_with_targets)

# Show comparison plots
st.plotly_chart(results.filter(metrics=['roc_auc']).compare(realized_results.filter(metrics=['roc_auc'])).plot())
st.plotly_chart(results.filter(metrics=['f1']).compare(realized_results.filter(metrics=['f1'])).plot())

from utils import display_source_code
display_source_code()
