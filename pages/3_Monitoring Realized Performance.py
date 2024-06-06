import streamlit as st
import nannyml as nml

st.title('Monitoring Realized Performance')
st.write('The realized performance of a machine learning model is typically a good proxy for the business impact of the model. A significant drop in performance normally means a lot of value generated by the model is at risk, so close monitoring and quick resolution of issues are essential.')

st.subheader('Estimated vs realized performance')
col1, col2 = st.columns(2)
with col1:
    st.markdown("""Estimated performance:
- measures how well model is expected to perform
- determined using estimators like CBPE, and DLE
- estimated when ground truth is not available""")

with col2:
    st.markdown("""Realized performance:
- represents measured performance
- determined using performance calculator
- calculated when ground truth is available""")

st.write('Source: https://nannyml.readthedocs.io/en/stable/tutorials/performance_calculation.html')

st.header('Example')

with st.spinner('Loading data'):
    reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
    analysis_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)

st.subheader('Reference')
st.write(reference_df.head())

st.subheader('Analysis')
st.write(analysis_df.head())

st.subheader('Realized performance')
with st.spinner('Calculating performance'):
    calc = nml.PerformanceCalculator(
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='repaid',
        timestamp_column_name='timestamp',
        problem_type='classification_binary',
        metrics=['roc_auc', 'f1', 'precision', 'recall'],
        chunk_size=5000)
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)
st.plotly_chart(results.plot())

st.subheader('Estimated business value')
with st.spinner('Calculating business value'):
    calc = nml.PerformanceCalculator(
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='repaid',
        timestamp_column_name='timestamp',
        problem_type='classification_binary',
        metrics=['business_value'],
        # [value_of_TN, value_of_FP], [value_of_FN, value_of_TP]]
        business_value_matrix=[[0, -200], [-100, 1000]],
    )
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)
st.plotly_chart(results.plot())

from utils import display_source_code
display_source_code()
