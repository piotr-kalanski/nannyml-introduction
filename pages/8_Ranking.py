import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Ranking')

st.write("""NannyML uses ranking to order columns in univariate drift results.
         The resulting order can help prioritize what to investigate further to fully address any issues with the model being monitored.
         There are currently two ranking methods in NannyML: alert count and correlation ranking.""")

st.write('Source: https://nannyml.readthedocs.io/en/stable/tutorials/ranking.html')

reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
analysis_full_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)
column_names = [
    'car_value', 'salary_range', 'debt_to_income_ratio', 'loan_length', 'repaid_loan_on_prev_car', 'size_of_downpayment', 'driver_tenure', 'y_pred_proba', 'y_pred', 'repaid'
]

st.header('Alert Count Ranking')
st.write('Alert count ranking ranks features according to the number of alerts generated within the ranking period. It is based on the univariate drift results of the features or data columns considered.')

with st.spinner('Calculating drift'):
    univ_calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        treat_as_categorical=['y_pred', 'repaid'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
        chunk_size=5000
    )

    univ_calc.fit(reference_df)
    univariate_results = univ_calc.calculate(analysis_full_df)
    st.subheader('Univariate drift results')
    st.write(univariate_results.filter(period='analysis', column_names=['debt_to_income_ratio']).to_df())

alert_count_ranker = nml.AlertCountRanker()
alert_count_ranked_features = alert_count_ranker.rank(
    univariate_results.filter(methods=['jensen_shannon']),
    only_drifting=False
)
st.subheader('Count ranking results')
st.table(alert_count_ranked_features)

st.header('Correlation Ranking')
st.write('Correlation ranking ranks features according to how much they correlate to absolute changes in the performance metric selected.')

with st.spinner('Calculating CBPE'):
    estimated_calc = nml.CBPE(
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='repaid',
        timestamp_column_name='timestamp',
        metrics=['roc_auc', 'recall'],
        chunk_size=5000,
        problem_type='classification_binary',
    )
    estimated_calc.fit(reference_df)
    estimated_perf_results = estimated_calc.estimate(analysis_full_df)
    st.subheader('Estimated performance (CBPE)')
    st.write(estimated_perf_results.filter(period='analysis').to_df())

with st.spinner('Calculating performance'):
    realized_calc = nml.PerformanceCalculator(
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='repaid',
        timestamp_column_name='timestamp',
        problem_type='classification_binary',
        metrics=['roc_auc', 'recall',],
        chunk_size=5000)
    realized_calc.fit(reference_df)
    realized_perf_results = realized_calc.calculate(analysis_full_df)
    st.subheader('Realized performance')
    st.write(realized_perf_results.filter(period='analysis').to_df())

with st.spinner('Calculating correlation'):
    ranker1 = nml.CorrelationRanker()
    # ranker fits on one metric and reference period data only
    ranker1.fit(estimated_perf_results.filter(period='reference', metrics=['roc_auc']))
    # ranker ranks on one drift method and one performance metric
    correlation_ranked_features1 = ranker1.rank(
        univariate_results.filter(methods=['jensen_shannon']),
        estimated_perf_results.filter(metrics=['roc_auc']),
        only_drifting=False
    )

st.subheader('Correlation rank results')
st.write(correlation_ranked_features1)

from utils import display_source_code
display_source_code()
