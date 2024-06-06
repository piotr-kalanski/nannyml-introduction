import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Summary Statistics')

st.markdown("""You can use NannyML to calculate summary statistcs of your data. There are five summary statistics available:
- Summation
- Average
- Standard Deviation
- Median
- Row Count
You can use the summary statistics to both perform simple drift checks as well as perform data quality checks depending on your particular use case.""")

st.write('Source: https://nannyml.readthedocs.io/en/stable/tutorials/summary_stats.html')

st.header('Example')

reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
st.subheader('Reference')
st.write(reference_df.head())

feature_column_names = [
    'car_value', 'debt_to_income_ratio', 'driver_tenure'
]
calc = nml.SummaryStatsSumCalculator(
    column_names=feature_column_names,
)

calc.fit(reference_df)
results = calc.calculate(analysis_df)
st.subheader('Summary results')
st.write(results.filter(period='all').to_df())

st.subheader('Summary results for each column')
for column_name in results.column_names:
    st.plotly_chart(results.filter(column_names=column_name).plot())

from utils import display_source_code
display_source_code()
