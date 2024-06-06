import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Data Quality Checks')
st.write('NannyML supports testing the data quality of your data. You can do this by monitoring missing values on all available columns and unseen values on categorical columns.')

st.write('Source: https://nannyml.readthedocs.io/en/stable/tutorials/data_quality.html')

st.header('Missing Values Detection')
st.write("""NannyML’s approach to missing values detection is quite straightforward.
For each chunk NannyML calculates the number of missing values. There is an option, called normalize, to convert the count of values to a relative ratio if needed.
The resulting values from the reference data chunks are used to calculate the alert thresholds.
The missing values results from the analysis chunks are compared against those thresholds and generate alerts if applicable.""")

reference_df, analysis_df, analysis_targets_df = nml.load_titanic_dataset()
st.subheader("Reference data")
st.write(reference_df.head())

feature_column_names = [
    'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',
]
with st.spinner('Calculating missing values'):
    calc = nml.MissingValuesCalculator(
        column_names=feature_column_names,
    )
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)
    st.subheader("Missing results")
    st.write(results.filter(period='all').to_df())

st.subheader("Missing rows for each column")
for column_name in results.column_names:
    st.plotly_chart(results.filter(column_names=column_name).plot())

st.header('Unseen Values Detection')
st.write("""NannyML defines unseen values as categorical feature values that are not present in the reference period.
NannyML’s approach to unseen values detection is simple. The reference period is used to create a set of expected values for each categorical feature.
For each chunk in the analysis period NannyML calculates the number of unseen values.
There is an option, called normalize, to convert the count of values to a relative ratio if needed.
If unseen values are detected in a chunk, an alert is raised for the relevant feature.""")

reference_df, analysis_df, analysis_targets_df = nml.load_titanic_dataset()
st.subheader("Reference data")
st.write(reference_df.head())

with st.spinner('Calculating unseen values'):
    feature_column_names = [
        'Sex', 'Ticket', 'Cabin', 'Embarked',
    ]
    calc = nml.UnseenValuesCalculator(
        column_names=feature_column_names,
    )

    calc.fit(reference_df)
    results = calc.calculate(analysis_df)
    st.subheader("Unseen results")
    st.write((results.filter(period='all').to_df()))

st.subheader("Unseen values for each column")
for column_name in results.column_names:
    st.plotly_chart(results.filter(column_names=column_name).plot())

from utils import display_source_code
display_source_code()
