import streamlit as st
import nannyml as nml

st.set_page_config(layout="wide")
st.title('Univariate Drift Detection')

st.write('Univariate Drift Detection looks at each feature individually and checks whether its distribution has changed compared to reference data. There are many ways to compare two data samples and measure their similarity. NannyML provides several drift detection methods so that users can choose the one that suits their data best or the one they are familiar with. Additionally, more than one method can be used together to gain different perspectives on how the distribution of your data is changing.')

st.markdown("""Supported methods:
- Jensen-Shannen distance - both categorical and continuous
- Hellinger - categorical and continuous
- Wasserstein - only continuous
- Kolgomorov-Smirnov - only continuous
- L-infinity - only categorical
- Chi2 - only categorical""")

st.markdown("""Source:
- https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html
- https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/univariate_drift_detection.html""")

st.header('Example')

reference_df, analysis_df, _ = nml.load_synthetic_car_loan_dataset()
column_names = ['car_value', 'salary_range', 'debt_to_income_ratio', 'loan_length']

st.subheader('Reference')
st.write(reference_df.head())

st.subheader('Analysis')
st.write(analysis_df.head())

with st.spinner('Calculating drift'):
    calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        treat_as_categorical=['y_pred'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
    )
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)

    st.subheader('Plots')
    st.plotly_chart(results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='drift'))
    st.plotly_chart(results.filter(column_names=results.categorical_column_names, methods=['chi2']).plot(kind='drift'))
    st.plotly_chart(results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='distribution'))
    st.plotly_chart(results.filter(column_names=results.categorical_column_names, methods=['chi2']).plot(kind='distribution'))

from utils import display_source_code
display_source_code()
