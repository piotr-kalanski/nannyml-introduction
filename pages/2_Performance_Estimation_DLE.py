import streamlit as st
import nannyml as nml

st.title('Direct Loss Estimation (DLE)')

st.markdown("""
- Used for regression tasks
- Estimates loss function of monitored model
- LGBM is used as an "extra" model
- NannyML supports various regression metrics like MAE, MSE or RMSE
""")
st.write('The idea behind DLE is to train an extra ML model to estimate the value of the loss of the monitored model by doing so, we can be later turn the difference of the estimated and actual loss into performance metric. For clarity we call this model a nanny model and sometimes we refer to the monitored model as a child model.')
st.write('Each prediction of the child model has an error associated with it (the difference between the actual target and the prediction). For both - learning and evaluation purposes this error is modified and it becomes loss (e.g. absolute or squared error for regression tasks). The value of the loss for each prediction of the child model becomes the target for the nanny model.')
st.write('Source: https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#direct-loss-estimation-dle')

st.header('Example')

with st.spinner('Loading data'):
    # Load real-world data:
    reference_df, analysis_df, _ = nml.load_synthetic_car_price_dataset()

st.subheader('Reference')
st.write(reference_df.head())

st.subheader('Analysis')
st.write(analysis_df.head())

st.subheader('Estimated performance')
with st.spinner('Estimating performance'):
    # Choose a chunker or set a chunk size:
    chunk_size = 5000

    # initialize, specify required data columns, fit estimator and estimate:
    estimator = nml.DLE(
        feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count', 'door_count', 'fuel', 'transmission'],
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        metrics=['rmse', 'rmsle'],
        chunk_size=6000,
        tune_hyperparameters=False
    )
    estimator = estimator.fit(reference_df)
    estimated_performance = estimator.estimate(analysis_df)

st.plotly_chart(estimated_performance.plot())

from utils import display_source_code
display_source_code()
