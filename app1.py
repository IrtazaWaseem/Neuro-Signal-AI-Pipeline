from SignalGenerator import SignalGenerator
from SignalFilter import SignalFilter
from AIFeatureExtractor import AIFeatureExtractor
import streamlit as st
with st.sidebar:
     st.title('Control Center')
     total_points = st.slider('choose total points',1000,5000,3000)
     frequency = st.slider('choose frequency', 1.0, 10.0,2.5)
     amplitude = st.slider('choose amplitude', 5, 20, 10)
     noise_intensity = st.slider('choose noisy intensity',0.0,10.0,3.5)
     epoch_size = st.slider('choose epoch_size',50,200,100)

st.title('Neurological Signal Processing & ML Feature Extractor')
st.subheader('Signal Generator')
s1 = SignalGenerator(total_points)
s1.create_brainwave(frequency,amplitude)
noisy_signal = s1.inject_static(noise_intensity)
st.line_chart(noisy_signal)
st.subheader('Filtering Station')
s4 = SignalFilter(noisy_signal)
s4.hardware_clips(-12,12)
filtered_signal = s4.remove_statistical_outliers()
st.line_chart(filtered_signal)
st.subheader('AI Feature Extractor')
s5 = AIFeatureExtractor(filtered_signal)
s5.segment_into_epochs(epoch_size)
s5.build_training_matrix()
train_set,test_set = s5.split_train_test()
st.metric(label="Training Set Shape", value=str(train_set.shape))

