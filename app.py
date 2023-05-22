import streamlit as st
# from utils import PrepProcessor, columns
import utils, utils_lstm, utils_naive_bayes, utils_svm, pandas, numpy


model = 0

st.title('Is it a spam tweet? :robot:')
st.text_input

tweet_content = st.text_input("Tweet Content", 'Big day.  #WeTheNorth #yyz #thesix #sunset #skyline @ The Six https://www.instagram.com/p/BFgrA9gBZay/') 
following = st.number_input("Input Following Number of the author account", 0,10000000)
followers = st.number_input("Input Followers Number", 0,10000000)
actions = st.number_input("Input Actions Number", 0,1000000)
is_retweet = st.selectbox("Is it a Retweet",[0,1])

def predict(): 
  # row = numpy.array([tweet_content,following,follower,actions,is_retweet]) 
  # X = pandas.DataFrame([row], columns = columns)
  # prediction = model.predict(X)

  # Load Model
  lstm_model = utils_lstm.load_LSTM_model()
  nb_model = utils_naive_bayes.load_naive_bayes_model()
  svm_model = utils_svm.load_svm_model()

  # Prepare data
  lstm_tweet_tensor, lstm_others_col_std = utils_lstm.preprocessing_input([[tweet_content, following, followers, actions, is_retweet]])
  nb_input = utils_naive_bayes.preprocess_input([tweet_content])
  svm_input = utils_svm.preprocess_input([[tweet_content, following, followers, actions, is_retweet]])    

  prediction = pandas.DataFrame(
  [
    ['LSTM', (lstm_model.predict([lstm_tweet_tensor, lstm_others_col_std]) >= 0.5)],
    ['Naive Bayes', nb_model.predict(nb_input)[0]],
    ['SVM', svm_model.predict(svm_input)[0]]
  ], 
  columns=['model', 'prediction'])
  
  prediction[prediction == True] = "Spam"
  prediction[prediction == False] = "Quality"

  # if prediction[0] == 1: 
  #   st.success('Passenger Survived :thumbsup:')
  # else: 
  #   st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)