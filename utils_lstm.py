from tensorflow import keras 
import pandas, tensorflow, utils

def load_LSTM_model():
  max_tweet_length = 106    # from training

  inputA = keras.layers.Input(shape = (max_tweet_length,1), name = 'tokenised_tweets')
  x = keras.layers.LSTM(128, return_sequences=True)(inputA)
  x = keras.layers.LSTM(64)(x)
  x = keras.layers.Dropout(0.2)(x)
  x = keras.layers.Dense(32, activation='relu')(x)

  inputB = keras.layers.Input(shape=(4,1), name='others')
  y = keras.layers.Flatten()(inputB)
  y = keras.layers.Dense(4, activation='relu')(y)

  combined = keras.layers.concatenate([x,y])
  output = keras.layers.Dense(32, activation='relu')(combined)
  output = keras.layers.Dropout(0.2)(output)
  output = keras.layers.Dense(1, activation='sigmoid')(output)
  lstm_model_2input = keras.models.Model([inputA, inputB], output)

  lstm_model_2input.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  lstm_model_2input.load_weights("LSTM_model_2inputs_969.h5")

  return lstm_model_2input


def preprocessing_input(input):
  columns_to_be_standardised = ['following', 'followers', 'actions']
  input_df = pandas.DataFrame(data=input)
  sigma, mu, tokenizer = utils.load_pickle('sigma_train.pkl'), utils.load_pickle('mu_train.pkl'), utils.load_pickle('tokeniser_train.pkl')
  print(sigma, mu)
  # PROCESS 'tweet'
  tweet_las, tweet_tokenised = utils.tokenise(input_df['tweet'], tokenizer)
  tweet_tensor = tensorflow.convert_to_tensor(tweet_tokenised)
  # print(tweet_las)
  # print(tweet_tokenised)
  # print(tweet_tokenised.shape, tweet_tensor.shape)

  # PROCESS 'following' 'followers' 'actions' 'is_retweet'
  others_col_std = utils.standardise(input_df, mu, sigma, columns_to_be_standardised)
  others_col_std.drop(['tweet'], axis=1, inplace=True)
  # print(others_col_std)

  return tweet_tensor, others_col_std