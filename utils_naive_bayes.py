import pandas, utils


def load_naive_bayes_model():
  nb_model = utils.load_pickle('pickles/NB_model.pkl')
  return nb_model


def preprocess_input(input):
  column_transformer = utils.load_pickle('pickles/NB_ColumnTransformer.pkl')
  input_df = pandas.DataFrame(data=input, columns=['Tweet'])
  processed_input = column_transformer.transform(input_df)
  return processed_input


### Testing
# my_input = preprocess_input(['trump vote 1 com'])
# my_nb_model = load_naive_bayes_model()
# print(my_nb_model.predict(my_input))

