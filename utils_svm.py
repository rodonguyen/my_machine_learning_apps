import pandas, utils


def load_svm_model():
  svm_model = utils.load_pickle('pickles/SVM_model.pkl')
  return svm_model


def preprocess_input(input):
  column_transformer = utils.load_pickle('pickles/SVM_ColumnTransformer_tfidf.pkl')
  input_df = pandas.DataFrame(data=input, columns=['Tweet', 'following', 'followers', 'actions', 'is_retweet'])
  processed_input = column_transformer.transform(input_df)
  return processed_input

svm_model = load_svm_model()
print(svm_model.predict(preprocess_input([['aaa', 1,2,3, 1]])))
