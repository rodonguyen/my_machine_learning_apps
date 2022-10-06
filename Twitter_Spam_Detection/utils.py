import pickle
import re
from tensorflow import keras 
import numpy

def load_pickle(filename):
    with open(f"{filename}", "rb") as f:
        return pickle.load(f)

def lower_and_seperate(a_string):
    copy = a_string.copy()
    for number, t in enumerate(copy):
        t = re.sub(r"[^a-z0-9#@\*'\":\-\n%,\.;?!]+", " ", str(t).lower())
        t = re.sub(r"#", " # ", t)
        t = re.sub(r"@", " @ ", t)
        t = re.sub(r"\*", " * ", t)
        t = re.sub(r"\'", " ' ", t)
        t = re.sub(r"\"", " \" ", t)
        t = re.sub(r"\:", " : ", t)
        t = re.sub(r"\-", " - ", t)
        t = re.sub(r"\%", " % ", t)
        t = re.sub(r"\,", " , ", t)
        t = re.sub(r"\.", " . ", t)
        t = re.sub(r"\;", " ; ", t)
        t = re.sub(r"\?", " ? ", t)
        t = re.sub(r"\!", " ! ", t)
        copy.iloc[number] = t
    return copy


def tokenise(input, tokenizer):
    input_las = lower_and_seperate(input)   # las = lower and separated
    input_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(input_las):
        input_sequences.append(seq)

    max_tweet_length = 106    # from training
    input_tokenised = numpy.array(keras.preprocessing.sequence.pad_sequences(
        input_sequences, maxlen=max_tweet_length, padding='post'))
    return input_las, input_tokenised

def standardise(input, mu, sigma, columns_to_be_standardised):
    input[columns_to_be_standardised] = (input[columns_to_be_standardised] - mu) / sigma
    return input