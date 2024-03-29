import streamlit as st

import tensorflow as tf
# from keras.layers import TextVectorization
import pandas as pd
st.title('Skimlit')
txt=st.text_area('Text to analyze', '''
    It was the best of times, it was the worst of times, it was
    the age of wisdom, it was the age of foolishness, it was
    the epoch of belief, it was the epoch of incredulity, it
    was the season of Light, it was the season of Darkness, it
    was the spring of hope, it was the winter of despair, (...)
    ''')
model_path = "skimlit_tribrid_model"
print("hi")
# Load saved model



def get_results(abs_text):
  loaded_model = tf.keras.models.load_model(model_path)
  labels = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
  sent_list = abs_text.split(sep='. ')
  sent_list.pop()
  sent_list = [sentence.strip() for sentence in sent_list]
  i = 0
  total_lines = len(sent_list)
  final_list = []
  temp = {}
  for line in sent_list:
    temp['text'] = line
    temp['line_number'] = i
    temp['total_lines'] = total_lines
    i += 1
    final_list.append(temp)
    temp = {}
  df = pd.DataFrame(final_list)
  chars = [" ".join(list(sentence)) for sentence in sent_list]
  line_numbers_one_hot = tf.one_hot(df.line_number.to_numpy(), depth=15)
  lines_total_one_hot = tf.one_hot(df.total_lines.to_numpy(), depth=20)
  preds = tf.argmax(loaded_model.predict(x=(line_numbers_one_hot,
                                    lines_total_one_hot,
                                    tf.constant(sent_list),
                                    tf.constant(chars)), verbose=0),
                    axis=1)
  i = 0
  for i in range(0, total_lines):
    if i!=0 and preds[i]!=preds[i-1]:
      st.text(f'{labels[preds[i]]}: {sent_list[i]}.')
      # print(f'\n\n\033[1m{labels[preds[i]]}:', end=' ')
      # print(f'\033[0m{sent_list[i]}.', end=' ')
    elif i==0:
      st.text(f'{labels[preds[i]]}: {sent_list[i]}.')

    else:
      st.text(f'{labels[preds[i]]}: {sent_list[i]}.')


print("\n\nAI formatted abstract is given below:\n")
if st.button('skimmed'):
  get_results(txt)
  
  
    