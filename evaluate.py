# Imports
import pandas as pd
from sklearn.metrics import f1_score

pred = pd.read_csv('task_A_En_output.csv')
test = pd.read_csv('task_A_En_test.csv')

print("F1 score:", f1_score(pred['sarcastic'], test['sarcastic'], average = "binary", pos_label = 1) * 100)