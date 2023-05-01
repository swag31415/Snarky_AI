# Snarky AI

To generate the predictions run
```
python project.py
```

This should generate a file with the name `task_A_En_output.csv`

To evaluate these predictions against `task_A_En_test.csv` run
```
python evaluate.py
```

Below is a description of our dataset



## iSarcasmEval Dataset
### The data
A new method of data collection has been introduced that eliminates the need for labeling proxies, such as predefined tags or third-party annotators, for sarcasm labels in texts. The authors themselves provide the sarcasm labels for the texts in two datasets, one in English and one in Arabic. In each dataset, the authors are also asked to rephrase the sarcastic text to convey the same intended message without sarcasm. Linguistic experts further label each text into one of the categories of ironic speech, as defined by Leggitt and Gibbs (2000), including sarcasm, irony, satire, understatement, overstatement, and rhetorical question (for the English dataset only). The Arabic dataset also includes a label for the dialect of the text. As a result, each text in the datasets contains information on its sarcastic nature, a non-sarcastic rephrasing, the category of ironic speech, and, for the Arabic dataset, the dialect label.

### Task Details
Given a text, determine whether it is sarcastic or non-sarcastic

#### Metrics:

F1-score for the sarcastic class. This metric should not be confused with the regular macro-F1. Please use the following code snippet:

```
from sklearn.metrics import f1_score, precision_score, recall_score

f1_sarcastic = f1_score(truths,submitted, average = "binary", pos_label = 1)
```

### Citation
```
@inproceedings{abu-farha-etal-2022-semeval,
    title = "{S}em{E}val-2022 Task 6: i{S}arcasm{E}val, Intended Sarcasm Detection in {E}nglish and {A}rabic",
    author = "Abu Farha, Ibrahim  and
      Oprea, Silviu Vlad  and
      Wilson, Steven  and
      Magdy, Walid",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.111",
    pages = "802--814",
    abstract = "iSarcasmEval is the first shared task to target intended sarcasm detection: the data for this task was provided and labelled by the authors of the texts themselves. Such an approach minimises the downfalls of other methods to collect sarcasm data, which rely on distant supervision or third-party annotations. The shared task contains two languages, English and Arabic, and three subtasks: sarcasm detection, sarcasm category classification, and pairwise sarcasm identification given a sarcastic sentence and its non-sarcastic rephrase. The task received submissions from 60 different teams, with the sarcasm detection task being the most popular. Most of the participating teams utilised pre-trained language models. In this paper, we provide an overview of the task, data, and participating teams.",
}
```
