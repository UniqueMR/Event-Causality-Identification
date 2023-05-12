# Event Causality Identification

![Event-Causality-Identification/all.png at main · UniqueMR/Event-Causality-Identification (github.com)](https://github.com/UniqueMR/Event-Causality-Identification/blob/main/Architecture-Diagram/all.png)

LSTM and GAT are used to improve the performance of the initial event causality identification model. 

All of the three models use Roberta as the pretrained tokenizer to obtain the word embeddings of the inital argument. The base model only use event trigger words to obtain a event-based relation representation. Then it use a MLP to predict whether the argument has a causal relation based on the representation.



The model modified with LSTM use embeddings of both event trigger words and the whole sentence to identify the existence of causal relation. The LSTM is used as a state machine where the final state denotes a knowledge-based relation representation. It accepts word embeddings as input to trigger the state transition. The predicted result is the weighted sum of the sentence-level prediction and the event-level prediction. 



The model modified with GAT also leverages the sentence-level and the event-level information to predict the causal relation. It manages to construct a attention graph which captures the correlation between each pair of word embeddings. The knowledge-level representations denote the contribution of each position to make the final prediction. 

## Requirements 

* torch
* transformers
* scikit-learn

## Usage

To execute the training, validation and testing, use the following command:

```
python main.py --model [MODEL_NAME]
```

There are 3 options for available models: base, lstm, and gat. You should specify the model you want to use.

## Results

The best epoch of all networks are shown as followed. The result is a conclusion of both intra-sentence and cross-sentence performance when the weight of event branch and sentence branch are 0.9 and 0.1, respectively. The network modified with LSTM achieved the best precision rate, while the network modified with GAT achieved the best recall and F1 rate.
| Method |    Precision    |     Recall      |       F1        |
| :----: | :-------------: | :-------------: | :-------------: |
|  Base  |   0.328928047   |   0.482695811   |   0.400324149   |
|  LSTM  | **0.356763926** |   0.489981785   |   0.412893323   |
|  GAT   |   0.347107438   | **0.535519126** | **0.421203438** |

Parameter sweep is executed on GAT method to find out the best weight distribution for event branch and sentence branch. The overall performance achieved the best when the weight of event branch is set to 0.85. 
| Event Branch Weight |    Precision    |     Recall      |       F1        |
| :-----------------: | :-------------: | :-------------: | :-------------: |
|         0.8         | **0.371812081** |   0.504553734   |   0.42812983    |
|        0.85         |   0.356321839   | **0.564663024** | **0.436927414** |
|         0.9         |   0.347107438   |   0.535519126   |   0.421203438   |
|        0.95         |   0.352417303   |   0.504553734   |   0.414981273   |
|          1          |   0.360583942   |   0.449908925   |   0.400324149   |

