# Event Causality Identification

![Event-Causality-Identification/all.png at main · UniqueMR/Event-Causality-Identification (github.com)](https://github.com/UniqueMR/Event-Causality-Identification/blob/main/Architecture-Diagram/all.png)

LSTM and GAT are used to improve the performance of initial event causality identification model. 

All of the three models use Roberta as the pretrained tokenizer to obtain the word embeddings of the inital argument. The base model only use event trigger words to obtain a event-based relation representation. Then it use a MLP to predict whether the argument has a causal relation based on the representation.



The model modified with LSTM use embeddings of both event trigger words and the whole sentence to identify the existence of causal relation. The LSTM is used as a state machine where the final state denotes a knowledge-based relation representation. It accepts word embeddings as input to trigger the state transition. The predicted result is the weighted sum of the sentence-level prediction and the event-level prediction. 



The model modified with GAT also leverages the sentence-level and the event-level information to predict the causal relation. It manages to construct a attention graph which captures the correlation between each pair of word embeddings. The knowledge-level representations denote the contribution of each position to make the final prediction. 

