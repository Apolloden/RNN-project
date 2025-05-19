# RNN-project
Recurrent Neural Network Project in the Course DD2424 - Deep Learning

<div>
    <a href="https://github.com/Apolloden" target="_blank">David Welzien</a>&emsp;
    <a href="https://github.com/Rick-cmy" target="_blank">Mingyang Chen</a>&emsp;
    <a href="https://github.com/Zhoukkkkkkk" target="_blank">Qianyu Zhou</a>&emsp;
    <a href="https://github.com/yxio11" target="_blank">Yuhui Xue</a>&emsp;
</div>

### Abstract
This project investigates the performance of four recurrent neural network (RNN) architectures, a vanilla RNN, 1‑layer and 2‑layer long short‑term memory (LSTM) networks, and a 2‑layer gated recurrent unit (GRU) model on character level text synthesis. The models were trained on text from Shakespeare and \textit{The Illiad} by Homer, totaling over 2 million characters. Character to number encoding was used and text was represented as vectors of one-hot encoded numbers. A coarse to fine search for hyperparameter was ran to find optimal hidden size, batch size, learning rate, and number of layers. The models were  evaluated on a test set and synthesized text was evaluated on perplexity, spelling accuracy, self‑BLEU, and BERT scores. 

Surprisingly, the more complex architectures LSTM and GRU did not dramatically outperform the simpler RNN. Our findings indicate that both the 1‑layer and 2‑layer LSTM models achieved the lowest test losses (1.90 and 1.89, respectively) compared to the RNN (1.94). However, when inspecting perplexity score the LSTM models both outperformed the other architectures showing better text synthesis ability. Though, from a human perspective, it was difficult to determine which model performed best, as all synthesized texts appeared equally good. In mixed‑domain experiments (75\% Shakespeare, 25\% Iliad), the 2‑layer LSTM generalized better (perplexity 2.51) than the RNN baseline (3.10). A deeper 4‑layer LSTM with dropout underperformed relative to the 2‑layer LSTM, suggesting that dropout was not efficient. Limitations include restricted computational resources, which impacted hyperparameter search, model depth, and training epochs, also the subjectivity of human evaluations. Future work could explore standard word embeddings. 
