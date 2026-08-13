"""Character-level text synthesis with RNN, LSTM and GRU networks."""

from rnn.data import Vocab, build_datasets, load_text
from rnn.models import CharRNN, generate

__all__ = ["CharRNN", "Vocab", "build_datasets", "generate", "load_text"]
__version__ = "1.0.0"
