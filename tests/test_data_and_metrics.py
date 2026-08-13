"""Tests for the data pipeline, the model and the evaluation metrics."""

import numpy as np
import pytest
import torch

from rnn.data import Vocab, build_datasets, make_sequences, split_sequences
from rnn.metrics import self_bleu, text_perplexity
from rnn.models import CharRNN, generate, nucleus_probs, parameter_count, training_flops

TEXT = "To be, or not to be, that is the question.\n" * 40
CELL_TYPES = ["rnn", "lstm", "gru"]


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


@pytest.fixture
def tiny_model():
    vocab = Vocab.from_text(TEXT)
    torch.manual_seed(0)
    return vocab, CharRNN(vocab_size=len(vocab), hidden_size=16, cell_type="lstm")


def test_vocab_roundtrip():
    vocab = Vocab.from_text(TEXT)
    assert vocab.decode(vocab.encode(TEXT)) == TEXT
    assert len(vocab) == len(set(TEXT))


def test_vocab_is_sorted_so_runs_are_reproducible():
    assert list(Vocab.from_text("cba").itos) == ["a", "b", "c"]


def test_targets_are_inputs_shifted_by_one():
    ids = np.arange(10, dtype=np.int64)
    x, y = make_sequences(ids, seq_length=3)
    assert x.shape == y.shape == (3, 3)
    assert np.array_equal(x[:, 1:], y[:, :-1])
    assert y[0, 0] == x[0, 0] + 1


def test_sequence_too_long_for_corpus_is_rejected():
    with pytest.raises(ValueError):
        make_sequences(np.arange(4, dtype=np.int64), seq_length=25)


def test_splits_are_contiguous_and_sum_to_the_whole():
    x, y = make_sequences(np.arange(1001, dtype=np.int64), seq_length=10)
    train_ds, val_ds, test_ds = split_sequences(x, y, val_frac=0.15, test_frac=0.15)
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(x)
    assert len(val_ds) == len(test_ds) == 15
    # Validation text follows the training text, with no overlap.
    assert torch.equal(val_ds[0][0], torch.from_numpy(x[len(train_ds)]))


def test_build_datasets_reads_a_file(corpus):
    vocab, train_ds, val_ds, test_ds = build_datasets(corpus, seq_length=25)
    assert len(vocab) == len(set(TEXT))
    assert min(len(train_ds), len(val_ds), len(test_ds)) > 0
    inputs, targets = train_ds[0]
    assert inputs.shape == targets.shape == (25,)
    assert inputs.dtype == torch.int64


def test_carriage_returns_stay_out_of_the_vocabulary(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"to be\r\nor not\r\n" * 30)
    vocab, _, _, _ = build_datasets(path, seq_length=5)
    assert "\r" not in vocab.itos


def test_limit_chars_truncates_the_corpus(corpus):
    _, train_ds, _, _ = build_datasets(corpus, seq_length=10, limit_chars=200)
    assert len(train_ds) == 15  # 19 sequences, minus 2 for validation and 2 for test


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_forward_returns_logits_and_usable_state(cell_type):
    vocab = Vocab.from_text(TEXT)
    model = CharRNN(len(vocab), hidden_size=8, num_layers=2, cell_type=cell_type)
    inputs = torch.randint(len(vocab), (4, 25))

    logits, hidden = model(inputs)
    assert logits.shape == (4, 25, len(vocab))

    # The returned state must be accepted as the state of the next batch.
    logits, _ = model(inputs, hidden)
    assert logits.shape == (4, 25, len(vocab))


def test_unknown_cell_type_is_rejected():
    with pytest.raises(ValueError):
        CharRNN(vocab_size=10, hidden_size=8, cell_type="transformer")


def test_parameter_count_matches_the_gate_count():
    sizes = {c: parameter_count(CharRNN(10, hidden_size=8, cell_type=c)) for c in CELL_TYPES}
    # An LSTM has four gate matrices, a GRU three and a vanilla RNN one, on top
    # of the output layer that all three share.
    output_layer = 8 * 10 + 10
    assert sizes["lstm"] - output_layer == 4 * (sizes["rnn"] - output_layer)
    assert sizes["gru"] - output_layer == 3 * (sizes["rnn"] - output_layer)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_training_flops_scale_with_tokens(cell_type):
    model = CharRNN(65, hidden_size=128, cell_type=cell_type)
    assert training_flops(model, 2000) == 2 * training_flops(model, 1000)
    # Forward and backward, at two FLOPs per multiply-accumulate. The biases are
    # parameters but barely any arithmetic, so the match is approximate.
    assert training_flops(model, 1000) == pytest.approx(6000 * parameter_count(model), rel=0.02)


@pytest.mark.parametrize("top_p", [None, 0.9])
def test_generate_continues_the_prompt(tiny_model, top_p):
    vocab, model = tiny_model
    text = generate(model, vocab, prompt="To be", length=50, temperature=0.5, top_p=top_p)
    assert text.startswith("To be")
    assert len(text) == len("To be") + 50
    assert set(text) <= set(vocab.itos)


def test_generate_leaves_the_model_in_training_mode(tiny_model):
    vocab, model = tiny_model
    model.train()
    generate(model, vocab, prompt="To", length=5)
    assert model.training


def test_nucleus_keeps_the_head_of_the_distribution():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    filtered = nucleus_probs(probs, top_p=0.8)
    assert filtered[2] == 0 and filtered[3] == 0
    assert filtered[0] > 0 and filtered[1] > 0
    assert filtered.sum().item() == pytest.approx(1.0)


def test_nucleus_with_a_tiny_threshold_is_greedy():
    probs = torch.tensor([0.1, 0.7, 0.2])
    filtered = nucleus_probs(probs, top_p=0.01)
    assert filtered.argmax().item() == 1
    assert filtered[1].item() == pytest.approx(1.0)


def test_untrained_perplexity_is_close_to_the_vocabulary_size(tiny_model):
    vocab, model = tiny_model
    perplexity = text_perplexity(model, vocab, "To be, or not to be")
    # An untrained model is roughly uniform over the vocabulary.
    assert 1.0 <= perplexity <= 2 * len(vocab)


def test_self_bleu_is_higher_for_repeated_text():
    repeated = ["the king is dead long live the king"] * 3
    varied = [
        "the king is dead long live the king",
        "a horse a horse my kingdom for a horse",
        "now is the winter of our discontent",
    ]
    assert self_bleu(repeated) > self_bleu(varied)


def test_self_bleu_needs_more_than_one_sample():
    with pytest.raises(ValueError):
        self_bleu(["only one"])


def test_spelling_accuracy_separates_words_from_gibberish():
    pytest.importorskip("spellchecker")
    from rnn.metrics import spelling_accuracy

    assert spelling_accuracy("the king is dead") == 1.0
    assert spelling_accuracy("thh kzng ys dxad") == 0.0
