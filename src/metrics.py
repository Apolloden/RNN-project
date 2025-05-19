import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from typing import List, Union, Optional, Tuple
import re
from spellchecker import SpellChecker #pip install pyspellchecker


class Perplexity:
    def __init__(self, model, processor, device=None):
        """
        Initialize perplexity calculator

        Args:
            model:     your PyTorch RNN/LSTM language model
            processor: instance of your DataPreprocessing class (one‐hot + char2ind)
            device:    torch.device; defaults to CPU if None
        """
        self.device = device or torch.device('cpu')
        self.model  = model.to(self.device).eval()
        self.processor = processor

    def compute_perplexity(self, text: str) -> float:
        """
        Compute character‐level perplexity for a single string.
        Args:
            text: e.g. "hello world"
        Returns:
            perplexity = exp(avg negative log‐likelihood), the smaller the better prediction
        """
        
        # Prepare for negative log‐likelihood, encoding the text and calculate log probs
        one_hot = self.processor.get_one_hot_encoding(text[:-1])    
        x = torch.tensor(
            one_hot.T,                       
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)                      

        target_ids = [self.processor.char_to_ind[c] for c in text[1:]]  # length T
        y = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)           

        hprev = self.model.init_hidden(
            batch_size=x.size(0),
            hidden_size=self.model.hidden_size
        ).to(self.device)

        # Forward pass to get the hidden state
        with torch.no_grad():
            logits, _ = self.model(x, hprev) 

        # Log‐softmax to get log‐probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Reshaping for easier calculation
        T, K = log_probs.size(1), log_probs.size(2)
        log_probs_flat = log_probs.view(-1, K)    
        targets_flat   = y.view(-1)               

        # Select the correct next‐char log‐prob at each position
        true_log_probs = log_probs_flat[torch.arange(T, device=self.device), targets_flat]

        # Negative log‐likelihood and average
        nll       = -true_log_probs.sum()         
        avg_nll   = nll / T                       

        # Perplexity = exp(NLL)
        ppl = torch.exp(avg_nll)
        return ppl.item()

class Spellpercentage:
    def __init__(self):
         """
         A class to compute the fraction of correctly spelled tokens in a string.
        Args: 
            spell: a dictionary-based spell checker from the `spellchecker` library
            contractions: a compiled regex of common English contractions
        """
         self.spell = SpellChecker()
         # compile a regex of common English contractions, may add more
         self.contractions = re.compile(
            r"^(?:I'll|I've|I'm|I'd|don't|can't|won't|wasn't|hasn't|you're|you've|you'll|you'd|"
            r"she's|it's|that'll|should've|aren't|couldn't|didn't|doesn't|"
            r"hadn't|haven't|isn't|mightn't|mustn't|needn't|here's|shouldn't|"
            r"weren't|wouldn't|let's|he'd|he'll|he's|shan't|she'd|she'll|that's|there's|they'd|"
            r"they'll|they're|they've|we'd|we're|we've|weren't|what's|where's|who'd|who'll|"
            r"who're|who's|who've)$"
        )
        
    def compute_spellpercentage(self, text: str) -> float:
        """
        Compute the fraction of “correctly spelled” tokens in genString,
        using a dictionary + allowed contractions.

        Args:
            genString: the generated string to check
        Returns:
            float: the fraction of correctly spelled tokens
        """
        # clean up stray punctuation
        text = re.sub(r"(?<=[A-Za-z])[”\.\,]", "", text)
        text = re.sub(r"(?=[A-Za-z])”", "", text)
        text = re.sub(r"\s{1}&\s{1}|(?<!\s)\?\s{1}", " ", text)

        words = text.lower().split()
        if not words:
            return 0.0

        correct = 0
        for w in words:
            if w in self.spell or self.contractions.match(w):
                correct += 1

        return correct / len(words)

class SelfBLEU:
    def __init__(self,
                 processor,
                 max_n: int = 4):
        """
        Args:
            processor: DP
            max_n: highest n-gram order to include (BLEU-n)
        """
        self.processor = processor
        self.max_n = max_n
        self.smoother = SmoothingFunction().method1
        # uniform weights, e.g. (0.25,0.25,0.25,0.25) for BLEU-4
        self.weights = tuple(1.0 / max_n for _ in range(max_n))


    def _self_bleu(self, texts: List[str]) -> float:
        """
        Calculate how similar the texts are to each other, using BLEU.
        Args:
            texts: list of strings to compare 
            Returns: BLEU score, the smaller the better diversity
        """
        # fewer than 2 samples = no diversity to measure
        if len(texts) < 2:
            return 0.0

        # split the sentences into single word
        words = [t.split() for t in texts]
        scores = []
        for i, hyp in enumerate(words):
            # use all other samples as references
            refs = words[:i] + words[i+1:]
            score = sentence_bleu(
                refs,
                hyp,
                weights=self.weights,
                smoothing_function=self.smoother
            )
            scores.append(score)
        return sum(scores) / len(scores)
