"""Layer 3 - NLP / ML body analysis (project centerpiece).

Signals specified in ARCHITECTURE.md section 4:
  l3.zero_width    regex over zero-width and BOM codepoints
  l3.urgency       TF-IDF (word 1-2gram + char 3-5gram) -> LogisticRegression
  l3.perplexity    GPT-2 (124M) mean per-token negative log-likelihood
  l3.burstiness    std-dev of per-sentence PPL and of sentence length
  l3.fusion        LogisticRegression over the four features above

Not implemented - Phase 0 scaffold. No model is loaded or downloaded.
"""
