# Reference papers

The Parakeet Deep Dive cites these papers throughout.
They are **not redistributed** in this repository — third-party PDFs are
`.gitignore`d to respect publisher copyright.

To read offline, download each from the link below into this directory.
The filenames in the table are the names the deep-dive prose refers to, so if
you keep that convention every inline `papers/NN_….pdf` reference resolves.

| # | Suggested filename | Citation | Link |
|---|---|---|---|
| 01 | `01_FastConformer_2305.05084.pdf` | Rekesh et al., *Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition*, 2023 | https://arxiv.org/abs/2305.05084 |
| 02 | `02_TDT_2304.06795.pdf` | Xu et al., *Efficient Sequence Transduction by Jointly Predicting Tokens and Durations* (TDT), ICML 2023 | https://arxiv.org/abs/2304.06795 |
| 03 | `03_Conformer_2005.08100.pdf` | Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020 | https://arxiv.org/abs/2005.08100 |
| 04 | `04_RNNT_Graves2012_1211.3711.pdf` | Graves, *Sequence Transduction with Recurrent Neural Networks* (RNN-T), 2012 | https://arxiv.org/abs/1211.3711 |
| 05 | `05_CTC_Graves2006.pdf` | Graves, Fernández, Gomez, Schmidhuber, *Connectionist Temporal Classification*, ICML 2006 | https://www.cs.toronto.edu/~graves/icml_2006.pdf |
| 06 | `06_TransformerXL_1901.02860.pdf` | Dai et al., *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*, ACL 2019 | https://arxiv.org/abs/1901.02860 |
| 07 | `07_AttentionIsAllYouNeed_1706.03762.pdf` | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 | https://arxiv.org/abs/1706.03762 |
| 08 | `08_SpecAugment_1904.08779.pdf` | Park et al., *SpecAugment*, Interspeech 2019 | https://arxiv.org/abs/1904.08779 |
| 09 | `09_Longformer_2004.05150.pdf` | Beltagy, Peters, Cohan, *Longformer: The Long-Document Transformer*, 2020 | https://arxiv.org/abs/2004.05150 |
| 10 | `10_Macaron_1906.02762.pdf` | Lu et al., *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View* (Macaron), 2019 | https://arxiv.org/abs/1906.02762 |
| 11 | `11_LongFormASR_2309.09950.pdf` | Koluguri et al., *Investigating End-to-End ASR Architectures for Long Form Audio Transcription*, ICASSP 2024 | https://arxiv.org/abs/2309.09950 |
| 12 | `12_ParakeetTDTv3_CanaryV2_2509.14128.pdf` | Sekoyan et al., *Parakeet-TDT v3 / Canary-1B v2 training recipe*, 2025 | https://arxiv.org/abs/2509.14128 |
| 13 | `13_StreamingRNNT_Mobile_1811.06621.pdf` | He et al., *Streaming End-to-End Speech Recognition for Mobile Devices*, ICASSP 2019 | https://arxiv.org/abs/1811.06621 |
| 14 | `14_MultiBlankTransducer_2211.03541.pdf` | Huang et al., *Multi-blank Transducers for Speech Recognition*, 2022 | https://arxiv.org/abs/2211.03541 |
| 15 | `16_DeepSpeech2_1512.02595.pdf` | Amodei et al., *Deep Speech 2: End-to-End Speech Recognition in English and Mandarin*, 2015 | https://arxiv.org/abs/1512.02595 |

## Quick download

With `arxiv` IDs, you can fetch them all at once (skips #05, which is publisher-hosted):

```bash
cd papers
for id in 2305.05084 2304.06795 2005.08100 1211.3711 1901.02860 \
          1706.03762 1904.08779 2004.05150 1906.02762 2309.09950 \
          2509.14128 1811.06621 2211.03541 1512.02595; do
  curl -L -o "${id}.pdf" "https://arxiv.org/pdf/${id}"
done
curl -L -o "05_CTC_Graves2006.pdf" "https://www.cs.toronto.edu/~graves/icml_2006.pdf"
```
