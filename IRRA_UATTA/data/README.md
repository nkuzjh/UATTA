# IRRA Tokenizer Vocabulary

`IRRA_UATTA/utils/simple_tokenizer.py` expects the CLIP tokenizer vocabulary at:

```text
IRRA_UATTA/data/bpe_simple_vocab_16e6.txt.gz
```

Download it from the official OpenAI CLIP repository:

```bash
cd /path/to/OpenSource_Release
mkdir -p IRRA_UATTA/data
curl -L https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
  -o IRRA_UATTA/data/bpe_simple_vocab_16e6.txt.gz
```
