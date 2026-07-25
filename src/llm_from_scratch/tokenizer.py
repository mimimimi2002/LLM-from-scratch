import re

UNK = "<|unk|>"
EOT = "<|endoftext|>"


class SimpleTokenizer:
    def __init__(self, vocab: list[str]):
        self.word_to_token = {word: token for token, word in enumerate(vocab)}
        self.token_to_word = {token: word for token, word in enumerate(vocab)}

    @staticmethod
    def create_vocabulary(file_name):
        try:
            with open(file_name, "r") as f:
                raw_text = f.read()
        except OSError as e:
            raise OSError(f"cannot read vocabulary from file: {file_name}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"cannot read {file_name} as UTF-8") from e

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [s.strip() for s in preprocessed if s.strip()]
        vocab = sorted(set(preprocessed))
        vocab.extend([EOT, UNK])
        return vocab

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [s.strip() for s in preprocessed if s.strip()]
        preprocessed = [s if s in self.word_to_token else UNK for s in preprocessed]
        return [self.word_to_token[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.token_to_word[id] for id in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
