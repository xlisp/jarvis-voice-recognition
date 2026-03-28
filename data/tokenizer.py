"""Character-level tokenizer for English ASR."""

from config import TokenizerConfig


class CharTokenizer:
    """Simple character-level tokenizer for English speech recognition.

    Vocabulary: <pad>=0, <sos>=1, <eos>=2, space, a-z, ', <blank>=last
    """

    def __init__(self, config: TokenizerConfig | None = None):
        if config is None:
            config = TokenizerConfig()

        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

        # Build char-to-id mapping: special tokens first, then vocab chars
        self.char2id = {}
        self.id2char = {}

        # Reserve 0=pad, 1=sos, 2=eos
        idx = 3
        for ch in config.vocab:
            self.char2id[ch] = idx
            self.id2char[idx] = ch
            idx += 1

        # Blank token for CTC (last index)
        self.blank_id = idx
        self.vocab_size = idx + 1

        # Add special tokens to id2char for decoding
        self.id2char[0] = ""
        self.id2char[1] = ""
        self.id2char[2] = ""
        self.id2char[self.blank_id] = ""

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids (with <sos> and <eos>)."""
        text = text.lower().strip()
        ids = [self.sos_id]
        for ch in text:
            if ch in self.char2id:
                ids.append(self.char2id[ch])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back to text."""
        chars = []
        for idx in ids:
            if idx in (self.pad_id, self.sos_id, self.eos_id, self.blank_id):
                continue
            if idx in self.id2char:
                chars.append(self.id2char[idx])
        return "".join(chars)

    def ctc_decode(self, ids: list[int]) -> str:
        """CTC decoding: collapse repeated tokens and remove blanks."""
        result = []
        prev = None
        for idx in ids:
            if idx == self.blank_id:
                prev = idx
                continue
            if idx != prev and idx not in (self.pad_id, self.sos_id, self.eos_id):
                if idx in self.id2char:
                    result.append(self.id2char[idx])
            prev = idx
        return "".join(result)
