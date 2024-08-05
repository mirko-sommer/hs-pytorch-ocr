import collections
import torch


class CTCLabelEncoder:
    """Converts between string labels and their corresponding indices for CTC.

    Args:
        alphabet (str): The set of possible characters.
        ignore_case (bool, default=True): Whether to ignore case sensitivity.
    """

    def __init__(self, alphabet: str, ignore_case: bool = True):
        self.ignore_case = ignore_case
        alphabet = alphabet.lower() if ignore_case else alphabet
        self.alphabet = alphabet + '-'  # Add '-' for 'blank' index

        self.char2idx = {char: i + 1 for i, char in enumerate(alphabet)}
        self.idx2char = {i: char for char, i in self.char2idx.items()}

    def encode_texts(self, texts):
        """Encodes text or batch of texts into indices.

        Args:
            texts (str or list of str): Input text(s).

        Returns:
            torch.IntTensor: Encoded texts.
            torch.IntTensor: Length of each text in batch.
        """
        if isinstance(texts, str):
            indices = [self.char2idx[char.lower() if self.ignore_case else char] for char in texts]
            lengths = [len(indices)]
        elif isinstance(texts, collections.abc.Iterable):
            lengths = [len(s) for s in texts]
            concatenated_text = ''.join(texts)
            indices, _ = self.encode_texts(concatenated_text)
            indices = list(indices)
        else:
            raise ValueError("Input must be a string or an iterable of strings.")

        return torch.IntTensor(indices), torch.IntTensor(lengths)

    def decode_indices(self, encoded_texts, lengths, raw=False):
        """Decodes encoded texts back into strings.

        Args:
            encoded_texts (torch.IntTensor): Encoded texts.
            lengths (torch.IntTensor): Length of each text.
            raw (bool, default=False): Whether to return raw output without blank removal.

        Returns:
            str or list of str: Decoded texts.
        """
        if lengths.numel() == 1:
            length = lengths.item()
            assert encoded_texts.numel() == length, f"Mismatch: {encoded_texts.numel()} vs {length}"
            if raw:
                return ''.join([self.alphabet[i - 1] for i in encoded_texts])
            return ''.join([self.alphabet[i - 1] for i in encoded_texts if i != 0 and (i != encoded_texts[i - 1] or i == 0)])
        
        assert encoded_texts.numel() == lengths.sum().item(), f"Mismatch: {encoded_texts.numel()} vs {lengths.sum().item()}"
        texts, index = [], 0
        for length in lengths:
            text = self.decode_indices(encoded_texts[index:index + length], torch.IntTensor([length]), raw=raw)
            texts.append(text)
            index += length

        return texts


def decode_logits(logits: torch.Tensor, label_encoder: CTCLabelEncoder) -> str:
    """Decodes model logits into text.

    Args:
        logits (torch.Tensor): Model predictions.
        label_encoder (CTCLabelEncoder): The label encoder instance.

    Returns:
        str: Decoded text.
    """
    tokens = logits.softmax(dim=2).argmax(dim=2).squeeze(dim=1).cpu().numpy()
    text = ''.join([label_encoder.idx2char.get(token, '-') for token in tokens])
    return ''.join(char for batch_token in text.split('-') for idx, char in enumerate(batch_token) if char != batch_token[idx - 1] or len(batch_token) == 1)
