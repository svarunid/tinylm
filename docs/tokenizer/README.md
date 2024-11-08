# Tokenizers

For any neural network to understand **data**, it has to represented in **numbers**. In case of LLMs, they expect the textual data to be in numerical format. They don't see actual texts rather they see a text to number mapping that we construct.

## Word Tokenizer

The simple and most straight forward way to represent text as number is to split the text into individual words and map them to integers (word-based tokenization).

E.g., Text: `"Hello, World! What is 10 + 100?"`; How can we split this into words?

1. `["Hello,", "World!", "What", "is", "10", "+", "100?]`
2. `["Hello,", " ", "World!", " ", "What", " ", "is", " ", "10", " ", "+", " ", "100?"]`
3. `["Hello", ",", " ", "World", "!", "What", " ", "is", " ", "10", " ", "+", " ", "100", "?"]`

Choosing one of the above strategies to split text poses a couple of problems:

- Might lead to large vocabulary sizes when extending multi-lingually.
- Not possible to learn character level information.

There are many other problems related to linguistics and computation, making word based tokenizers obsolete.

## Character Tokenizer

Splitting text into individual characters is one simple yet great solution. The above text is tokenized into: `['H', 'e', 'l', 'l', 'o', '!', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!', ' ', 'W', 'h', 'a', 't', ' ', 'i', 's', ' ', '1', '0', ' ', '+', ' ', '1', '0', '0', '?']`.

Character-based tokenization worked well with various neural network architectures and hence was used in many popular works. Yet, it posed a few important limitations:

- The model's context is limited and the information the model sees before predicting the next token is also limited resulting in poor quality predictions.
- Increasing the context is computationally expensive at both during inference & training.
- Model demands more data to capture semantic and linguistic features of the text.

## Sub-word Tokenizer (Byte-Pair Encoding)

In order to keep the vocabulary to minimal and also capture character level dependencies to an extent, sub-word tokenizers split text into words and in turn split them into different sub-word which are then represented as individual tokens. One such algorithm is **Byte-Pair Encoding** that works on bytes as opposed to character. **Byte** is used to represent an individual character natively on computers. Hence, the algorithm works on individual bytes.

An example of how our text might be tokenized is: `["Hello", ",", " World", "!"]`.

**Note**: This is just a naive introduction to tokenizers. To learn in-depth about tokenizers, check out [Andrej Karpathy](https://x.com/karpathy)'s [lecture](https://www.youtube.com/watch?v=zduSFxRajkE&t=6453s).
