Provides training and testing functions for a part-of-speech identifier.

Usage: call train_POS on a file of line-delimited sentences tokenized into words and respective parts of speech. As an example:
[('Merger', 'NN-HL'), ('proposed', 'VBN-HL')]
would be a valid line in this format.

To use the resulting trained identifier, call test_POS on a list of words of a sentence. The function will return a list of respective parts-of-speech for each word in the sentence.
