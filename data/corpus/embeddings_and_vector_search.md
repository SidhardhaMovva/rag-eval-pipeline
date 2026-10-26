# Embeddings, TF-IDF, and Vector Search

An embedding is a numeric vector that represents the meaning of a piece of text, so that texts about similar topics end up with similar vectors. Modern embedding models are neural networks trained to place paraphrases close together and unrelated texts far apart in vector space.

TF-IDF is the classical, non-neural way to build text vectors. Term frequency counts how often a word appears in a document, and inverse document frequency downweights words that appear in almost every document, because common words carry little topical signal. A TF-IDF vector has one dimension per vocabulary term, and rare, distinctive words dominate the representation.

Cosine similarity is the standard way to compare two vectors. It is the dot product of the vectors divided by the product of their lengths, which makes it a measure of direction rather than magnitude: two documents about the same topic score near one even if one is much longer than the other. Cosine similarity ranges from zero for unrelated texts to one for identical direction.

Vector search means finding the stored vectors with the highest cosine similarity to a query vector. Small collections can be searched by brute force, comparing the query against every stored vector. Large collections use approximate nearest neighbour indexes to trade a little accuracy for a large speed gain.

Metadata filtering restricts vector search to a subset of chunks, for example only chunks from one source document or one date range. Filtering should be applied before ranking, otherwise irrelevant chunks from outside the filter can occupy top positions and push valid results out of the top k.
