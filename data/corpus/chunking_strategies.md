# Chunking Strategies for Retrieval

Chunking is the step that splits long documents into smaller pieces before they are embedded and indexed. Chunk size matters because retrieval operates on chunks, not whole documents: a chunk that is too large dilutes its embedding with unrelated topics, while a chunk that is too small loses the context needed to answer a question.

Fixed-size chunking slices text into windows of a set number of characters or tokens, often with a small overlap between consecutive windows. Overlap helps because a fact that falls on a boundary still appears whole in at least one chunk. The weakness of fixed-size chunking is that it is blind to structure, so it happily splits a sentence or a paragraph in the middle.

Recursive chunking is separator-aware. It tries to split on paragraph breaks first, then on line breaks, then on sentence endings, and only falls back to splitting inside a sentence when a piece is still too large. Recursive chunking keeps paragraphs and sentences intact whenever possible, which usually preserves meaning better than fixed windows.

Semantic chunking groups sentences by topical similarity. Adjacent sentences whose embeddings are similar stay in the same chunk, and a new chunk begins when similarity drops, signalling a topic shift. Semantic chunking produces coherent chunks but costs more computation and depends on the quality of the sentence embeddings.

There is no universally best chunking strategy. The right choice depends on the document structure and the questions being asked, which is why chunkers should be compared empirically with retrieval metrics on a labeled golden set rather than chosen by intuition.
