# Retrieval-Augmented Generation Fundamentals

Retrieval-augmented generation, usually shortened to RAG, is a pattern for answering questions with a language model by first retrieving relevant passages from a document collection and then giving those passages to the model as context. The model generates its answer grounded in the retrieved context instead of relying only on what it memorized during training.

A RAG system has two phases. The indexing phase runs offline: documents are loaded, split into chunks, converted into embeddings, and stored in a vector index. The query phase runs online: a user question is embedded, the most similar chunks are retrieved, and the question plus the retrieved chunks are assembled into a prompt for generation.

Retrieval quality bounds generation quality. If the retriever returns irrelevant chunks, even a strong generator will produce a poor or misleading answer, because the model can only ground its response in the context it is given. This is why RAG engineers say garbage retrieval in means garbage answers out.

Grounding is the central benefit of retrieval-augmented generation. Because the answer is based on retrieved passages, the system can show citations that point back to the source documents, and users can verify each claim against the original text. Grounding also makes the knowledge updatable: adding a new document to the index makes it answerable immediately, without retraining the model.

RAG is commonly used for question answering over internal documentation, support knowledge bases, research papers, and policy manuals. It works best when the answers exist verbatim or near-verbatim in the source documents and when the collection is too large or changes too often to fit into a prompt.
