# Evaluating Retrieval with Recall, Precision, and MRR

A golden set is a labeled collection of queries where each query is paired with the document or chunk that should be retrieved for it. Evaluation means running every golden query through the retriever and comparing what comes back against the labels. Without a golden set, claims about retrieval quality are just opinions.

Recall at k measures the fraction of relevant items that appear in the top k results. When each query has exactly one relevant document, recall at k is the share of queries whose correct document is somewhere in the first k results. Recall at five is a common headline metric because users and generators rarely look beyond the first few chunks.

Hit rate is closely related: it is the fraction of queries for which at least one relevant item appears in the top k. With a single relevant item per query, hit rate and recall at k are the same number.

Mean reciprocal rank, abbreviated MRR, rewards putting the correct result early. For each query, the reciprocal rank is one divided by the rank of the first relevant result, so a correct result at rank one scores 1.0, at rank two scores 0.5, and at rank three scores about 0.33. MRR is the average of those values across all queries, and it falls to zero for a query when the relevant item is not retrieved at all.

Precision at k measures what fraction of the top k results are relevant. Precision matters when the generator's context window is limited, because irrelevant chunks waste space and can distract the model.

Evaluation should be run as a regression check whenever the chunker, the embedding model, or the corpus changes, so that a change which quietly hurts retrieval quality is caught before users notice.
