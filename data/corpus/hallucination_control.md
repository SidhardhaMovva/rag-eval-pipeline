# Controlling Hallucination in Question Answering

Hallucination is when a system produces a fluent, confident answer that is not supported by its source material. In retrieval-based question answering, hallucination usually happens in two ways: the retriever returns weak or irrelevant passages and the generator invents an answer anyway, or the generator adds claims that go beyond what the retrieved passages say.

Abstention is the most direct control. The system computes a confidence score from retrieval strength, and when the confidence falls below a threshold the system refuses to answer, saying it does not have enough evidence in the indexed documents. An honest refusal is far better than a fabricated answer, especially for factual questions.

Citations are the second control. When every answer sentence is tied to the chunk it came from, a user can check the claim against the source in seconds. Citation requirements also discipline the generator, because a claim that cannot be attached to a retrieved chunk has nowhere to hide.

Extractive answering is the strictest form of grounding. Instead of generating new text, the system returns sentences copied verbatim from the retrieved chunks. An extractive system cannot invent a fact that is absent from the corpus, although it can still select an irrelevant sentence, which is why sentence selection is scored against the query.

Confidence thresholds should be tuned on a golden set that includes unanswerable questions — questions whose answers are not in the corpus at all. A well-tuned system answers the answerable questions and abstains on the unanswerable ones, and both behaviours are measurable.
