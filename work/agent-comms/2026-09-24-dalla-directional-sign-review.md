# Dalla sign follow-up: advisory citations and directional explanations

The user agreed that citation mistakes must not reject an otherwise supported
decision. An exact string match is mechanically testable; scientific relevance
is not established by that match. V2 records both separately and does not credit
an inaccurate citation as evidence. Missing/null quotations are tolerated.

Not every reviewed sign was wrong. In the quoted named R13 repair, meal and
production additions and utilization/self-removal subtractions are consistent
with the declared roles. The negative tissue-glucose contribution reflected a
questionable transfer interpretation; pruning removed that term from the final
model. Correct remaining signs do not establish active mechanisms when fitted
gains are near zero, nor do they establish intervention accuracy.

Implementation: `directional_sign_review.py` requests source/sink/transfer/feedback
interpretations, checks internal direction consistency, and obtains a separately
cached semantic assessment using public context only. It locks supported slots
across retries. Unsupported directions remain unrestricted and unchanged after
the bounded budget. Citation issues do not themselves trigger rejection. The
semantic assessment is fallible advice from the same model, not a certified
scientific critic. No benchmark-specific signs or private equations are supplied.

The separate `dalla-sign-repair-2` pilot selects only `full_perturbed_r4` from the
original rescue packet, including its frozen starting vector. At most six LLM
calls (three pairs) precede matched repaired/control rescue and pruning fits.
Original campaigns and current pipeline defaults are preserved. No benchmark
fitting or live LLM call is performed locally; ACES commands are in
`docs/DALLA_SIGN_REPAIR.md`. Intervention quality remains an open empirical question.
