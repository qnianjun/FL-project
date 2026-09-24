# Update-similarity diagnostic

1. **Are Sybil–Sybil updates more similar than honest–honest updates?**

   Unknown from saved data. No full client updates or round-start model
   checkpoints are saved. Replica identities 0, 5, 6 share partition 0,
   poisoned positions and shuffle seeds, so identical nonzero updates are
   expected under deterministic training. This is an implementation expectation,
   not an observed cosine similarity. Unique attacker data remains 12,000 samples.

2. **Is there visible separation across seeds and rounds?**

   No measured evidence is available. For seeds 42, 43, 44 and rounds 1–10,
   honest–honest, attacker–honest and Sybil–Sybil similarity ranges are all
   **unavailable**. This does not show that the distributions overlap or separate.
   Coverage includes clean five-client FL, Backdoor (ratio 0.2), Sybil-Backdoor
   (three attacker identities plus four honest identities), and model-update
   poisoning (scale 10), all with equal-size alpha=0.1 partitions.

   [per_pair.csv](../results/defense_diagnostics/update_similarity/per_pair.csv)
   has headers only because there are no measured pairs.
   [per_round_summary.csv](../results/defense_diagnostics/update_similarity/per_round_summary.csv)
   preserves 360 seed/round/group availability rows;
   [summary.csv](../results/defense_diagnostics/update_similarity/summary.csv)
   contains 12 group availability rows. Blank statistics mean unavailable,
   not zero. Groups with no eligible pairs are marked `not_applicable`.

3. **Do honest Non-IID clients sometimes have very high similarity?**

   Unknown. Their update directions were not recorded. Equal L2 norms do not
   imply similar directions; clean accuracy, ASR and aggregation outcomes also
   cannot recover pairwise angles. No mean, sample SD, minimum or maximum cosine
   can be estimated from those scalars.

4. **Would similarity-based grouping appear plausible?**

   It remains an untested hypothesis. Shared data and shuffle behavior motivate
   checking replica similarity, but there is no observed honest-client comparison.
   Matched clean Sybil runs also duplicate identities and data: high similarity
   alone would not establish malicious behavior. No threshold was selected and
   no new defense is recommended or implemented.

5. **What limitations prevent a conclusion?**

   All 436 checksum entries across the defense pilot and the three relevant
   attack archives passed verification. The inspected NPZ files contain sample
   indices, not model parameters. Saved source, configurations, CSV schemas,
   hashes and the missing-data status are recorded in
   [provenance.json](../results/defense_diagnostics/update_similarity/provenance.json).

   Model-poisoning `updates.csv` records local and submitted norms; the submitted
   norm is after scaling in attacked runs. Defense `aggregation.csv` records
   submitted norms before server clipping and separately labelled post-clipping
   norms. Backdoor and Sybil FedAvg outputs contain performance metrics but no
   client update norms. Scalar norms cannot recover directions, and later-round
   defense trajectories cannot substitute for FedAvg trajectories. Analysis
   stopped at this missing-vector check; no norm distributions were substituted
   for similarity measurements.

   A separately authorized run with additional logging would be required unless
   previously unsupplied vector artifacts become available. Save lossless
   round-start global parameters and every submitted client parameter vector
   **after attack transformation and before server clipping or aggregation**.
   Include each tensor’s name, order, shape and native dtype; compute deltas by
   float64 subtraction of the saved arrays, or save lossless deltas with the
   subtraction precision explicitly recorded. Link every vector to its seed,
   round, condition, method, logical identity, original partition/hash, sample
   count, shuffle seed, source/environment provenance and global-model checksum.
   Save these artifacts in a new directory, with checksums. Keep any post-clipping
   vectors separately labelled and mark zero-norm cosine undefined.

   No training was run and inspected raw-result hashes remained unchanged.
