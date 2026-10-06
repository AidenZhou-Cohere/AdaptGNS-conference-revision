# Sand CUDA numerical mechanism diagnostic

This bounded same-input diagnostic explains the principal gradient discrepancies in the separate strict CPU/CUDA checks. **Both strict validation failures remain failures.** No scientific model, test result or backend-equivalence claim is produced here.

All three original size-selected training batches, both objectives, initialization and numerical tolerances are unchanged. Four branches compare natural CPU, natural deterministic CUDA, a CUDA own-mask sham, and a CUDA backward-mask intervention using CPU ReLU signs. The intervention changes only backpropagation and is never used for training.

The middle and largest cases have 19 and 17 near-zero ReLU sign disagreements; the smallest has none. Common backward masks remove all 13 previously failing gradient entries and all post-update prediction failures. Exact saved-array audit verifies all 30 NPZ hashes and byte-identical natural-CUDA versus sham gradients, parameters, Adam states, updates and pre/post predictions in every case.

Natural first-Adam updates still fail the prior tolerance in five of six cases, and post-update predictions in three. Maximum update difference is 1.989007e-4. Small NLL retains two update discrepancies (maximum 6.1132014e-6) even without ReLU crossings. Thus the mask intervention explains the dominant observed discrepancy, not every floating-point difference, and does not establish equal long-run training trajectories.

Independent code and scalar reviews support a separately declared **native deterministic CUDA timing measurement only**. The root assessment requires a complete runtime forecast and a new scientific protocol before scientific training. Infrastructure checkpoints may not become scientific models. The original WaterDrop experiment remains unchanged.

The diagnostic took 79.59 seconds. Source, freeze, full scalar comparisons, unsuccessful strict-check references, independent reviews and exact-array hash audit are included. Raw tensor NPZs remain on the research machine; their hashes are recorded. Exact pinned trainer/validation helpers and the deterministic v2 input report are included for audit reproduction. Earlier validation outcomes remain in the sibling `sand_cuda_validation_20261006` family. Real diagnostic reproduction additionally needs the core repository and hash-bound saved training batches. Tiny CPU tests need PyTorch/pytest and the included helper. An initial packaging check found a missing copied helper; that failure is retained and the unchanged dependency is now included.
