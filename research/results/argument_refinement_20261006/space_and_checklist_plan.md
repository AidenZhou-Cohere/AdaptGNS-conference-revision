# Main-text space and current checklist wording

Proposal only. Manuscript sources, registered title/abstract, all measurements, table contents and existing labels remain unchanged. This supplements the bounded argument review; it uses no pending experimental outcome.

## Recommended space plan

Move the complete compact-study block out of the main text, rather than compressing its tables or removing unfavorable results. In the current `work/manuscript_body.tex`, the exact block begins at `\subsection{Controlled WaterDrop pilot}` (line 143) and ends immediately before `\section{Limitations}` (line 175). It contains the pilot protocol, the `% PILOT_RESULTS_INSERT` marker, dense-benefit diagnostic, physical-control synopsis, benefit-head analysis/equation and `% ROLLOUT_RESULTS_INSERT` marker. The assembled equivalent is the complete block from `\subsection{Controlled WaterDrop pilot}` to the next `\section{Limitations}` (currently lines 830–924).

Place that exact block in the supplement under:

```tex
\section{Compact WaterDrop study and exploratory follow-ups}
\label{sec:compact-study}
```

A natural insertion point is after the Additional Derivations section and its three insert markers, immediately before `\section{Fixed original-architecture extension}`. Retain every original subsection and paragraph without editing. The two table insert files are not changed: `work/pilot_results.tex` and `work/rollout_results.tex`. `work/build_manuscript.py` replaces the markers globally, so moving them does not require a builder change. There must still be exactly one copy of each marker and exactly one occurrence of each table label (`tab:pilot`, `tab:rollout`) in the assembled output.

The proposed appendix fragment is saved separately as `compact_study_relocation_candidate.tex`. Its content after the new section/label header is byte-identical to the extracted body block. This is a relocation candidate, not an edited manuscript or a standalone LaTeX document.

Replace the main-text block with the following synopsis, saved separately as `compact_study_synopsis_candidate.tex`:

```tex
\subsection{Compact-study context}
A separate five-seed compact study compares corrected NLL, beta-NLL and
faithful regression on a small WaterDrop convenience subset, with random
graph expansion during training. Its architecture, training budget, noise
and graph exposure differ from the full-model study, so cross-study changes
do not isolate any one of these factors. Residual ranking is stronger than
dense-benefit ranking, but policy ordering varies with objective and split.
Exploratory benefit-head and physical controls do not establish a general
advantage. Autonomous outcomes further separate short-horizon accuracy from
full-horizon failure. Appendix~\ref{sec:compact-study} retains the complete
methods, all one-step measurements and all rollout outcomes, including
Tables~\ref{tab:pilot} and~\ref{tab:rollout}; the physical diagnostics remain
in Appendices~\ref{sec:physical-rank} and~\ref{sec:physical-allocation}.
```

This removes two full-width tables, four detailed subsections and a displayed equation from the main text while retaining their contents verbatim in the same paper. It should release substantially more space than several paragraph-level cuts. A rough planning allowance is around one to two main-text pages; this is not a measured pagination result. Only the regenerated native PDF and its existing eight-page assertion can establish the actual capacity. Do not reduce type size or margins to force the result.

Use the released space in this order, conditional on complete audited results:

1. Native autonomous control: one concise table or panel with all five policies, both objectives, explicit failures and a small set of full-horizon/physical/time findings. Keep the original no-loop study in its own table and family.
2. Paired faithful 110k continuation: make the five absolute policy outcomes and the primary risk-minus-random interaction legible. Show all three seed effects or retain them immediately adjacent in the appendix, and distinguish observed histories from autonomous rollouts. Null full-horizon values remain null.
3. A short interpretation linking the observed result to graph exposure and deployment. Do not spend the recovered space on every audit counter or another general limitations paragraph.

Do not reserve prose that assumes an improvement. The synopsis and relocation can be made before the pending experiments finish. The new results sections require the frozen analysis and independent audits first.

## Checklist explanation replacements

Keep all official question text verbatim and retain every current bracketed answer, including the conservative `[No]` values. The following replacements affect only explanations after `[No]`; no new completed experiment is implied.

**Anonymized source code question, body line 218:**

```tex
[No] Tested source, dependency specifications and compact result records
cover the new studies. A complete anonymized submission package and its
accessible release have not yet been verified.
```

This avoids reducing the current source/results work to “a local code package,” while preserving the actual anonymity/release blocker.

**Code, data and instructions for empirical results, body line 230:**

```tex
[No] Scripts, fixed protocols and measurement artifacts cover the
historical-array reanalysis, compact pilot, fixed full-model study and
completed exploratory controls. A complete anonymized reproduction package
has not yet been verified; the original historical checkpoints and complete
historical training/evaluation provenance remain unavailable.
```

The qualifier “completed” excludes the running native cohort and any pending continuation result. Raw checkpoints/numeric archives existing locally is not equivalent to a verified public reproduction package.

**All training details question, body line 231:**

```tex
[No] The compact-pilot, benefit-head and fixed full-model protocols specify
their data splits, architectures, objectives, noise, hyperparameters and
checkpoint-selection rules. The exploratory controls have explicit input
and evaluation protocols. Exact historical training and validation-selection
details remain unverified.
```

This distinguishes new training recipes, training-only fitted benefit heads and inference-only controls, rather than describing all follow-ups as training runs.

**New assets question, body line 240:**

```tex
[No] New source, fitted coefficients and compact measurement records are
preserved alongside local checkpoints and numeric archives. A complete
anonymized submission archive or accessible release has not yet been
verified.
```

The existing license explanation and remaining `[Yes]`/`[Not Applicable]` explanations need no change in this batch. Do not claim that a repository push resolves anonymity, raw-asset availability or historical provenance.

## Integration verification

After root integrates the proposal: compare the relocated raw block byte-for-byte with the candidate; compare every existing table by label rather than table number; verify each label and insert marker is present once; verify all existing numerical content persists in the assembled source; keep title and abstract bytes unchanged; run the builder and native compilation with the eight-page assertion; inspect page breaks and the moved appendix tables. Table/section numbers may change automatically because of relocation, but labels and contents must not.
