# A paper about learning to use and place messages

This author-requested revision replaces the stale audit-led abstract with the completed paired experiments. The paper now asks three linked questions: can training make extra messages useful, does residual risk place them well, and do placement gains survive autonomous feedback? The exact paired Goop result becomes Figure 1; native graph construction precedes the controller, and the benefit/cost interpretation follows the empirical puzzle. Main tables preserve every numeric entry and failure meaning with clearer grouping.

- Current source: [revised_manuscript.tex](manuscript/outputs/revised_manuscript.tex), SHA25ff4acafc8dd2fd8879162e322193a130a3749293167697ce4157a7e1e91e05.
- [Research companion](report/revision_report.pdf):23pages, SHA10dcfb78f2ed510a47c04d66ba8aff03a9aa71d6efeb90787930756d4c69b06e. Pages22-23 add a visual synthesis and all-policy observedGoop3D findings; the first21pages retain identical content and120dpi pixels.
- Three independent editorial roles reviewed narrative, scientific meaning and first-figure presentation. Their proposals, issues and corrected-source approvals are preserved under editorial/.
- Native compilation succeeds within the eight-page main-text guard. Native PDF export and final visual page inspection remain pending; the figure PNG is a shared-primitives preview, not an exported manuscript page.

The actual abstract was revised with explicit author authorization. The title is unchanged. Registration comparison and author verification/disclosure/rights/anonymity checks remain in [AUTHOR_HANDOFF](../../../AUTHOR_HANDOFF.md). This public research package and report are author-facing; they are not anonymous submission archives.

No scientific evaluation or completed audit was rerun for this revision. All29main-table mean/SD strings, earlier appendix results, adverse outcomes and null effects remain. The directed-pair notation was clarified, and the correct297WaterDrop observed test histories per endpoint are now explicit. The Goop3D autonomous cohort remains in progress; it is the final experiment, with no new sweeps planned. Current Goop3D numerical claims are observed-history only.

## Reproduction and history

The manuscript subtree includes the single-file native source, builder and current fragment inputs. Run the builder from that subtree's root to reconstruct the source; compile the saved standalone source with the native editor. Preserve the venue style and eight-page guard.

Figure sources use the existing exact source-coordinate records; the reference snapshot is commit323ff7d8c4de15a41d70cdeb3b3c9fdaae1d0593 and its Goop figure predecessors. This package changes visual presentation, not data.

Report builders are archived exact workspace sources. Their expected input is the previous21-page report (SHA bb76664ad0aea0989438af9ad7474f1fd0d109aa6482816e255ef9a741149cbb), available in the preceding Sand report package. To reproduce in a scratch workspace, restore that input plus the source-bound figure and prepared observed statistics at the recorded paths before running the fresh-path builder. Do not run it against the promoted23-page report. v1 and v2 preserve the initial wording and its two reader corrections. Local rendering warnings and the default-Python missing-PIL attempt did not change PDF/scientific content; bundled-Python pixel checks and both final new-page inspections passed.

The prior manuscript, abstract, report and full scientific packages remain unchanged in history. Manifest payload paths and hashes identify exactly what this presentation package copies; large redundant render/log/reference files are catalogued instead of copied.
