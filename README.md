# RFD4-Proteina Design Pipeline (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amiralito/RFD4_Proteina_Colab/blob/main/RFD4_Proteina_design_pipeline.ipynb)

Binder and motif design on Google Colab with
[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) (Baker Lab / IPD × NVIDIA
GenAIR) — a flow-matching model that **co-designs sequence and all-atom structure** in one shared
chemical and conformational latent space covering standard and modified amino acids, sugars, DNA
and RNA. One checkpoint serves binders, motif scaffolding, enzyme active sites, antibodies,
partial diffusion and inverse folding, driven here by form fields with a raw-YAML escape hatch.

Its "thinking" budget (`recycle_strategy`) is a recurrent refinement schedule: more trunk passes
buys a higher pass rate at lower throughput, selectable per run.

> **This notebook does not validate designs.** Reward-guided search and every `*_refolding`
> metric need an AlphaFold 2 / AlphaFold 3 / OpenFold3 runtime shipped as an `apptainer` image,
> which Colab cannot run. Generation, the oracle-free metrics and partial diffusion all work here;
> the refold-and-compare step belongs in the
> [AF3](https://github.com/amiralito/Protenix_Colab)/Protenix/OpenFold3 notebooks, fed by the
> FASTA this one exports. Treat that step as mandatory.

## Features

- **Two ways to configure a run** — a form-field builder for binders, motif scaffolds, partial
  diffusion and inverse folding, or paste a raw condition spec for full control (`C_RAS` burial,
  `C_HB` hydrogen-bond roles, `C_PISTACK`, `C_CATH` topology, `C_CRDGRP` internal-coordinate
  groups, per-token sequence probabilities, several designs in one file, `global_args`).
- **The target is conditioned for you** — the form emits the target's full sequence plus its
  backbone as one rigid internal-coordinate group, as the shipped PD-L1 spec does. Anything you
  do not mention is not provided to the model, and an unconditioned target gets **redesigned**
  alongside the binder.
- **Author → label numbering conversion** — contigs and selections index `label_seq_id`, not the
  author numbering on a PDB entry. The notebook prints both, reports the offset, and flags when
  it is not constant.
- **Atom-level hotspots** — hotspots default to two distal sidechain atoms per residue
  (`TYR → [CG,OH]`, `MET → [CG,SD]`), so you specify which *face* to engage rather than letting
  the binder satisfy a hotspot by packing against backbone.
- **Epitope diversification** — optional per-design hotspot patch sampling
  (`SampleNearbyHotspots`), so `num_replicates` designs explore a surface instead of all
  attacking the same three atoms.
- **Auto-selected metrics** — reads your spec and enables every oracle-free metric its conditions
  make meaningful: clashes, bond geometry, secondary structure, sequence adherence, hotspot
  obedience, motif RMSD.
- **Google Drive integration** — persist a whole run to Drive, and cache the model bundle so it
  downloads **once** and is reused by every future session.
- **Timestamped, named runs** — each run writes to its own `runs/<date-time>_<name>/` folder, so
  runs never overwrite each other, with `RESUME_RUN` to reattach after a disconnect.
- **Code-free UI** — every step is a form; code is hidden by default (toggle "Show code").

## Pipeline

```
RFD4-Proteina                  →   refold + compare        →   order
(sequence + all-atom               (AF3 / Protenix /            (wet lab)
 co-design, one model,              OpenFold3 — NOT in
 one latent space)                  Colab)
```

A single model replaces the generate → inverse-fold → validate chain of earlier pipelines: the
sequence comes out with the structure, so the only external step is independent refolding.

## Requirements

- Google Colab with a **GPU runtime** (`Runtime → Change runtime type → GPU`).
- **A100 (40 GB) is the comfortable case.** An L4 (24 GB) is fine for a cropped target plus a
  60–120 aa binder. A **T4 (16 GB) is the floor** and a slow one — Turing has no bfloat16 and no
  fused cuEquivariance kernels, so the notebook drops to fp16 and the torch backends
  automatically.
- ~7 GB for the environment (mostly PyTorch and CUDA libraries) and ~1.7 GB for the model bundle,
  cached on Drive across sessions.
- Python 3.12 — the wheel pins `>=3.12,<3.13`, so it installs into its own `uv` venv beside
  Colab's interpreter.

## Quick start

1. Open the notebook in Colab and select a GPU runtime.
2. Run **Step 1** (GPU check — also fixes the CUDA wheel, precision and kernel backends) and
   **Step 2** (name the run, mount Drive).
3. Run **Step 3** (install) and **Step 4** (weights — skipped automatically if cached).
4. Provide a target (**Step 5**) and convert your hotspot numbering (**Step 6**).
5. Build the spec — the **form (Step 7)** or the **advanced YAML cell (Step 7 · advanced)**.
6. Set sampling (**Step 8**), generate (**Step 9**), collect and rank (**Step 10**), export the
   FASTA (**Step 11**), plot (**Step 12**), view (**Step 13**), download (**Step 14**).

Start with `NUM_REPLICATES=2` and `thinking_low` to confirm a spec before committing to a
campaign.

## Notebook structure

| Step  | Purpose |
|-------|---------|
| 1     | Check the GPU; set the CUDA wheel extra, precision and kernel backends |
| 2     | Mount Drive, name the run, set the per-run `WORK` directory, point at cached weights |
| Setup | Campaign status — what is done, what is left, which step to run next *(run any time)* |
| 3     | Install `rfd4-proteina` into an isolated Python 3.12 `uv` venv |
| 4     | Cache the `nvidia/RFD4-Proteina` bundle (skips if already present) |
| 5     | Upload or fetch the target; convert to CIF, strip, crop to the chains you name |
| 6     | Map author → `label_seq_id` numbering and build the hotspot selection |
| 7     | Build `design.yaml` from form fields |
| 7 · advanced | Paste a raw condition spec to override the form (every condition, several designs) |
| 8     | Sampling settings; resolve metric tags; write the hotspot-sampling config |
| 9     | Run `rfd4-generate`, streaming to console and `generate.log` |
| 10    | Parse designs, merge the metrics CSV, rank |
| 11    | Export designed + target sequences as FASTA for refolding elsewhere |
| 12    | Distributions and a Cβ distance map with chain boundaries marked |
| 13    | MolView structure browser |
| 14    | Zip / persist results |

## Configuring a design

A design is one short YAML block: an `input` structure, a `contig` saying which parts of it to
keep and where to insert variable-length designed segments, and `conditions` annotating what the
model should respect.

### Contig tokens

| Token | Meaning |
|-------|---------|
| `A12-180` | Preserved segment — kept from input by chain and `label_seq_id`, low to high |
| `/0` | Chain break; everything after starts a new chain with an automatic ID |
| `BINDER:60-120` | Named designed segment; length sampled in 60–120, `BINDER` becomes its `res_name` |
| `20-30`, `10R` | Unnamed designed segment (trailing `P`/`R`/`D` sets chain type) |

Chain assignment is per run (the tokens between `/0`s). A run containing a preserved segment
takes that segment's chain ID; a run without one gets a fresh ID. **Every contig needs at least
one preserved segment.**

### Conditions the form sets

| Condition | Value | Effect |
|-----------|-------|--------|
| `C_CRDGRP` | `1` | Target backbone as one rigid internal-coordinate group — geometry held, global pose and sidechains free |
| `C_CRD` | `true` | Alternative: pin the target where it sits in the input file |
| `C_SEQ` | `true` | Target sequence given in full, so it is not redesigned |
| `C_HOT` | `true` | Hotspot atoms the binder should engage |
| `C_SEQ_PROB_CYS` | `0.0` | Forbid cysteine in the designed segment |

| Task | Contig shape | Needs |
|------|--------------|-------|
| De-novo binder | `BINDER:<min>-<max>, /0, A<range>` | target |
| Motif scaffold | `A1-50, LINK:10-30, /0, BINDER:60-90` | motif structure |
| Partial diffusion | *(whole input)* | a generated design to refine, plus `PARTIAL_FRACTION` |
| Inverse folding | *(whole input)* | a backbone to re-sequence |
| Monomer | *(none — shipped benchmark input)* | repository clone |

### Advanced spec (Step 7 · advanced)

For anything the form cannot express, paste raw YAML. Running that cell overrides the form. The
full grammar — every annotation, `sel(...)` path syntax, query syntax, `global_args` — is in the
[YAML/JSON reference](https://rfd4-proteina.org/latest/reference/yaml-json.html).

## Hotspots

Hotspots mark where the binder should engage. They are a **bias, not a constraint** — check
obedience afterwards with the `hotspot_conditioning` metric.

Give **Step 6** your residues in author numbering and it returns the selection string for
Step 7, at one of two levels:

- `sidechain_tip` *(default)* — two distal atoms per residue, e.g.
  `sel('A/*/44/[CG1,CD1]')`. Falls back to CB/CA when a sidechain is not fully modelled.
- `residue` — every atom of the residue. Looser, and it lets the binder satisfy a hotspot by
  packing against backbone instead of the sidechain you cared about.

**Patch sampling** (`HOTSPOT_SAMPLING`, Step 8) re-draws 2–3 exposed hotspots per design from a
patch around your seeds — SASA-filtered, so they stay reachable. This is how the shipped VHH
campaign explores an epitope. It *replaces* the seed set, so the spec must already set `C_HOT`.

## Metrics

`METRICS_TAGS="auto"` enables the oracle-free tags your conditions make meaningful:

| Tag | Reports | Enabled when |
|-----|---------|--------------|
| `design` | Clashes, bond-length geometry, secondary-structure distribution and accuracy, sequence adherence | always |
| `hotspot_conditioning` | Hotspot obedience | `C_HOT` set |
| `internal_coordinate_conditioning` | Motif all-atom RMSD — confirms the rigid group was held | `C_CRDGRP` set |
| `3d_coordinate_conditioning` | Motif coordinate RMSD | `C_CRD` set |
| `rasa_conditioning` | RASA adherence | `C_RAS` set |
| `interchain_interactions` | Interchain interaction adherence and counts | multi-chain contig |
| `cms` | Contact molecular surface — the standard interface-quality number for a designed binder | multi-chain contig |
| `partial_diffusion` | Sequence similarity and Cα RMSD to the input — how far a child moved from its parent | partial-diffusion task |

`sequence_adherence.n_violations` is the one to watch: nonzero means the model did not reproduce
a sequence you conditioned, which on a binder run means **the target was redesigned**.

Every `*_refolding` tag, the `antibody` tag's templated refolding, and any search checkpoint
scoring `oracle.*` need a refolding runtime and are unavailable here. `developability` is half
usable — `sequence_developability` runs, `rosetta_developability` wants a PyRosetta container.

## Resuming an interrupted campaign

Colab disconnects are the normal reason a long campaign stops part-way. Nothing is lost — every
run is a self-contained folder on Drive.

1. Re-run **Step 2** with `RESUME_RUN = latest` (or a run folder name, or a full path). `WORK`
   reattaches to that run instead of creating a new one.
2. Re-run **Step 3** (install) and **Step 4** (weights — skipped, already cached).
3. Run the **campaign status** cell to see what is done and which step to run next.
4. Re-run **Step 9**. With `SKIP_EXISTING = True` (the default), finished designs are kept and
   only the missing ones are produced — which is also how you *grow* a campaign: raise
   `NUM_REPLICATES` and re-run.

## Outputs

Everything lands under your run's `WORK` directory (local or Drive):

```
runs/<date-time>_<name>/
├── design.yaml                          # the condition spec that produced this run
├── rfd4_colab.yaml                      # composed config (only when hotspot sampling is on)
├── generate.log
├── inputs/
│   ├── <name>_target.cif                # cropped, cleaned target
│   └── <pdb>_raw.cif
├── inference_samples/<example_id>/
│   ├── <example_id>_generated.cif       # the design
│   ├── <example_id>_input.cif           # conditioning sidecar — what the model actually saw
│   └── <example_id>_chain.cif           # designed chain only (save_chain_of_interest)
├── inference_metrics/
│   └── per_example_metrics_job0.csv
├── figures/
│   ├── <name>_distributions.png
│   └── <name>_<design>_distmap.png
├── <name>_designs.csv                   # ranked table, metrics merged
└── <name>_designs.fasta                 # design + target, ready to refold
```

Open `<example_id>_input.cif` whenever a contig did not do what you expected — it is the
conditioning as the model received it, not as you wrote it.

## Notes & gotchas

- **Condition the target, always.** Anything not mentioned in `conditions` is not provided. A
  spec with only `C_HOT` leaves the target's sequence free, and you get a redesigned target and a
  *"Could not unambiguously determine chain of interest"* warning — the heuristic picks the least
  sequence-conditioned protein chain, so an unconditioned target ties with the binder and metrics
  silently score the wrong chain. The form handles this; hand-written specs must too.
- **Numbering.** Contigs and `sel(...)` index `label_seq_id`. A fetched PDB entry's author
  numbering usually differs by a constant offset, but not always — insertion codes and gaps break
  it. Step 6 reports the offset and says when it is not constant.
- **`uv` needs `--python`.** A `VIRTUAL_ENV=<dir> uv pip install ...` shell prefix is not reliably
  honoured: uv installs into Colab's own interpreter instead and still exits 0, leaving an empty
  venv. Always target the interpreter explicitly.
- **`MPLBACKEND` breaks the venv.** Colab exports
  `MPLBACKEND=module://matplotlib_inline.backend_inline`, which does not exist inside the venv, so
  matplotlib raises on import and takes `rfproteina` down with it (via `atomworks` → `biotite`).
  Every subprocess call into the venv must pass `MPLBACKEND=Agg`.
- **Pre-oracle filters are disabled on purpose.** They decide what is worth sending to an oracle,
  and upstream `delete_failing_pre_oracle=true` would **delete** the structure files of designs
  that fail one. With no oracle in the loop, Step 8 forces `false` so you keep every design and
  filter yourself.
- **Memory.** In order: `BATCH_SIZE=1`, crop the target to the domain you care about, shorten the
  binder range, then drop to `thinking_low`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
  already set.
- **`torch.compile` rarely pays back here.** Warm-up compiles 15 length buckets, several minutes,
  and only amortises over a large run. Off by default (`compile_lengths=null`).
- **If a run dies inside a metric**, set `METRICS_TAGS` manually and drop the suspect tag — `cms`
  first, since contact molecular surface runs through `tmol`. Generation itself is unaffected;
  metrics run after each design is written.
- **Reproducibility.** The default sampler is stochastic and will not reproduce even at a fixed
  seed. Set `DETERMINISTIC_ODE=True` for ODE sampling, and pin the checkpoint with
  `hf_ckpt_revision=<commit>`.
- **Validation is out-of-family by necessity.** Nothing here refolds, so there is no in-family
  bias to worry about — but equally no validation. Refold with AF3, Protenix or OpenFold3 and keep
  designs with a confident interface **and** a low RMSD back to the design model. Interface
  confidence alone is not enough: a refold can be confident about a different pose.
- **Scale.** Published throughput is roughly 1,000–2,400 designs per GPU-hour on datacentre
  hardware; a Colab L4 sits well below that and a T4 further still. Use this notebook to get a
  spec right on a handful of designs, then move the campaign to a SLURM array — the repository
  ships `scripts/launch_generate.sh` and the `gen_njobs` / `job_id` options for exactly that.
  FoldSeek-clustered pass rate, the metric the paper headlines, also belongs there.

## References

```bibtex
@article{corley2026rfd4proteina,
  author  = {Corley, Nathaniel and Didi, Kieran and Kazman, Maxwell W. and Brent, Rafael I. and others},
  title   = {Biomolecular design through latent reasoning with {RFD4-Proteina}},
  year    = {2026},
  note    = {Preprint},
  url     = {https://rfd4-proteina.org/assets/paper/RFD4-Proteina.pdf}
}

@article{rfdiffusion3,
  title   = {Atomic functional site scaffolding in biopolymers with {RFdiffusion3}},
  journal = {Science},
  doi     = {10.1126/science.aeh2982}
}

@article{abramson2024af3,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and others},
  title   = {Accurate structure prediction of biomolecular interactions with {AlphaFold} 3},
  journal = {Nature},
  volume  = {630},
  pages   = {493--500},
  year    = {2024},
  doi     = {10.1038/s41586-024-07487-9}
}
```

## License & acknowledgements

This notebook is a convenience wrapper. RFD4-Proteina is released under the **BSD 3-Clause**
license by the Baker Lab (Institute for Protein Design, University of Washington) and NVIDIA
GenAIR; the model weights are distributed by NVIDIA on
[Hugging Face](https://huggingface.co/nvidia/RFD4-Proteina) under their own terms. AtomWorks,
biotite, gemmi and MolView are the work of their respective authors and carry their own licenses —
consult each upstream repository. Please cite the RFD4-Proteina preprint for any work using this
notebook.
