# Style-Action Coupling — Evaluation Pipeline

Diagnostic toolkit for the paper *Diagnosing Style-Action Coupling in
Text-to-Motion Models*. Measures whether style conditioning preserves the
action-by-action structure of a text-to-motion model's latent space, using
CKA, GED, style-vector consistency, and Action Preservation, against a raw
CLIP text-encoder control.

## Repository layout

```
eval/
├── prompts/
│   ├── grid.py            v1: 8 actions × 7 styles; 3 paraphrase templates
│   ├── grid_v2.py         v2: 24 actions × 13 styles, corpus-mined
│   ├── mine_grid.py       action/adverb frequencies from HumanML3D captions
│   └── validate.py        CLIP near-synonymy screen for the vocabulary
├── extract/
│   ├── base.py            LatentExtractor ABC + cosine_kernel
│   ├── clip_control.py    raw CLIP-ViT-B/32 text embeddings (the control)
│   ├── mdm.py             MDM: denoised x0, mean-pooled over time (263-d)
│   ├── mdm_probe.py       single-prompt probe; validates extraction sites (run first)
│   └── t2mgpt.py          T2M-GPT: pre-quantisation VQ latents (512-d)
├── metrics/
│   ├── cka.py             linear CKA + permutation null
│   ├── ged.py             normalised graph edit distance + null
│   ├── tau_select.py      data-adaptive GED threshold (quantiles of K_S)
│   ├── style_vectors.py   δ_j(a) consistency + null + transfer targets
│   └── ap.py              action preservation via HumanML3D classifier
├── classifier/humanml3d.py
├── pipeline.py            extract / generate / score stages
├── score_hpc.py           score multi-template HPC results
├── score_seeds.py         aggregate multi-seed runs (mean ± sd)
├── robustness.py          re-runs CKA/GED under all paraphrase templates
├── robustness_stats.py    leave-2-out subsample CIs + tau sensitivity
├── linear_probe.py        leave-one-action-out supervised style recovery
├── bayes_escape.py        hierarchical Bayesian escape model (PyMC)
├── make_figures.py        kernel heatmaps + per-style CKA bars
├── make_phase_portrait.py arrow-level style vectors, escapes in red
├── make_landscape.py      coherence flow fields (+3-panel progression)
├── make_manifold.py       hillshaded 3D landscape (talks/posters)
├── make_schematic.py      Figure 1 conceptual schematic
├── slurm/                 sbatch scripts for the full run
├── requirements-hpc.txt   verified cluster install (see header for order)
├── environment.yml        local conda env (laptop pilot)
└── results/{model}/       all artifacts land here
```

## Kernel layout

Matches the paper's per-style formulation:

- `Z_S.npy` `(8, d)` neutral action embeddings → `K_S.npy` `(8, 8)`
- `Z_T.npy` `(7, 8, d)` styled embeddings → `K_T.npy` `(7, 8, 8)`, one kernel per style
- CKA/GED computed per style against the shared `K_S`; reported as per-style profiles + means
- The GED threshold τ is selected per representation space from quantiles of
  the off-diagonal `K_S` values. Fixed thresholds do not transfer: CLIP text
  embeddings are anisotropic (all pairwise cosines in [0.83, 0.94]), so any
  fixed τ below that band yields a complete graph and GED ≡ 0.

## Reproducing the CLIP control (laptop, CPU, ~2 min)

This is real paper data — the text-encoder baseline column and both current
figures — and needs no motion model:

```bash
pip install torch numpy matplotlib git+https://github.com/openai/CLIP.git

python pipeline.py --model clip --stage extract   # downloads ViT-B/32 (~340MB)
python pipeline.py --model clip --stage score     # → results/clip/report.json
python robustness.py --model clip                 # → results/clip/robustness.json
python make_figures.py --model clip --out ../overleaf/figures
```

Expected headline numbers (template 1): mean CKA 0.957, mean GED 0.301,
BER 0.036 (2/56 escapes, both under *tiredly*). Style-vector consistency
mean 0.69; *heavily* dissociates (highest CKA 0.984, lowest consistency 0.547).

## Motion models on the cluster

Environment (once — see `requirements-hpc.txt` header for the full recipe
and the reasons behind each pin):

```bash
python3 -m venv ~/sac-env && source ~/sac-env/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-hpc.txt
pip install --no-build-isolation chumpy
```

MDM setup (paths assume a `noexec` data mount — venv in `$HOME`, everything
else in data storage):

```bash
cd /data/$USER/sac
git clone https://github.com/GuyTevet/motion-diffusion-model
cd motion-diffusion-model
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
gdown 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr -O save/ckpt.zip && (cd save && unzip ckpt.zip)

# HumanML3D normalisation stats + caption texts (no AMASS agreement needed):
git clone --depth 1 https://github.com/EricGuo5513/HumanML3D ../hml3d-repo
mkdir -p dataset/HumanML3D
cp ../hml3d-repo/HumanML3D/{Mean,Std}.npy ../hml3d-repo/HumanML3D/*.txt dataset/HumanML3D/
(cd dataset/HumanML3D && unzip -q ../../../hml3d-repo/HumanML3D/texts.zip)
```

Validate the extraction site before anything else (single prompt, ~1 min on GPU):

```bash
srun -c 4 --mem=16G --gres=gpu:1 --time=00:10:00 \
  python mdm_probe.py \
    --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
    --text_prompt "a person is walking"
# expect: text emb (1, 1, 512); x0 (1, 263, 1, 120); pooled latent (263,)
```

Known MDM codebase gotchas (all handled in `mdm_probe.py`):
- `MDM.to()` returns `None` — never chain `.to(device).eval()`
- `create_model_and_diffusion(args, data)` needs `data` with a `.dataset`
  attribute; pass `SimpleNamespace(dataset=SimpleNamespace())`
- classifier-free guidance goes through `ClassifierFreeSampleModel` with the
  scale in `model_kwargs["y"]["scale"]`

Full grid run:

```bash
mkdir -p logs
EXTRACT_JID=$(sbatch --parsable slurm/extract.sh)
GEN_MDM=$(sbatch --parsable --export=MODEL=mdm    --dependency=afterok:$EXTRACT_JID slurm/generate.sh)
GEN_T2M=$(sbatch --parsable --export=MODEL=t2mgpt --dependency=afterok:$EXTRACT_JID slurm/generate.sh)
sbatch --export=MODEL=mdm    --dependency=afterok:$GEN_MDM slurm/score.sh
sbatch --export=MODEL=t2mgpt --dependency=afterok:$GEN_T2M slurm/score.sh
```

## Output format

`results/{model}/report.json`:

```json
{
  "model": "clip",
  "tau": 0.8906,
  "cka_mean": 0.9573,
  "ged_mean": 0.3010,
  "ap": NaN,
  "ber": 0.0357,
  "per_style_cka": {"angrily": 0.9497, "tiredly": 0.9264, "...": "..."},
  "per_style_ged": {"...": "..."},
  "per_action_ber": {"a person is walking": 0.0, "...": "..."},
  "type_a_actions": [],
  "type_b_actions": []
}
```

`results/clip/report.json` and `results/clip/robustness.json` are committed
as the reference run.
