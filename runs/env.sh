# Model-asset locations every runner needs. Sourced by runs/launch.sh so a job can never fail
# for a missing env var that the queue line simply forgot -- PertAdapt T4 failed exactly that way.
# Paths are asserted at source time: a wrong path must fail loudly here, not as a model decline.
# The vendored copy under benchmark/vendor has load.py but NOT OS_scRNA_gene_index.19264.tsv,
# which the runner also reads -- it fails one step later. Use the full upstream tree.
export IVCBENCH_SCFOUNDATION_DIR="/data1/home/chlee/projects/single_cell_fm/scFoundation/model"
export IVCBENCH_SCFOUNDATION_CKPT="/data1/home/chlee/projects/single_cell_fm/models/scFoundation/models.ckpt"
export IVCBENCH_SCGPT_MODEL_DIR="/data1/home/chlee/projects/single_cell_fm/models/scGPT_human"
export IVCBENCH_CELLOT_PROGRESS="/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/cellot_progress.tsv"
# AttentionPert's released source tree (the runner does sys.path.insert on this).
export IVCBENCH_ATTNPERT_SRC="/data1/home/chlee/projects/scPerturBench/image_rootfs/home/software/AttentionPert"
# Jobs per GPU. Two, not three. Memory is not the constraint -- these L40s hold 46-49 GB and the
# jobs take 3-10 GB -- but COMPUTE is: with three jobs per card every GPU sat at 100% and CPA's
# training, which is many small kernels, went from 18 s/epoch to 7 min/epoch. Small-kernel jobs
# degrade far worse than the job count under time-slicing, so the third slot cost more than it
# bought. dispatch.sh's FREE_MB gate still guards memory on top of this.
export IVCBENCH_JOBS_PER_GPU=2
export IVCBENCH_GENE2GO="/data1/home/chlee/projects/single_cell_fm/scFoundation/GEARS/data/gene2go.pkl"

# Where a run DEPOSITS its prediction bundle. Without this dump_bundle is a silent no-op: every
# native-coverage run of 2026-09-14 produced a score in a CSV and NO bundle, so none of them could
# have entered the census, which is assembled from predictions/**/*.npz. A separate subdirectory,
# never predictions/ itself -- a new run must not overwrite the historical bundle at the same
# filename, because withdrawn_bundles.csv pins those by path AND sha256.
export IVCBENCH_PRED_DUMP="/data1/home/chlee/projects/immune virtual cell/ivcbench/predictions/v2_native"
export IVCBENCH_PRED_DUMP_MEANS=1   # per-stratum means: the form the deposited bundles use
mkdir -p "$IVCBENCH_PRED_DUMP"

for _v in IVCBENCH_SCFOUNDATION_DIR IVCBENCH_SCFOUNDATION_CKPT IVCBENCH_SCGPT_MODEL_DIR IVCBENCH_GENE2GO IVCBENCH_ATTNPERT_SRC; do
  eval "_p=\$$_v"
  [ -e "$_p" ] || { echo "runs/env.sh: $_v points at a missing path: $_p" >&2; exit 90; }
done
for _f in load.py OS_scRNA_gene_index.19264.tsv; do
  [ -e "$IVCBENCH_SCFOUNDATION_DIR/$_f" ] || { echo "runs/env.sh: IVCBENCH_SCFOUNDATION_DIR missing $_f" >&2; exit 90; }
done
[ -e "$IVCBENCH_SCGPT_MODEL_DIR/vocab.json" ] || { echo "runs/env.sh: IVCBENCH_SCGPT_MODEL_DIR missing vocab.json" >&2; exit 90; }
