#!/usr/bin/env bash
# Run the prepared full system and three frozen-water fragment pairs.
set -eo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Usage: bash run_all.sh
Run this script after preparing the interaction-energy inputs.
Environment:
  VASP_EXECUTABLE   VASP executable or path (default: vasp_gam)
  VASP_ENV          Optional shell file that initializes the VASP environment
  MPIEXEC          MPI launcher executable or path (default: mpirun)
  AB_RANKS         MPI ranks for the full system (default: 60)
  COMPONENT_RANKS  MPI ranks per fragment (default: 30)
  MAX_PARALLEL     Number of simultaneous fragment jobs (default: 1)
  PYTHON_EXECUTABLE Python used for the final summary (default: python3)
Inputs include a locally supplied, licensed POTCAR in each calculation directory.
USAGE
    exit 0
fi
if (( $# )); then
    echo "Unexpected argument: $1; use --help" >&2
    exit 2
fi
if [[ -n ${VASP_ENV:-} ]]; then
    source "$VASP_ENV"
fi
set -u
root=$(cd "$(dirname "$0")" && pwd)
vasp=${VASP_EXECUTABLE:-vasp_gam}
mpi=${MPIEXEC:-mpirun}
ab_ranks=${AB_RANKS:-60}
component_ranks=${COMPONENT_RANKS:-30}
parallel=${MAX_PARALLEL:-1}
python=${PYTHON_EXECUTABLE:-python3}
for value in "$ab_ranks" "$component_ranks" "$parallel"; do
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
        echo "Rank and concurrency values must be positive integers" >&2
        exit 2
    fi
done
resolve_executable() {
    local resolved
    resolved=$(command -v "$1") || { echo "Executable not found: $1" >&2; return 2; }
    [[ "$resolved" == /* ]] || resolved="$PWD/$resolved"
    [[ -f "$resolved" && -x "$resolved" ]] || { echo "Not an executable file: $resolved" >&2; return 2; }
    printf '%s\n' "$resolved"
}
vasp=$(resolve_executable "$vasp")
mpi=$(resolve_executable "$mpi")
python=$(resolve_executable "$python")
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export I_MPI_PIN=0
unset I_MPI_PIN_PROCESSOR_LIST
shopt -s nullglob
waters=("$root"/water*_O*_z*A)
if (( ${#waters[@]} != 3 )); then
    echo "Expected three prepared water directories in $root" >&2
    exit 2
fi
calcs=()
for water in "${waters[@]}"; do
    calcs+=("$water/a_without_water" "$water/b_single_water")
done
for calc in "$root/ab_full" "${calcs[@]}"; do
    for name in POSCAR INCAR KPOINTS POTCAR; do
        [[ -f "$calc/$name" ]] || { echo "Missing input: $calc/$name" >&2; exit 2; }
    done
done

run_calc() {
    local calc=$1 ranks=$2
    if grep -q 'General timing and accounting' "$calc/OUTCAR" 2>/dev/null; then
        echo "Existing completed output: $calc"
        return
    fi
    echo "Starting $calc with $ranks MPI ranks"
    (
        cd "$calc"
        "$mpi" -np "$ranks" "$vasp" > vasp.log 2>&1
    )
    grep -q 'General timing and accounting' "$calc/OUTCAR"
}

run_calc "$root/ab_full" "$ab_ranks"
pids=()
status=0
for calc in "${calcs[@]}"; do
    run_calc "$calc" "$component_ranks" &
    pids+=("$!")
    if (( ${#pids[@]} >= parallel )); then
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        (( status == 0 )) || exit 1
        pids=()
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 )) || exit 1
"$python" "$root/finalize_results.py"
