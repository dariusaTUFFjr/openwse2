#!/bin/bash
# Phase 5 R7 GDB helper (`docs/RECOMP_PLAN.md` "R7 — WSL x86-64 Bring-Up & Savegame Oracle"):
# Runs `r4_run` under gdb with `solib-search-path <work>/so` so faults or signals inside translated
# x86-64 `.so` batches show symbolicated backtraces, registers, and disassembly.
#
# Usage:
#   tools/r7_gdb.sh [--batch] [--bin <r4_run>] [--] <install dir> [r4_run flags...]

set -euo pipefail

BATCH=0
BIN="${WSE_WSL_BUILD:-/root/wse/build-recomp}/r4_run"
ARGS=()

while (($#)); do
    case "$1" in
        --batch)
            BATCH=1
            shift
            ;;
        --bin)
            BIN="$2"
            shift 2
            ;;
        --)
            shift
            ARGS+=("$@")
            break
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "${#ARGS[@]}" -eq 0 ]; then
    echo "usage: $0 [--batch] [--bin <r4_run>] <install dir> [r4_run flags...]" >&2
    exit 2
fi

if [ ! -x "$BIN" ]; then
    echo "r7_gdb.sh: binary not found or not executable: $BIN" >&2
    exit 2
fi

WORK="/root/wse/r4"
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" = "--work" ] && ((i + 1 < ${#ARGS[@]})); then
        WORK="${ARGS[$((i + 1))]}"
        break
    fi
done

SOLIB="${WORK}/so"
echo "== r7_gdb: bin=$BIN solib-search-path=$SOLIB batch=$BATCH" >&2

if [ "$BATCH" -eq 1 ]; then
    exec gdb -batch \
        -ex "set pagination off" \
        -ex "set confirm off" \
        -ex "set solib-search-path $SOLIB" \
        -ex "run" \
        -ex "echo \n=== Backtrace ===\n" \
        -ex "bt 30" \
        -ex "echo \n=== Registers ===\n" \
        -ex "info registers" \
        -ex "echo \n=== Disassembly around PC ===\n" \
        -ex "x/16i \$pc" \
        -ex "echo \n=== Thread list ===\n" \
        -ex "info threads" \
        --args "$BIN" "${ARGS[@]}"
else
    exec gdb \
        -ex "set pagination off" \
        -ex "set solib-search-path $SOLIB" \
        --args "$BIN" "${ARGS[@]}"
fi
