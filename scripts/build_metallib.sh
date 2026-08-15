#!/bin/bash
# Compiles Metal shaders into a precompiled .metallib binary
# Requires Xcode (not just Command Line Tools) for xcrun metal
# Usage: build_metallib.sh <header_path> <shader_path> <output_dir>

HEADER="$1"
SHADER="$2"
OUTDIR="$3"

# Check if Metal compiler is available (requires full Xcode). Remove any
# stale metallib from a previous toolchain so the app can never silently
# prefer an outdated binary over freshly edited source.
if ! xcrun --find metal >/dev/null 2>&1; then
    rm -f "${OUTDIR}/geodesic.metallib"
    echo "Note: Metal compiler not found (requires Xcode). Using runtime shader compilation."
    exit 0
fi

set -e

COMBINED="${OUTDIR}/combined_shader.metal"
AIR="${OUTDIR}/geodesic.air"
METALLIB="${OUTDIR}/geodesic.metallib"

# Concatenate header + shader (mirrors runtime prepend approach)
cat "$HEADER" "$SHADER" > "$COMBINED"

# Compile to AIR (Metal Intermediate Representation).
# PRECISION POLICY: relaxed math (non-IEEE sqrt/division allowed, Inf/NaN
# preserved) — matches the runtime path's MTLMathModeRelaxed exactly. Full
# fast math remains forbidden. Toolchains without the flag fall back to
# -fno-fast-math, which is strictly safer.
if echo | xcrun -sdk macosx metal -x metal -fmetal-math-mode=relaxed -fsyntax-only - >/dev/null 2>&1; then
    MATH_FLAG="-fmetal-math-mode=relaxed"
else
    MATH_FLAG="-fno-fast-math"
fi
xcrun -sdk macosx metal -c -std=metal3.0 $MATH_FLAG "$COMBINED" -o "$AIR"

# Link to .metallib binary
xcrun -sdk macosx metallib "$AIR" -o "$METALLIB"

# Clean intermediates
rm -f "$COMBINED" "$AIR"

echo "Built $METALLIB"
