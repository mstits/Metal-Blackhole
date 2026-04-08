#!/bin/bash
# Compiles Metal shaders into a precompiled .metallib binary
# Requires Xcode (not just Command Line Tools) for xcrun metal
# Usage: build_metallib.sh <header_path> <shader_path> <output_dir>

HEADER="$1"
SHADER="$2"
OUTDIR="$3"

# Check if Metal compiler is available (requires full Xcode)
if ! xcrun --find metal >/dev/null 2>&1; then
    echo "Note: Metal compiler not found (requires Xcode). Using runtime shader compilation."
    exit 0
fi

set -e

COMBINED="${OUTDIR}/combined_shader.metal"
AIR="${OUTDIR}/geodesic.air"
METALLIB="${OUTDIR}/geodesic.metallib"

# Concatenate header + shader (mirrors runtime prepend approach)
cat "$HEADER" "$SHADER" > "$COMBINED"

# Compile to AIR (Metal Intermediate Representation)
xcrun -sdk macosx metal -c -std=metal3.0 "$COMBINED" -o "$AIR"

# Link to .metallib binary
xcrun -sdk macosx metallib "$AIR" -o "$METALLIB"

# Clean intermediates
rm -f "$COMBINED" "$AIR"

echo "Built $METALLIB"
