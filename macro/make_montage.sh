#!/bin/bash

PATTERN="$1"      # e.g. h_clusSize
OUT="${PATTERN}.png"

FILES=$(ls *_${PATTERN}.png 2>/dev/null)

if [ -z "$FILES" ]; then
  echo "No files matching *_${PATTERN}.png"
  exit 1
fi

# Avoid self-inclusion
rm -f "$OUT"

N=$(echo "$FILES" | wc -l)
if [ "$N" -gt 5 ]; then
  echo "Warning: more than 5 files ($N); proceeding anyway."
fi

echo "Making montage for $PATTERN with $N files"

montage $FILES -tile x1 -geometry +12+12 "$OUT"

echo "Saved: $OUT"

