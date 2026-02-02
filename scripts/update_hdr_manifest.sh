#!/bin/bash
# Regenerate the HDR environment map manifest
# Run this after adding new .hdr files to assets/env/

ENV_DIR="$(dirname "$0")/../assets/env"
MANIFEST="$ENV_DIR/manifest.json"

echo "[" > "$MANIFEST"

first=true
for f in "$ENV_DIR"/*.hdr; do
  [ -e "$f" ] || continue
  filename=$(basename "$f")
  # Convert filename to display name: remove _1k.hdr, replace _ with space, title case
  name=$(echo "${filename%_1k.hdr}" | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')

  if [ "$first" = true ]; then
    first=false
  else
    echo "," >> "$MANIFEST"
  fi

  printf '  { "file": "%s", "name": "%s" }' "$filename" "$name" >> "$MANIFEST"
done

echo "" >> "$MANIFEST"
echo "]" >> "$MANIFEST"

echo "Updated $MANIFEST with $(grep -c '"file"' "$MANIFEST") HDR maps"
