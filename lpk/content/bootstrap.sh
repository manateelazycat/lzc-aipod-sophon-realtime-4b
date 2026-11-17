#!/bin/sh
set -eu
# Fixed, anonymously verified controller delivery. Model downloads belong to Host.
BUNDLE_SHA='8ebffd53cd6e19d3df44441dbdff88ec8a4d21c0659d332d232d6d6d1ec995bc'
BUNDLE_SIZE='35927797'
MS_URL='https://modelscope.cn/models/manateelazycat/Sophon-Realtime-4B-Flash-AIPod/resolve/49a8fae806883409b1b8aae502a67cc89afb1066/control/sophon-realtime-4b-control-0.1.0.tar.gz'
HF_URL='https://huggingface.co/manateelazycat/Sophon-Realtime-4B-Flash-AIPod-HF/resolve/af4139f49e083aa1765a841b8c08f6a6cc966e30/control/sophon-realtime-4b-control-0.1.0.tar.gz'
DATA_ROOT="${SOPHON_REALTIME_DATA_DIR:-/lzcapp/var/sophon-realtime-4b}"
CACHE="$DATA_ROOT/control-plane"
ROOT="$CACHE/$BUNDLE_SHA"
ARCHIVE="$CACHE/$BUNDLE_SHA.tar.gz"
mkdir -p "$CACHE"
verify() {
  [ -f "$1" ] && [ "$(wc -c < "$1" | tr -d ' ')" = "$BUNDLE_SIZE" ] &&
    [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$BUNDLE_SHA" ]
}
if ! verify "$ARCHIVE"; then
  temp="$CACHE/.download-$$"
  trap 'rm -f "$temp" "$temp".*; rm -rf "$CACHE/.unpack-$$"' EXIT HUP INT TERM
  success=false
  for url in "$MS_URL" "$HF_URL"; do
    echo 'Downloading verified Sophon-Realtime-4B-Flash controller'
    attempt=0
    while [ "$attempt" -lt 4 ]; do
      if wget -q -c -T 90 -O "$temp" "$url" && verify "$temp"; then
        mv "$temp" "$ARCHIVE"
        success=true
        break
      fi
      if [ -f "$temp" ] && [ "$(wc -c < "$temp" | tr -d ' ')" -ge "$BUNDLE_SIZE" ]; then
        rm -f "$temp"
      fi
      attempt=$(( attempt + 1 ))
      [ "$attempt" -ge 4 ] || sleep 2
    done
    [ "$success" = false ] || break
    rm -f "$temp"
  done
  if [ "$success" != true ]; then
    echo 'Controller download or size/SHA256 validation failed' >&2
    exit 1
  fi
fi
if [ ! -f "$ROOT/.complete" ]; then
  temp_root="$CACHE/.unpack-$$"
  trap 'rm -rf "$temp_root"' EXIT HUP INT TERM
  mkdir -p "$temp_root"
  tar -tzf "$ARCHIVE" > "$temp_root/list"
  if awk '/^\// || /(^|\/)\.\.($|\/)/ {bad=1} END {exit !bad}' "$temp_root/list"; then
    echo 'Invalid controller archive path' >&2
    exit 1
  fi
  rm "$temp_root/list"
  tar -xzf "$ARCHIVE" -C "$temp_root"
  [ -x "$temp_root/lib/ld-linux-x86-64.so.2" ]
  [ -x "$temp_root/usr/local/bin/python3.12" ]
  [ -f "$temp_root/app/control/app.py" ]
  touch "$temp_root/.complete"
  mv "$temp_root" "$ROOT"
fi
export PYTHONHOME="$ROOT/usr/local"
export PYTHONPATH="$ROOT/app"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/usr/local/lib"
export SSL_CERT_FILE="$ROOT/etc/ssl/certs/ca-certificates.crt"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "$ROOT/lib/ld-linux-x86-64.so.2" --library-path "$LD_LIBRARY_PATH" \
  "$ROOT/usr/local/bin/python3.12" -m uvicorn control.app:app --host 0.0.0.0 --port 8080
