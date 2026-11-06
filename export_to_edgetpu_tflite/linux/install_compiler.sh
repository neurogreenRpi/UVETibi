#!/usr/bin/env bash
# install_compiler.sh - copie le compilateur depuis le dossier Windows vers Linux
set -euo pipefail
SRC="/mnt/a/Camion/compiler_coral_candidate"
DIR="/root/tools/coral-compiler-candidate"
mkdir -p "$DIR"
cp -a "$SRC"/. "$DIR"/
chmod +x "$DIR/edgetpu_compiler" "$DIR/ld-linux-x86-64.so.2"
"$DIR/ld-linux-x86-64.so.2" --library-path "$DIR" "$DIR/edgetpu_compiler" --version
