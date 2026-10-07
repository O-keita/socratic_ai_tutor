#!/bin/sh
# Refresh the deploy copies from ../backend. Run before `vercel --prod`.
set -e
cd "$(dirname "$0")"
B=../backend
rm -rf ml evaluation pages
mkdir -p data pages
cp -r $B/ml ml && cp -r $B/evaluation evaluation
rm -rf ml/__pycache__ evaluation/__pycache__ evaluation/logs ml/fine_tune_lora.py ml/test_inference.py
cp $B/data/config.json data/config.json
cp $B/index.html $B/chat.html $B/admin.html $B/logo.png pages/
echo "synced"
