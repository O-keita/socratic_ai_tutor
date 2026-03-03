#!/bin/sh
set -e

# Always sync config.json from the image (code) into the mounted /app/data/
# volume. This ensures that config changes in git take effect after rebuild,
# while runtime data (users.json, admin_sessions.json) persists via the volume.
if [ -f /app/_image_config.json ]; then
    cp /app/_image_config.json /app/data/config.json
    echo "[entrypoint] Synced config.json from image into data volume."
fi

exec "$@"
