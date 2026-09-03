#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
manifest="$project_dir/assets/staunton/sources.json"
asset_dir="$project_dir/assets/staunton"

for dependency in curl jq rg unzip zipinfo cargo; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Required command is missing: $dependency" >&2
        exit 1
    fi
done

sketchfab_token="${SKETCHFAB_API_TOKEN:-}"
if [[ -z "$sketchfab_token" ]]; then
    if [[ ! -t 0 ]]; then
        echo "Set SKETCHFAB_API_TOKEN or run this script in a terminal to enter it securely." >&2
        exit 1
    fi
    read -r -s -p "Sketchfab API token: " sketchfab_token
    echo
fi
if [[ ! "$sketchfab_token" =~ ^[A-Za-z0-9._~-]+$ ]]; then
    echo "The token contains characters that cannot be passed safely to curl." >&2
    exit 1
fi

auth_scheme="${SKETCHFAB_AUTH_SCHEME:-Token}"
if [[ "$auth_scheme" != "Token" && "$auth_scheme" != "Bearer" ]]; then
    echo "SKETCHFAB_AUTH_SCHEME must be Token or Bearer." >&2
    exit 1
fi

working_dir="$(mktemp -d)"
cleanup() {
    rm -rf -- "$working_dir"
}
trap cleanup EXIT

mkdir -p -- "$asset_dir"

while IFS=$'\t' read -r piece uid; do
    if [[ ! "$piece" =~ ^(pawn|rook|knight|bishop|queen|king)$ ]]; then
        echo "Unexpected piece name in manifest: $piece" >&2
        exit 1
    fi
    if [[ ! "$uid" =~ ^[a-f0-9]{32}$ ]]; then
        echo "Unexpected Sketchfab UID in manifest: $uid" >&2
        exit 1
    fi

    echo "Fetching $piece metadata..."
    metadata="$working_dir/$piece.metadata.json"
    curl --fail --silent --show-error \
        "https://api.sketchfab.com/v3/models/$uid" \
        --output "$metadata"
    jq --exit-status \
        --arg uid "$uid" \
        '.uid == $uid and .isDownloadable == true and .license.slug == "by"' \
        "$metadata" >/dev/null

    download_response="$working_dir/$piece.download.json"
    printf 'header = "Authorization: %s %s"\n' "$auth_scheme" "$sketchfab_token" | \
        curl --config - --fail --silent --show-error \
            "https://api.sketchfab.com/v3/models/$uid/download" \
            --output "$download_response"
    download_url="$(jq --exit-status --raw-output '.gltf.url' "$download_response")"

    archive="$working_dir/$piece.zip"
    curl --fail --location --silent --show-error "$download_url" --output "$archive"
    if zipinfo -1 "$archive" | rg --quiet '(^/|(^|/)\.\.(/|$))'; then
        echo "Refusing an archive containing an unsafe path for $piece." >&2
        exit 1
    fi

    extracted="$working_dir/$piece-source"
    mkdir -p -- "$extracted"
    unzip -q "$archive" -d "$extracted"
    scene_path="$(find "$extracted" -type f -name 'scene.gltf' -print -quit)"
    if [[ -z "$scene_path" ]]; then
        scene_path="$(find "$extracted" -type f \( -name '*.gltf' -o -name '*.glb' \) -print -quit)"
    fi
    if [[ -z "$scene_path" ]]; then
        echo "The downloaded archive for $piece contains no glTF scene." >&2
        exit 1
    fi

    prepared="$working_dir/$piece.mesh"
    cargo run --release --quiet \
        --manifest-path "$project_dir/Cargo.toml" \
        --bin prepare_staunton_asset -- \
        --piece "$piece" \
        --input "$scene_path" \
        --output "$prepared"

    prepared_tmp="$asset_dir/$piece.mesh.tmp"
    install -m 0644 "$prepared" "$prepared_tmp"
    mv -f -- "$prepared_tmp" "$asset_dir/$piece.mesh"

    metadata_tmp="$asset_dir/$piece.metadata.json.tmp"
    jq '{
        uid,
        name,
        viewerUrl,
        user: {username: .user.username, profileUrl: .user.profileUrl},
        license,
        faceCount,
        vertexCount,
        publishedAt
    }' "$metadata" >"$metadata_tmp"
    mv -f -- "$metadata_tmp" "$asset_dir/$piece.metadata.json"
done < <(jq --raw-output '.models[] | [.piece, .uid] | @tsv' "$manifest")

echo "Installed six optimized CC BY Staunton meshes in $asset_dir"
