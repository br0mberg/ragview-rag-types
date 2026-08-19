#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
check="$script_dir/check-public-release.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

new_repository() {
  local name="$1"
  local directory="$work/$name"
  git init -q "$directory"
  printf '%s\n' "$directory"
}

expect_failure() {
  local repository="$1"
  local expected="$2"
  local output
  if output="$(bash "$check" "$repository" 2>&1)"; then
    echo "expected failure for $repository" >&2
    exit 1
  fi
  if ! grep -Fq -- "$expected" <<<"$output"; then
    echo "missing diagnostic '$expected' for $repository" >&2
    exit 1
  fi
}

safe="$(new_repository safe)"
mkdir -p "$safe/data/eval" "$safe/results/v3"
printf '%s\n' 'data/local/' > "$safe/.gitignore"
printf '%s\n' '{"sourceUrl":"https://www.nalog.gov.ru/","questionCount":240}' > "$safe/data/eval/v3_source_config.json"
printf '%s\n' 'qid,method,k,hits' 'fns-001,dense,10,1' > "$safe/results/v3/per-query.csv"
git -C "$safe" add .
bash "$check" "$safe" >/dev/null

safe_bge="$(new_repository safe-bge)"
mkdir -p "$safe_bge/results/v3-bge"
printf '%s\n' 'data/local/' > "$safe_bge/.gitignore"
printf '%s\n' \
  '{"qid":"fns-001","targetDocIds":["nk-1-article-1-chunk-001"]}' \
  > "$safe_bge/results/v3-bge/cited-clause-labels.json"
printf '%s\n' \
  '{"qid":"fns-001","bge":[{"docId":"nk-1-article-1-chunk-001","score":1.0}]}' \
  > "$safe_bge/results/v3-bge/bge-ranking.jsonl"
git -C "$safe_bge" add .
bash "$check" "$safe_bge" >/dev/null

missing_ignore="$(new_repository missing-ignore)"
printf '%s\n' 'safe' > "$missing_ignore/README.md"
git -C "$missing_ignore" add .
expect_failure "$missing_ignore" 'data/local is not ignored'

local_data="$(new_repository local-data)"
mkdir -p "$local_data/data/local"
printf '%s\n' '{}' > "$local_data/data/local/raw.json"
git -C "$local_data" add -f .
expect_failure "$local_data" 'tracked local artifact:'

raw_packet="$(new_repository raw-packet)"
mkdir -p "$raw_packet/data/eval/v3"
printf '%s\n' '{}' > "$raw_packet/data/eval/v3/review_packet.json"
git -C "$raw_packet" add .
expect_failure "$raw_packet" 'text-bearing FNS/v3 artifact name:'

text_field="$(new_repository text-field)"
mkdir -p "$text_field/results/v3"
printf '%s\n' '{"qid":"fns-001","question":"verbatim"}' > "$text_field/results/v3/metrics.jsonl"
git -C "$text_field" add .
expect_failure "$text_field" 'text-bearing field in FNS/v3 artifact:'

weights="$(new_repository weights)"
printf '%s\n' 'fixture' > "$weights/model.safetensors"
git -C "$weights" add .
expect_failure "$weights" 'tracked model weights:'

secret="$(new_repository secret)"
credential_name='API_'"KEY"
credential_value='prod_0123456789abcdef'
printf '%s=%s\n' "$credential_name" "$credential_value" > "$secret/config.env"
git -C "$secret" add -f .
expect_failure "$secret" 'secret-like content:'

known_secret="$(new_repository known-secret)"
printf '%s\n' 'data/local/' > "$known_secret/.gitignore"
token_prefix='hf_'
token_body='0123456789abcdefghijklmn'
printf '%s%s\n' "$token_prefix" "$token_body" > "$known_secret/Example.java"
git -C "$known_secret" add .
expect_failure "$known_secret" 'secret-like content:'

json_secret="$(new_repository json-secret)"
printf '%s\n' 'data/local/' > "$json_secret/.gitignore"
printf '%s\n' '{"apiKey":"prod_0123456789abcdef"}' > "$json_secret/settings.json"
git -C "$json_secret" add .
expect_failure "$json_secret" 'secret-like content:'

multiline_secret="$(new_repository multiline-secret)"
printf '%s\n' 'data/local/' > "$multiline_secret/.gitignore"
printf '%s\n' '{' '  "apiKey"' '  :' '  "prod_0123456789abcdef"' '}' > "$multiline_secret/settings.json"
git -C "$multiline_secret" add .
expect_failure "$multiline_secret" 'secret-like content:'

fns_export="$(new_repository fns-export)"
mkdir -p "$fns_export/docs"
printf '%s\n' 'data/local/' > "$fns_export/.gitignore"
printf '%s\n' '{"question":"verbatim question","answer":"verbatim answer","sourceUrl":"https://www.nalog.gov.ru/"}' > "$fns_export/docs/fns_export.json"
git -C "$fns_export" add .
expect_failure "$fns_export" 'text-bearing field in FNS/v3 artifact:'

multiline_json="$(new_repository multiline-json)"
mkdir -p "$multiline_json/docs"
printf '%s\n' 'data/local/' > "$multiline_json/.gitignore"
printf '%s\n' '{' '  "sourceUrl": "https://www.nalog.gov.ru/",' '  "question"' '  :' '  "verbatim question"' '}' > "$multiline_json/docs/export.json"
git -C "$multiline_json" add .
expect_failure "$multiline_json" 'text-bearing field in FNS/v3 artifact:'

archive="$(new_repository archive)"
printf '%s\n' 'data/local/' > "$archive/.gitignore"
printf '%s\n' 'opaque payload' > "$archive/review-materials.zip"
git -C "$archive" add .
expect_failure "$archive" 'tracked archive:'

for extension in db sqlite sqlite3 parquet pdf doc docx xls xlsx; do
  container="$(new_repository "container-$extension")"
  printf '%s\n' 'data/local/' > "$container/.gitignore"
  printf '%s\n' 'opaque payload' > "$container/evidence.$extension"
  git -C "$container" add .
  expect_failure "$container" 'tracked binary/document/database container:'
done

sqlite_payload="$(new_repository sqlite-payload)"
mkdir -p "$sqlite_payload/docs"
printf '%s\n' 'data/local/' > "$sqlite_payload/.gitignore"
printf 'SQLite format 3\000{"question":"verbatim","answer":"verbatim","sourceUrl":"https://www.nalog.gov.ru/"}' > "$sqlite_payload/docs/fns-export.sqlite"
git -C "$sqlite_payload" add .
expect_failure "$sqlite_payload" 'tracked binary/document/database container:'

disguised_archive="$(new_repository disguised-archive)"
printf '%s\n' 'data/local/' > "$disguised_archive/.gitignore"
printf 'PK\003\004hidden payload' > "$disguised_archive/evidence.payload"
git -C "$disguised_archive" add .
expect_failure "$disguised_archive" 'tracked archive:'

unknown_binary="$(new_repository unknown-binary)"
printf '%s\n' 'data/local/' > "$unknown_binary/.gitignore"
printf '\000opaque payload' > "$unknown_binary/evidence.asset"
git -C "$unknown_binary" add .
expect_failure "$unknown_binary" 'unapproved tracked binary:'

unapproved_image="$(new_repository unapproved-image)"
mkdir -p "$unapproved_image/docs"
printf '%s\n' 'data/local/' > "$unapproved_image/.gitignore"
printf '\211PNG\r\n\032\nfixture' > "$unapproved_image/docs/unreviewed.png"
git -C "$unapproved_image" add .
expect_failure "$unapproved_image" 'unapproved tracked binary:'

changed_safe_image="$(new_repository changed-safe-image)"
mkdir -p "$changed_safe_image/docs"
printf '%s\n' 'data/local/' > "$changed_safe_image/.gitignore"
printf '\211PNG\r\n\032\nchanged' > "$changed_safe_image/docs/cover.png"
git -C "$changed_safe_image" add .
expect_failure "$changed_safe_image" 'unapproved tracked binary:'

safe_image="$(new_repository safe-image)"
mkdir -p "$safe_image/docs"
printf '%s\n' 'data/local/' > "$safe_image/.gitignore"
cp "$script_dir/../docs/cover.png" "$safe_image/docs/cover.png"
git -C "$safe_image" add .
bash "$check" "$safe_image" >/dev/null

safe_svg="$(new_repository safe-svg)"
mkdir -p "$safe_svg/docs"
printf '%s\n' 'data/local/' > "$safe_svg/.gitignore"
cp "$script_dir/../docs/article-retrieval-v3.svg" "$safe_svg/docs/article-retrieval-v3.svg"
cp "$script_dir/../docs/author-seal.png" "$safe_svg/docs/author-seal.png"
cp "$script_dir/../docs/telegram-plane.png" "$safe_svg/docs/telegram-plane.png"
git -C "$safe_svg" add .
bash "$check" "$safe_svg" >/dev/null

safe_embedded_svg="$(new_repository safe-embedded-svg)"
mkdir -p "$safe_embedded_svg/docs"
printf '%s\n' 'data/local/' > "$safe_embedded_svg/.gitignore"
cp "$script_dir/../docs/article-reranking-v3.svg" "$safe_embedded_svg/docs/article-reranking-v3.svg"
git -C "$safe_embedded_svg" add .
bash "$check" "$safe_embedded_svg" >/dev/null

unapproved_svg="$(new_repository unapproved-svg)"
mkdir -p "$unapproved_svg/docs"
printf '%s\n' 'data/local/' > "$unapproved_svg/.gitignore"
printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>' > "$unapproved_svg/docs/unreviewed.svg"
git -C "$unapproved_svg" add .
expect_failure "$unapproved_svg" 'unapproved tracked SVG:'

changed_safe_svg="$(new_repository changed-safe-svg)"
mkdir -p "$changed_safe_svg/docs"
printf '%s\n' 'data/local/' > "$changed_safe_svg/.gitignore"
cp "$script_dir/../docs/article-retrieval-v3.svg" "$changed_safe_svg/docs/article-retrieval-v3.svg"
printf '%s\n' ' ' >> "$changed_safe_svg/docs/article-retrieval-v3.svg"
git -C "$changed_safe_svg" add .
expect_failure "$changed_safe_svg" 'unapproved tracked SVG:'

script_svg="$(new_repository script-svg)"
mkdir -p "$script_svg/docs"
printf '%s\n' 'data/local/' > "$script_svg/.gitignore"
printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>' > "$script_svg/docs/payload.svg"
git -C "$script_svg" add .
expect_failure "$script_svg" 'active SVG content:'

external_svg="$(new_repository external-svg)"
mkdir -p "$external_svg/docs"
printf '%s\n' 'data/local/' > "$external_svg/.gitignore"
printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.test/pixel.png"/></svg>' > "$external_svg/docs/payload.svg"
git -C "$external_svg" add .
expect_failure "$external_svg" 'external SVG href:'

data_svg="$(new_repository data-svg)"
mkdir -p "$data_svg/docs"
printf '%s\n' 'data/local/' > "$data_svg/.gitignore"
printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg"><image href="data:image/png;base64,AA=="/></svg>' > "$data_svg/docs/payload.svg"
git -C "$data_svg" add .
expect_failure "$data_svg" 'data URI in SVG:'

fns_text_svg="$(new_repository fns-text-svg)"
mkdir -p "$fns_text_svg/docs"
printf '%s\n' 'data/local/' > "$fns_text_svg/.gitignore"
printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg"><text>questionText: raw FNS answer</text></svg>' > "$fns_text_svg/docs/payload.svg"
git -C "$fns_text_svg" add .
expect_failure "$fns_text_svg" 'raw FNS text in SVG:'

safe_source="$(new_repository safe-source)"
mkdir -p "$safe_source/src/main/java/example" "$safe_source/src/main/resources"
printf '%s\n' 'data/local/' > "$safe_source/.gitignore"
printf '%s\n' 'class Config { String apiKey = System.getenv("SERVICE_API_KEY"); }' > "$safe_source/src/main/java/example/Config.java"
printf '%s\n' 'service:' '  api-key: ${SERVICE_API_KEY}' > "$safe_source/src/main/resources/application.yml"
printf '%s\n' '{"apiKey":"example-placeholder-value"}' > "$safe_source/settings.json"
printf '%s\n' '{"lockfileVersion":3,"packages":{"node_modules/api-key-helper":{"version":"1.0.0","integrity":"sha512-dGVzdA=="}}}' > "$safe_source/package-lock.json"
git -C "$safe_source" add .
bash "$check" "$safe_source" >/dev/null

echo "public-release check tests: OK"
