#!/usr/bin/env bash
set -uo pipefail

repository="${1:-.}"
if ! root="$(git -C "$repository" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "public-release check: not a Git repository: $repository" >&2
  exit 2
fi

violations=()
sensitive_path_pattern='(^|/)(raw([._/-]|$)|snapshot([._/-]|$)|review[-_]packet([._/-]|$)|full[-_]articles?([._/-]|$)|corpus([._/-]|$)|chunks?([._/-]|$)|questions?([._/-]|$)|answers?([._/-]|$)|qrels?([._/-]|$)|candidates?([._/-]|$)|pooling[-_]provenance([._/-]|$))'

if ! git -C "$root" check-ignore -q -- data/local/.public-release-probe; then
  violations+=("data/local is not ignored")
fi

while IFS= read -r -d '' path; do
  lower_path="${path,,}"

  case "$lower_path" in
    data/local|data/local/*)
      violations+=("tracked local artifact: $path")
      ;;
  esac

  case "$lower_path" in
    *.safetensors|*.ckpt|*.onnx|*.gguf|*.pth|*.pt)
      violations+=("tracked model weights: $path")
      ;;
  esac

  case "$lower_path" in
    *.zip|*.7z|*.rar|*.tar|*.tgz|*.tar.gz|*.tbz|*.tbz2|*.tar.bz2|*.txz|*.tar.xz|*.tar.zst|*.gz|*.bz2|*.xz|*.zst|*.jar|*.war|*.ear)
      violations+=("tracked archive: $path")
      ;;
  esac

  case "$lower_path" in
    *.db|*.db3|*.sqlite|*.sqlite3|*.sqlite-wal|*.sqlite-shm|*.duckdb|*.mdb|*.accdb|*.parquet|*.arrow|*.feather|*.orc|*.avro|*.pdf|*.doc|*.docx|*.xls|*.xlsx|*.ppt|*.pptx|*.odt|*.ods|*.odp|*.rtf|*.epub|*.pkl|*.pickle|*.joblib|*.h5|*.hdf5|*.npy|*.npz)
      violations+=("tracked binary/document/database container: $path")
      ;;
  esac

  fns_or_v3=0
  if [[ "$lower_path" =~ (^|[/_.-])fns($|[/_.-]) ]]; then
    fns_or_v3=1
  else
    case "$lower_path" in
      data/eval/*v3*|results/*v3*)
        fns_or_v3=1
        ;;
    esac
  fi

  if [[ "$fns_or_v3" -eq 1 && "$lower_path" =~ $sensitive_path_pattern ]]; then
    violations+=("text-bearing FNS/v3 artifact name: $path")
  fi

  if [[ "$fns_or_v3" -eq 1 ]]; then
    case "$lower_path" in
      *.html|*.htm|*.txt|*.xml)
        violations+=("verbatim-capable FNS/v3 artifact: $path")
        ;;
    esac
  fi
done < <(git -C "$root" ls-files -z)

if ! structured_output="$(python3 - "$root" <<'PY'
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import PurePosixPath


root = sys.argv[1]
secret_keys = {
    "apikey",
    "accesstoken",
    "authtoken",
    "clientsecret",
    "password",
    "passwd",
}
text_keys = {
    "question",
    "questiontext",
    "answer",
    "answertext",
    "rawanswer",
    "title",
    "text",
    "chunktext",
    "sourcetext",
    "sourceanswer",
}
structured_suffixes = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".properties", ".env"}
safe_images = {
    "docs/article-reranking-v3.png": "9ff4a69e33868af52dafaa9cbbfef7ede6ae33cbdbf419008bb7c3f163450f40",
    "docs/article-retrieval.png": "77407c43a841d6fd7ab485b2d679048b81fa25aac47ac7164e73428e3fb02a83",
    "docs/article-retrieval-v3.png": "f076df119bfb31418816021206066255ffafcc121947d7349bbc61357c8aaf45",
    "docs/author-seal.png": "2694c92dca65dab46aed7e29baf3d5129f113ac6afb7b922b662fc67ba564f0b",
    "docs/cover.png": "d141abc22f9a920fbb4391c50b630fea964bce76a4ea95c57c53a2241e39cb77",
    "docs/telegram-plane.png": "c69b620dd09565002a578893a8beea320ae9d5e962c029eae9b35fd77e8cf55f",
}
safe_svgs = {
    "docs/article-reranking-v3.svg": {
        "sha256": "b034797fbb775577bec138f135eae292d227e71cd9069695db7b054523e56aae",
        "hrefs": set(),
    },
    "docs/article-retrieval-v3.svg": {
        "sha256": "12136f5e17e4ef40af9959358c3cf96174540cea8716bb814450145dc0c64027",
        "hrefs": {"author-seal.png", "telegram-plane.png"},
    },
    "docs/article-retrieval.svg": {
        "sha256": "891857e5debd8aaf1d435f3927627b2458700faae403b609d0847f473c9c8575",
        "hrefs": {"author-seal.png", "telegram-plane.png"},
    },
    "docs/results.svg": {
        "sha256": "b19292c48b903c29999b7d95e8651595b5bd44f57eecd99ad008dbda0ff74cb8",
        "hrefs": set(),
    },
}
known_secret = re.compile(
    r"AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{50,}|sk-[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}",
    re.IGNORECASE,
)
private_key = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
fns_url = re.compile(r"https?://(?:[A-Za-z0-9-]+\.)*nalog\.gov\.ru(?:[/:]|$)", re.IGNORECASE)
raw_fns_marker = re.compile(
    r"nalog\.gov\.ru|raw[_-]?answer|cited[_-]?source|question[_-]?text|"
    r"chunk[_-]?text|source[_-]?answer|fns-[0-9a-f]{16,}",
    re.IGNORECASE,
)
generic_assignment = re.compile(
    r"(?im)(?:^|[,{])\s*[\"']?(api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|password|passwd)[\"']?\s*[:=]\s*"
    r"[\"']?([A-Za-z0-9][A-Za-z0-9_./+=:-]{11,})"
)
text_field = re.compile(
    r"(?im)(?:^|[,{])\s*[\"']?(question|questionText|answer|answerText|rawAnswer|raw_answer|"
    r"title|text|chunkText|sourceText|source_answer)[\"']?\s*:"
)
placeholder = re.compile(
    r"example|placeholder|change[-_]?me|replace[-_]?me|dummy|fake|"
    r"not[-_]?a[-_]?secret|redacted|masked|your[-_]?|sample|test[-_]?only",
    re.IGNORECASE,
)


def canonical(value):
    return re.sub(r"[-_\s]", "", str(value)).casefold()


def looks_secret(value):
    if not isinstance(value, (str, int, float)):
        return False
    candidate = str(value).strip()
    lowered = candidate.casefold()
    if len(candidate) < 12:
        return False
    if placeholder.search(candidate):
        return False
    if "://" in candidate or "${" in candidate or "{{" in candidate:
        return False
    if "getenv" in lowered or lowered.startswith(("env.", "env:", "process.env", "system.getenv")):
        return False
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./+=:-]{11,}", candidate) is not None


def has_archive_magic(content):
    signatures = (
        b"PK\x03\x04",
        b"PK\x05\x06",
        b"PK\x07\x08",
        b"\x1f\x8b",
        b"BZh",
        b"\xfd7zXZ\x00",
        b"7z\xbc\xaf\x27\x1c",
        b"Rar!\x1a\x07",
        b"\x28\xb5\x2f\xfd",
        b"\x04\x22\x4d\x18",
    )
    return content.startswith(signatures) or (len(content) >= 262 and content[257:262] == b"ustar")


def has_document_or_database_magic(content):
    signatures = (
        b"SQLite format 3\x00",
        b"%PDF-",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
        b"PAR1",
    )
    return content.startswith(signatures)


def is_binary(content):
    if b"\x00" in content:
        return True
    if any(byte < 32 and byte not in (9, 10, 13) for byte in content[:8192]):
        return True
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def parsed_json_documents(path, text):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def path_marks_fns_or_v3(path):
    lowered = path.casefold()
    if re.search(r"(^|[/_.-])fns($|[/_.-])", lowered):
        return True
    return re.search(r"^(data/eval|results)/.*v3", lowered) is not None


def svg_violations(path, content, digest):
    found = set()
    approved = safe_svgs.get(path)
    if approved is None or approved["sha256"] != digest:
        found.add(f"unapproved tracked SVG: {path}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        found.add(f"invalid SVG encoding: {path}")
        return found
    lowered = text.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        found.add(f"active SVG content: {path}")
    if raw_fns_marker.search(text):
        found.add(f"raw FNS text in SVG: {path}")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        found.add(f"invalid SVG XML: {path}")
        return found
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        found.add(f"invalid SVG root: {path}")
    allowed_hrefs = approved["hrefs"] if approved is not None else set()
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].casefold()
        if tag in {"script", "foreignobject", "iframe", "object", "embed", "style"}:
            found.add(f"active SVG content: {path}")
        for attribute, value in element.attrib.items():
            name = attribute.rsplit("}", 1)[-1].casefold()
            candidate = value.strip()
            candidate_lower = candidate.casefold()
            if name.startswith("on"):
                found.add(f"active SVG content: {path}")
            if name == "href":
                if candidate_lower.startswith("data:"):
                    if approved is None:
                        found.add(f"data URI in SVG: {path}")
                elif candidate_lower.startswith(("http:", "https:", "//", "/", "javascript:")):
                    found.add(f"external SVG href: {path}")
                elif candidate not in allowed_hrefs:
                    found.add(f"unapproved SVG href: {path}")
            if name == "style" and re.search(r"url\s*\(|@import|javascript\s*:", value, re.IGNORECASE):
                found.add(f"active SVG content: {path}")
    return found


stage = subprocess.run(
    ["git", "-C", root, "ls-files", "--stage", "-z"],
    check=True,
    stdout=subprocess.PIPE,
).stdout
blobs = {}
violations = set()

for record in stage.split(b"\0"):
    if not record:
        continue
    metadata, encoded_path = record.split(b"\t", 1)
    mode, blob_id, index_stage = metadata.split(b" ", 2)
    if index_stage != b"0":
        raise RuntimeError("unmerged index entry")
    path = encoded_path.decode("utf-8", "surrogateescape")
    suffix = PurePosixPath(path).suffix.casefold()
    if blob_id not in blobs:
        blobs[blob_id] = subprocess.run(
            ["git", "-C", root, "cat-file", "blob", blob_id.decode("ascii")],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    content = blobs[blob_id]

    if has_archive_magic(content):
        violations.add(f"tracked archive: {path}")
        continue
    if has_document_or_database_magic(content):
        violations.add(f"tracked binary/document/database container: {path}")
        continue
    if is_binary(content):
        digest = hashlib.sha256(content).hexdigest()
        if safe_images.get(path) == digest and content.startswith(b"\x89PNG\r\n\x1a\n"):
            continue
        violations.add(f"unapproved tracked binary: {path}")
        continue
    text = content.decode("utf-8")

    if suffix == ".svg":
        violations.update(svg_violations(path, content, hashlib.sha256(content).hexdigest()))

    if known_secret.search(text) or private_key.search(text):
        violations.add(f"secret-like content: {path}")

    is_json = suffix in {".json", ".jsonl"}
    is_csv = suffix == ".csv"
    path_is_fns = path_marks_fns_or_v3(path)
    content_is_fns = fns_url.search(text) is not None
    parsed = []
    if is_json:
        try:
            parsed = parsed_json_documents(path.casefold(), text)
        except (json.JSONDecodeError, TypeError):
            parsed = []

    has_text_field = any(canonical(key) in text_keys for document in parsed for key, _ in walk(document))
    if is_csv:
        try:
            rows = csv.reader(io.StringIO(text))
            has_text_field = any(canonical(header) in text_keys for header in next(rows, []))
        except csv.Error:
            has_text_field = text_field.search(text) is not None
    elif is_json and not parsed:
        has_text_field = text_field.search(text) is not None
    elif not is_json and suffix in structured_suffixes:
        has_text_field = text_field.search(text) is not None

    if (path_is_fns or content_is_fns) and has_text_field:
        violations.add(f"text-bearing field in FNS/v3 artifact: {path}")

    if suffix not in structured_suffixes:
        continue
    structured_secret = any(
        canonical(key) in secret_keys and looks_secret(value)
        for document in parsed
        for key, value in walk(document)
    )
    fallback_secret = any(looks_secret(match.group(2)) for match in generic_assignment.finditer(text))
    if structured_secret or (not parsed and fallback_secret):
        violations.add(f"secret-like content: {path}")

for violation in sorted(violations):
    print(violation)
PY
)"; then
  echo "public-release check: structured scan failed" >&2
  exit 2
fi

if [[ -n "$structured_output" ]]; then
  while IFS= read -r violation; do
    violations+=("$violation")
  done <<< "$structured_output"
fi

if ((${#violations[@]} > 0)); then
  printf '%s\n' "${violations[@]}" | LC_ALL=C sort -u >&2
  exit 1
fi

echo "public-release check: OK"
