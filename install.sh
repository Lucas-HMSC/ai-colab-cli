#!/usr/bin/env bash
set -euo pipefail
if ! command -v python3 >/dev/null; then echo 'Python 3 obrigatório.' >&2; exit 1; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HOME/.local/share/colab-ai"
BIN="$HOME/.local/bin"
mkdir -p "$TARGET/remote" "$BIN"
cp "$ROOT/ai_colab.py" "$TARGET/ai_colab.py"
cp "$ROOT"/remote/*.py "$TARGET/remote/"
# Remover somente módulo e wrappers obsoletos gerenciados por este instalador.
rm -f "$TARGET/remote/colab_ai_cache.py" "$BIN/claude-colab" "$BIN/opencode-colab"
cat > "$BIN/ai-colab" <<'EOF'
#!/usr/bin/env bash
exec python3 "$HOME/.local/share/colab-ai/ai_colab.py" "$@"
EOF
chmod +x "$BIN/ai-colab"
printf 'Instalado! Verifique se ~/.local/bin está no PATH. Execute: ai-colab models\n'
