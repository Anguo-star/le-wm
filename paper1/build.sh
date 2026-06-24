#!/usr/bin/env bash
# Build the paper1 PDF. Requires pdflatex + bibtex (texlive-latex-recommended +
# texlive-bibtex-extra is enough). Use --clean to remove intermediates.
set -e
cd "$(dirname "$0")"

if ! command -v latexmk >/dev/null 2>&1 && ! command -v pdflatex >/dev/null 2>&1; then
  if [[ -d "$HOME/.TinyTeX/bin/x86_64-linux" ]]; then
    export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
  fi
fi

if [[ "$1" == "--clean" || "$1" == "-c" ]]; then
  rm -f main.aux main.bbl main.blg main.log main.out main.toc \
        main.fdb_latexmk main.fls main.synctex.gz
  shift
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n "Citation .* undefined|Reference .* undefined|There were undefined references|Undefined control sequence|Fatal error|No file main.bbl" main.log >/tmp/paper1_build_grep.log 2>/dev/null; then
    cat /tmp/paper1_build_grep.log
    echo "ERROR: unresolved LaTeX references/citations or fatal build diagnostics" >&2
    exit 1
  fi
else
  if grep -En "Citation .* undefined|Reference .* undefined|There were undefined references|Undefined control sequence|Fatal error|No file main.bbl" main.log >/tmp/paper1_build_grep.log 2>/dev/null; then
    cat /tmp/paper1_build_grep.log
    echo "ERROR: unresolved LaTeX references/citations or fatal build diagnostics" >&2
    exit 1
  fi
fi

echo
echo "OK: paper1/main.pdf built ($(stat -c '%s' main.pdf 2>/dev/null || stat -f '%z' main.pdf) bytes)"
