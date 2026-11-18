set -e
python src/make_readme.py
python src/analysis_md.py > /dev/null
python src/cost_sens.py > /dev/null
grep -c "### 4.7\|### 4.8" README.md
printf '\n## Observability screenshot\n\n![Phoenix trace of a /summarize call](docs/phoenix_trace.png)\n' >> README.md
cat docs/license_section.md >> README.md
[ -f docs/banner.md ] && cat docs/banner.md README.md > /tmp/readme_b.md && mv /tmp/readme_b.md README.md
