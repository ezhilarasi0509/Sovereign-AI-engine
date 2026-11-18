set -e
python src/make_readme.py
python src/analysis_md.py > /dev/null
python src/cost_sens.py > /dev/null
grep -c "### 4.7\|### 4.8" README.md
