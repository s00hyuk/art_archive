# Ground-truth descriptions

Drop one UTF-8 plain-text file per artwork here, named ``{artwork_id}.txt``,
where ``artwork_id`` matches the Kaggle filename stem (e.g.
``Vincent_van_Gogh_42.txt``).

When a ground-truth file exists, ``src/evaluator.py`` computes BLEU, ROUGE-L,
and BERTScore against it. Artworks without a matching file are scored only on
reference-free metrics (color-word ratio, length, latency, cost).

This directory is gitignored except for this README and the ``.gitkeep`` —
ground-truth text is a research artefact you curate yourself.
