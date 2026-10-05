# Documentation

- `QuDPy_FDGF_API_documentation.pdf`: reference manual of the solver. It describes the model
  contract, the solver and its methods, the backends and their options, the observables, the
  pathway conventions and a Quick start with its expected output. It is also distributed as
  supplementary material of the article.
- `QuDPy_FDGF_API_documentation.tex`: LaTeX source (self-contained, no figures or bibliography).
  To rebuild the PDF, run `pdflatex` twice, for example
  `pdflatex QuDPy_FDGF_API_documentation.tex`. It needs the packages `physics`, `bbold`, `amsmath`,
  `amssymb`, `booktabs` and `hyperref`.

The source of the manual is maintained together with the manuscript; if the two differ, the
version in the article working folder is the reference.
