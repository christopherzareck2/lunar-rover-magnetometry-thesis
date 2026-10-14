# thesis/

This folder holds the figure files of the SM thesis *Lunar Rover Magnetometry: Requirements for Recovering the
Remanent Magnetization of Exposed Mare Bedrock*. The thesis text is not in this repository.

The thesis PDF will be linked from DSpace@MIT (MIT's institutional repository) once it is published there. The link
and the thesis's own citation will be added to the top-level [README](../README.md) and
[CITATION.cff](../CITATION.cff) at that point.

## What is here

| Path | Contents |
|---|---|
| `MIT-thesis-template/figures/` | The thesis figure PDFs, written by `make figures` and committed. The folder keeps the name it has in the thesis's LaTeX source, because `make figures`, the tests and `figures/manifest.json` refer to it by that path |
| `MIT-thesis-template/figures/src/` | The script and plotted values behind Figure 5.2 |

`make figures` redraws every figure from the stored results into `MIT-thesis-template/figures/`.
`python -m lunarmag figures --out DIR` writes them to another folder. Which script draws which figure, and which stored
result it reads, is listed in [docs/reproducing.md](../docs/reproducing.md) and
[docs/thesis-to-code.md](../docs/thesis-to-code.md).
