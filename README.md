# Lunar rover magnetometry thesis: code and results

Code, stored results and figures for the SM thesis *Lunar Rover Magnetometry: Requirements for Recovering the
Remanent Magnetization of Exposed Mare Bedrock* by Christopher A. Zareck (Massachusetts Institute of Technology,
Department of Aeronautics and Astronautics; advisor Jeffrey A. Hoffman). The thesis PDF will be linked from
[DSpace@MIT](https://dspace.mit.edu/) once it is published there; the thesis text itself is not in this repository.

This is the thesis repository. It is separate from the code of the companion paper (Narrett, Zareck, Weiss and
Hoffman, manuscript in preparation), which is public at
[christopherzareck2/lunar-bedrock-magnetometry](https://github.com/christopherzareck2/lunar-bedrock-magnetometry),
release [v1.0.0](https://github.com/christopherzareck2/lunar-bedrock-magnetometry/releases/tag/v1.0.0) (commit
`3498ead`). This repository builds on that release: its simulation engine, scenario scripts, stored results and the
paper's reproduction notebook are the starting point, and the changes to them are listed in
[docs/changes-from-paper-release.md](docs/changes-from-paper-release.md).

## What is new in the thesis

| Part | What it adds to the paper's code |
|---|---|
| `studies/` | Studies added for the thesis: reruns for four review findings (Y6-01 to Y6-04, including a fix to the skate-park mesh), a flow-mean simulation for a lava flow of ten outcrops, a Sobol variance attribution of the error budget, the regolith study and a Fisher-information check |
| `ml/` | A learned (amortized) estimator for the isolated outcrop, compared with the least-squares estimators and with the Cramer-Rao and van Trees bounds (thesis Chapter 7) |
| `figures/`, `thesis/` | The thesis figures (Figs. 3.1 to 7.1), with a manifest that records the script, the inputs and the hash of each |
| `src/lunarmag/` | The engine as an installable package, with documented reference implementations of the thesis equations and the figure code |
| `tests/`, `docs/` | A 117-test suite that reruns stored numbers, and documentation of the code, the equations and the known issues |

## What it asks

Mare bedrock that is still in place can preserve the direction of the field it cooled in. That holds only if its
magnetization is a primary thermoremanent magnetization (TRM) free of overprints (shock, alteration, viscous
acquisition, reheating) and not strongly anisotropic, and if the outcrop's orientation since it cooled is known (and, to test how the field changed in time, if the flows
are in known stratigraphic order and independent in age). The thesis takes these as assumptions that a magnetic survey cannot test. Given them, it asks whether a rover-mounted
vector magnetometer, referenced to a stationary lander, can recover the natural remanent magnetization (NRM) of an
exposed outcrop, and what that requires of the instrument, the imaging, the survey and the mission.

## Headline result and its main limit

Everything here is simulation. The simulations recover the direction and intensity of an outcrop's bulk NRM. Turning
that into a paleofield intensity needs a further assumption, the TRM recording efficiency; the direction does not
depend on it.

- **Reference case.** A 4 x 4 x 1 m outcrop with a known imaged shape, magnetized at 2.4e-3 A/m (about a 2 microtesla
  paleofield for the assumed recording efficiency, a deliberately weak case), with white sensor noise of 0.5 nT per
  axis as the only error, and one loop driven around it at 1 m standoff. For each of 42 magnetization directions I
  average the angle between the recovered and the true direction over 4,000 noise draws. Those 42 means lie between
  3.2 and 3.6 degrees.
- **Requirements derived for the modeled scenarios.** With all modeled error sources included, the error budget gives:
  slow drift near 0.01 nT rms about the survey mean with each loop driven both ways, loops centered to 0.1 m, imaging
  errors within 1.5 degrees and 0.075 m, fitted heading terms for sensor tilt, and a measured susceptibility. Under
  these, the mean direction error is 3.0 to 4.9 degrees over six directions with up to 0.5 m of unmodeled burial.
- **Main limits.** The drift level is beyond any rover stability the thesis found demonstrated. The 5 degree goal for
  one outcrop is met as a point estimate but not with statistical confidence, and the goal for a lava flow of ten
  outcrops is not met in simulation. Regolith displaced to 1 m depth around the outcrop biases the direction by
  about 22 degrees in a 100 nT site field at the reference magnetization (less for stronger magnetization), so the
  site must be characterized. Chapters 5 and 9 of the thesis
  and [docs/known-issues.md](docs/known-issues.md) give the rest.

## Quickstart

You need conda.

```bash
conda env create -f environment.yml      # Python 3.10, the pinned SimPEG stack, pytest; installs `lunarmag`
conda activate lunar-mag
make test                                # 117 tests, about 40 s
make figures                             # redraw every thesis figure from the stored results, about 10 s
make reproduce                           # one-report check of the stored results, about 20 s
make help                                # every target
```

These commands establish different things. `make figures` regenerates the figures from stored results.
`make test` checks the code and the stored numbers against each other: seeded parts are rerun and must match, and the
374 numbers quoted from the companion paper are recomputed from the stored results. That is a consistency check, not
an independent replication. `make rerun` runs every simulation again from its seed (3.5 to 4 hours). I have not run
it on the thesis version. [docs/reproducing.md](docs/reproducing.md) explains the three, lists the script behind each
figure, describes the tests and covers troubleshooting.

## Repository map

| Path | Contents |
|---|---|
| `src/lunarmag/` | Python package: simulation engine, reference implementations of the thesis equations, figure code |
| `engine/`, `analysis/` | Scenario and analysis scripts that wrote `results/`, kept as they were run |
| `results/` | Stored results (JSON) |
| `studies/`, `ml/` | Studies added for the thesis, and the learned estimator of Chapter 7. The 350 MB of simulated data in `ml/data/` is not included; `bash ml/run_all.sh` regenerates it |
| `figures/`, `notebook/` | Figure manifest, plotted values and previews; the companion paper's reproduction notebook |
| `tests/` | The pytest suite |
| `thesis/` | The thesis figure PDFs ([notes](thesis/README.md)); the thesis text is not included |
| `docs/` | Documentation, listed below |

## Documentation

- [docs/reproducing.md](docs/reproducing.md): setup, what each check shows, figure-to-script table, tests, troubleshooting
- [docs/thesis-to-code.md](docs/thesis-to-code.md): each figure, table and quoted number mapped to its script and stored result
- [docs/architecture.md](docs/architecture.md) and [docs/theory.md](docs/theory.md): how the code is organized, and equations to code
- [docs/known-issues.md](docs/known-issues.md): where the code and a sentence of the thesis disagree
- [docs/changes-from-paper-release.md](docs/changes-from-paper-release.md): what changed from the companion paper's code release v1.0.0

## Use of AI tools

Much of the code was written with AI coding assistants under my direction (Anthropic's Claude through Claude Code and
Cowork, and the Cursor editor), and the code and derivations were reviewed with Claude and OpenAI's ChatGPT. I am
responsible for the content. The thesis records the use by stage, tool, period and check in its "Use of AI Tools"
section and Appendix D.

## License, citation and disclaimer

**License.** MIT License ([LICENSE](LICENSE)), Copyright (c) 2026 Christopher A. Zareck. The thesis text is not part
of this repository and is distributed separately under the terms of its MIT Libraries deposit.

**Citation.** [CITATION.cff](CITATION.cff) holds the citation. If you use the thesis code, cite the thesis; if you use
results that come from the companion paper, also cite the paper and its code release.

> Zareck, C. A. (2027). *Lunar Rover Magnetometry: Requirements for Recovering the Remanent Magnetization of Exposed
> Mare Bedrock*. SM thesis, Department of Aeronautics and Astronautics, Massachusetts Institute of Technology. In
> preparation.

The views expressed are those of the author and do not necessarily reflect the official policy or position of the
Air Force, Space Force, Department of Defense, or U.S. Government.

Contact: Christopher A. Zareck, czareck2@mit.edu.
