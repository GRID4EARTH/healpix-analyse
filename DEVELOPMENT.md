# Development workflow

This document describes the lightweight development workflow used by `healpix-analyse`. It is
intended to support both regular contributions and coordinated work across GRID4EARTH
repositories without introducing a complex branching model.

GRID4EARTH development is primarily supported by ESA-funded activities, and external
contributions are welcome.

## Branches

- `main` is stable and release-ready. It depends only on released versions of other packages.
- Feature branches and pull requests contain individual developments. They should be focused
  and linked to the relevant issue where possible. Start them from `main`.
- Releases are tagged from `main`.

```text
feature branch ──> PR to main ──> main ──> release tag
```

## Cross-repository development

Some `healpix-analyse` changes need unreleased functionality from `healpix-geo` or another
GRID4EARTH repository. In that situation:

1. Develop the change on a focused feature branch and open the pull request to `main`.
2. Pin the development dependency in `pixi.toml` to the exact commit of the other repository
   that contains the needed change, and say so in the pull request.
3. Run the repository tests against that commit.
4. Once the other repository has released the change, replace the pin with the released version
   (and a lower bound in `pyproject.toml`), run the tests again, then merge.

A pull request that still pins an unreleased commit is not merged into `main`.

## Continuous integration and releases

Tests and documentation builds run for pull requests targeting `main` and for pushes to `main`.
Deployment remains deliberately narrower:

- GitHub Pages is deployed only from a push to `main`.
- PyPI publishing occurs only from a published GitHub release.
- Releases and Zenodo archives are created from tags on `main`.

## Authorship and credit

Contributions should retain accurate authorship through commits and pull requests. Project
documentation and release records should acknowledge contributors where appropriate.

The creators used for Zenodo release records are maintained in [`.zenodo.json`](.zenodo.json).
Before tagging a release, review this metadata against the Git history, including
`Co-authored-by` trailers, and against the previous Zenodo record. Add ORCID identifiers only
after verifying them against a source controlled by the contributor. The release tag supplies
the version to Zenodo, so the version is not stored in `.zenodo.json`.

Core developers are listed first in the agreed contribution order. Other creators are listed
alphabetically by family name. Changes to the core developer list or creator order require
maintainer agreement; do not reorder the list mechanically by commit count or lines changed.

Creator status reflects either ongoing core-development responsibility or a substantial
contribution to the software's design, implementation, or documentation. More limited
operational or maintenance contributions should still be credited in the Zenodo `contributors`
list with an appropriate role.
