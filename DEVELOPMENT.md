# Development workflow

This document describes the lightweight development workflow used by `healpix-analyse`. It is
intended to support both regular contributions and coordinated work across GRID4EARTH
repositories without introducing a complex branching model.

GRID4EARTH development is primarily supported by ESA-funded activities, and external
contributions are welcome.

## Branches

- `main` is stable and release-ready.
- `integration` is a shared branch for combining and validating unreleased changes before they
  are promoted to `main`. It is used primarily for integration and cross-repository validation,
  not as the default branch for day-to-day feature development.
- Feature branches and pull requests contain individual developments. They should be focused
  and linked to the relevant issue where possible. Start them from `main` by default.
- Releases are tagged from `main`.

The usual path remains a focused feature pull request to `main`. Use `integration` only when a
change or compatible group of changes needs combined or upstream validation before promotion:

```text
feature branch ────────────────────────────────> PR to main ──> main ──> release tag
       \
        └─ when integration validation is needed ─> integration
                                                       ├─> cross-feature validation
                                                       ├─> cross-repository validation
                                                       └─> PR: integration -> main
```

## Cross-repository development

Some `healpix-analyse` changes need unreleased functionality from `healpix-geo`, or must be
tested with another GRID4EARTH repository before they are ready for release. In that situation:

1. Develop the change on a focused feature branch and submit it through the normal pull request
   process.
2. When combined validation is needed, integrate the feature into `healpix-analyse`'s
   `integration` branch after its relevant checks pass.
3. Point the development dependency temporarily to the other repository's `integration` branch
   and document that unreleased dependency in the pull request. For example, test against
   `GRID4EARTH/healpix-geo@integration` when unreleased geometry changes are required.
4. Run the repository tests and the relevant cross-repository workflows. The `integration`
   branch may contain temporary dependency pins for this purpose, but release branches must not.
5. After the combined checks pass, promote the compatible, release-ready changes to `main`
   through an `integration`-to-`main` pull request and the normal review process. Keep unrelated
   or incomplete changes out of that promotion.
6. Tag the release from `main`, then replace temporary branch dependencies with released
   versions.

The `integration` branch is a validation and promotion point. It is not a replacement for
focused feature branches, pull-request review, or releases, and feature branches should not use
it as their base unless the feature specifically depends on unreleased integrated work.

## Continuous integration and releases

Tests and documentation builds run for pull requests targeting `main` or `integration`, and for
pushes to either branch. Deployment remains deliberately narrower:

- GitHub Pages is deployed only from a push to `main`.
- PyPI publishing occurs only from a published GitHub release.
- Releases and Zenodo archives are created from tags on `main`, never from `integration`.

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
