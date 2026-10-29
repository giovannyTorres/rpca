# Publishing rpca-kit

The distribution name on PyPI is `rpca-kit`; the importable module is `rpca_kit`.
Use a new version for every release. PyPI distribution files cannot be replaced
with different bytes under an already published version.

## Validate the intended source

Start from a clean checkout of the exact commit to release. Update the version
consistently in package metadata and the module's `__version__`, then run:

```bash
python -m pip install -e '.[dev]'
python scripts/release.py
python examples/readme_quickstart.py
python examples/readme_sensors.py
```

The release script runs tests and lint checks, builds in a temporary directory,
and validates metadata with Twine. It copies the validated wheel and source
distribution to `dist/`. Without `--publish`, it does not upload anything.

For PyTorch CPU coverage, install the intended CPU build and rerun the tests.
CUDA checks require compatible GPU hardware. Report skipped tests as skipped;
a passing CPU suite does not establish CUDA correctness or performance.

## Check an isolated installation

Install the wheel into a separate environment and run outside the checkout:

```bash
python -m venv /tmp/rpca-wheel-test
/tmp/rpca-wheel-test/bin/python -m pip install dist/rpca_kit-0.1.1-py3-none-any.whl
cd /tmp
/tmp/rpca-wheel-test/bin/python -c 'import rpca_kit; print(rpca_kit.__version__)'
/tmp/rpca-wheel-test/bin/python -m pip check
```

Also run the README examples with that interpreter using their absolute paths.
The source distribution can be tested separately with `pip install` on its
`.tar.gz` file. Check that the wheel includes the package, license, and `py.typed`.

## Publish with the existing GitHub Trusted Publisher

The authorized publisher for this project is scoped to:

| Field | Value |
| --- | --- |
| PyPI account | `GiovannyTorres674` |
| PyPI project | `rpca-kit` |
| GitHub owner | `giovannyTorres` |
| Repository | `rpca` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

The workflow lives at
[.github/workflows/publish.yml](https://github.com/giovannyTorres/rpca/blob/main/.github/workflows/publish.yml).
It is triggered manually through
[GitHub Actions](https://github.com/giovannyTorres/rpca/actions/workflows/publish.yml).
Select the intended revision, confirm the commit and version, and run **Publish
to PyPI**. The build job runs NumPy and PyTorch CPU tests, lint, build, and Twine
checks. Only after the build succeeds does the `pypi` environment job publish.

Trusted Publishing exchanges GitHub's OIDC identity for a short-lived publishing
credential. No long-lived PyPI token belongs in the repository, source archive,
chat, or workflow logs. The workflow file alone does not grant publication
rights; its account-side publisher configuration must match the table above.

If the job is waiting for an environment review, obtain the required approval.
If it fails, inspect the actual error before retrying. For `invalid-publisher`,
check the account/project configuration and exact workflow/environment identity;
do not generate a new API token as an automatic workaround. Changes to trusted
publishers or credentials require the project owner's authorization.

Official guides:
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) and
[troubleshooting](https://docs.pypi.org/trusted-publishers/troubleshooting/).

## Verify the published release

A successful build or GitHub artifact is not proof of a PyPI upload. Confirm
that [the PyPI project](https://pypi.org/project/rpca-kit/) shows the expected
version and both the wheel and source distribution. Then install that exact
version from PyPI into a new environment:

```bash
python -m venv /tmp/rpca-pypi-test
/tmp/rpca-pypi-test/bin/python -m pip install --index-url https://pypi.org/simple rpca-kit==0.1.1
/tmp/rpca-pypi-test/bin/python -c 'import rpca_kit; print(rpca_kit.__version__)'
/tmp/rpca-pypi-test/bin/python -m pip check
```

Run the README examples from outside the source checkout using this interpreter.
Only after registry readback and installation checks should release notes state
that the new version is available on PyPI.
