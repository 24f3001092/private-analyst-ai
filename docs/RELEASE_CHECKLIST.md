# Release checklist

Before you give a build to a customer:

- [ ] Version in `src/private_analyst/version.py` updated
- [ ] `python -m pytest` passes on a clean Windows machine
- [ ] Sample CSV totals still match the documented expected values
- [ ] Streamlit usage stats confirmed off
- [ ] `requirements.txt` has no cloud-AI or telemetry packages
- [ ] `docs/LICENCE_INVENTORY.md` matches pinned versions
- [ ] No passwords, keys, or customer files in the zip
- [ ] Desktop and server start scripts run
- [ ] Offline checklist completed once per major version
- [ ] Known limitations document shipped with the build
- [ ] Customer written licence (your EULA) attached
- [ ] Model bundle (if any) has a verified licence and checksum

Packaging with PyInstaller is **planned**, not completed in 0.1.0. Until then, ship source + venv instructions or a wheelhouse.
