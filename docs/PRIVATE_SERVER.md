# Private-server deployment

Use this only on a **customer-owned** machine or VM. There is no hosted backend from the software author.

## Differences from desktop edition

| Topic | Desktop | Server |
| --- | --- | --- |
| Environment | `PRIVATE_ANALYST_EDITION=desktop` | `PRIVATE_ANALYST_EDITION=server` |
| Login | Local owner, no password | Required |
| Bind address | `127.0.0.1` | Still `127.0.0.1` by default |
| HTTPS | Not required on loopback | Required if users connect over a network |
| Isolation | Single operator | `org_id` on users, audit events, saved objects |

## First admin

```powershell
$env:PYTHONPATH = "$pwd\src"
$env:PRIVATE_ANALYST_EDITION = "server"
python scripts\create_admin.py --org customer-a --username admin
```

Then `scripts\run_server.bat`.

## Network exposure

The Streamlit process must **not** be bound to `0.0.0.0` on an untrusted network.

If internal users need access:

1. Keep Streamlit on `127.0.0.1`.
2. Put nginx or IIS in front with TLS certificates the customer controls.
3. Restrict firewall to the internal VLAN.
4. Do not publish Ollama (`11434`) beyond localhost.

HTTPS, certificates, and firewall rules are the customer’s operational responsibility. This MVP does not ship a production reverse-proxy installer.

## Organization data

SQLite lives under the configured data directory. Always pass `org_id` when creating users. Application queries for audit events are filtered by organization. Uploaded tables in the UI remain in the signed-in session (not a shared global table).

## Backup

Use the Python API `create_backup` / `restore_backup` or zip the data directory while the app is stopped. Restore refuses a non-empty destination so it will not silently mix two customers.
