# Offline Verification & Network Isolation Guide

A successful offline demo does **not** by itself prove that no code path can ever transmit data. This guide provides a dual-layer verification protocol:
1. **Automated / Unit-Level Isolation** (software mocks).
2. **Operating System & Network-Level Verification** (real Windows Defender Firewall, socket inspection, packet isolation).

---

## 1. Unit Test Socket Blocking vs. Real Operating System Verification

> [!IMPORTANT]
> **Understanding the Boundary:**
> - Unit tests (`tests/test_backup_offline.py`, `tests/test_security.py`) monkeypatch Python's internal `socket.socket.connect` to raise `OSError` on non-loopback addresses (`127.0.0.1`, `::1`).
> - While this proves our application-level code does not attempt external network calls during automated test suites, it does **not** replace physical network disconnection or operating system firewall packet filtering.
> - For commercial customer acceptance, you must execute both the unit verification and the OS firewall verification outlined below.

---

## 2. Expected Local Loopback Listeners

During routine operation, only the following local loopback listeners should exist:

| Service | Host Binding | Port | Protocol | Purpose |
| --- | --- | --- | --- | --- |
| **Streamlit Web UI** | `127.0.0.1` | `8501` | TCP (HTTP/WS) | Interactive local analyst dashboard |
| **Ollama AI Runtime** | `127.0.0.1` | `11434` | TCP (HTTP REST) | Local-only LLM inference engine |

Neither service should ever be bound to `0.0.0.0` or a public IP address on an untrusted network.

---

## 3. Ollama Hardened Offline Configuration

If using the optional Ollama local LLM runtime, configure the following environment variables on the host system to eliminate telemetry and prevent network exposure:

```powershell
# Set in system environment or start script before launching Ollama:
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_ORIGINS = "http://127.0.0.1:*,http://localhost:*"
$env:OLLAMA_NO_TELEMETRY = "1"
```

- **`OLLAMA_HOST=127.0.0.1:11434`**: Forces Ollama to bind strictly to loopback, refusing remote connections from the LAN.
- **`OLLAMA_ORIGINS`**: Restricts Cross-Origin Resource Sharing (CORS) strictly to localhost Streamlit sessions.
- **`OLLAMA_NO_TELEMETRY=1`**: Permanently disables Ollama's telemetry reporting to external servers.

---

## 4. Operating System Firewall Configuration (Windows Defender)

To provide cryptographic and network-level assurance for customer audits, configure Windows Defender Firewall to block all outbound traffic from the application and model processes:

### A. Create Outbound Blocking Rules via PowerShell (Run as Administrator)

```powershell
# 1. Block outbound internet access for the Python interpreter
New-NetFirewallRule -DisplayName "PrivateAnalyst - Block Python Outbound" `
    -Direction Outbound `
    -Program "C:\Users\<Username>\AppData\Local\Programs\Python\Python313\python.exe" `
    -Action Block `
    -Profile Any

# 2. Block outbound internet access for Ollama runtime
New-NetFirewallRule -DisplayName "PrivateAnalyst - Block Ollama Outbound" `
    -Direction Outbound `
    -Program "C:\Users\<Username>\AppData\Local\Programs\Ollama\ollama.exe" `
    -Action Block `
    -Profile Any

# Note: Windows loopback traffic (127.0.0.1) bypasses external outbound firewall rules,
# allowing the local Streamlit UI and local Ollama process to communicate seamlessly.
```

### B. Verify Active Listeners and Network Isolation

Run the following PowerShell commands while Private Analyst AI and Ollama are running:

```powershell
# Inspect all listening ports for python and ollama
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalAddress -eq "127.0.0.1" } | Select-Object LocalAddress, LocalPort, OwningProcess

# Verify that external connectivity fails completely
Test-NetConnection -ComputerName 8.8.8.8 -Port 53
Test-NetConnection -ComputerName google.com -Port 443
```

Confirm that neither `python.exe` nor `ollama.exe` has any active TCP connections to remote foreign IP addresses.

---

## 5. Manual Customer Acceptance Workflow (Air-Gapped / Disconnected)

1. Disconnect all network interfaces: disable Wi-Fi and unplug Ethernet (or enable Airplane Mode).
2. Launch `scripts\run_desktop.bat` (or `scripts\run_server.bat`).
3. Import `samples\sales_known.csv`.
4. Verify Dashboard metrics:
   - Units sum = **50**
   - Mean = **10**
   - Median = **10**
   - Min = **0**, Max = **20**
5. Verify Data Quality: Status missing count = **1**.
6. Execute SQL Console query: `SELECT Region, SUM(Units) as TotalUnits FROM sales_known GROUP BY Region ORDER BY TotalUnits DESC`.
7. Index `samples\notes.txt` in the Documents tab; search for `Widget`.
8. Export PDF and CSV reports and confirm they are saved directly to local disk.
9. Verify Resource Monitor (`resmon.exe`) -> Network tab: Confirm 0 bytes/sec external network send/receive.
