# Commercial End-User Licence Agreement (EULA) Template

*This is a customizable legal template for licensing Private Analyst AI to corporate customers. Have legal counsel review and adapt it to your jurisdiction and commercial terms before signing.*

---

## COMMERCIAL SOFTWARE LICENCE AGREEMENT

**BETWEEN:**

1. **LICENSOR**: [Your Company / Legal Name], with offices at [Your Registered Address] ("Licensor").
2. **CUSTOMER**: [Customer Corporate Name], a company registered in [Jurisdiction] with offices at [Customer Registered Address] ("Licensee").

**EFFECTIVE DATE**: [Date]

---

### 1. DEFINITIONS

- **"Software"**: The Private Analyst AI software application (object code, scripts, and documentation) provided by Licensor.
- **"Deployment Mode"**: Either (a) Standalone Desktop Installation on Licensee-owned workstations, or (b) Private Server Installation hosted on Licensee-owned private infrastructure.
- **"Customer Data"**: All datasets, spreadsheets, documents, queries, prompts, and analytical outputs processed by the Software.
- **"Third-Party Open Source"**: Open-source libraries and runtimes listed in the Software's Licence Inventory, each governed by its respective open-source licence.

---

### 2. GRANT OF LICENCE

2.1. **Scope of Grant**: Subject to the terms of this Agreement, Licensor grants to Licensee a non-exclusive, non-transferable, terminable licence to install and run the Software solely on Licensee-owned or controlled computers/servers for Licensee's internal business data analysis.

2.2. **Authorised Instances**:
- **Desktop Edition**: Licensed for up to [Number] designated workstations.
- **Server Edition**: Licensed for [Number] private server instances for up to [Number] authorized internal users within Licensee’s organization.

2.3. **Restrictions**: Licensee shall not:
- Sublicense, resell, rent, or lease the Software to third parties;
- Modify, reverse engineer, decompile, or disassemble proprietary portions of the Software, except as permitted by applicable law;
- Remove or obscure any proprietary copyright notices or trademarks.

---

### 3. CUSTOMER DATA OWNERSHIP & PRIVACY GUARANTEE

3.1. **Absolute Data Ownership**: Licensee retains all right, title, and interest in and to all Customer Data. Licensor acquires no rights or licences to Customer Data.

3.2. **Offline & Air-Gapped Architecture**:
- The Software does **not** transmit Customer Data, prompts, file metadata, or analytical results to Licensor, vendor servers, or external third-party cloud AI APIs.
- The Software is designed to execute locally on localhost (`127.0.0.1`) without requiring continuous internet connectivity.
- Software telemetry, remote analytics, and crash-reporting services are disabled.

3.3. **Right to Audit Network Isolation**: Licensee is expressly permitted to audit, firewall, and verify the network isolation of the Software using operating system firewalls, packet capture tools, or air-gapped network testing.

---

### 4. LOCAL AI MODELS & STATISTICAL OUTPUT DISCLAIMER

4.1. **Model Weight Distribution**: The Software does not bundle third-party machine learning model weights. Any third-party model pulled or operated via local runtimes (such as Ollama) is subject to that model's own licence terms and is the responsibility of Licensee to license and procure.

4.2. **Statistical Nature of Output**:
- Licensee acknowledges that machine learning models, statistical forecasts, and outlier detection routines generate probabilistic estimates and hypotheses, **not** definitive ground-truth facts.
- The Software is an analytical productivity aid and is not intended to substitute for human judgment, professional financial advice, medical diagnosis, or certified accounting audits.

---

### 5. SUPPORT, UPDATES & MAINTENANCE

5.1. **Offline Updates**: Any updates, patches, or bug fixes shall be provided as standalone downloadable archives or packages. No automated background updates shall be applied without Licensee's manual execution and consent.

5.2. **Technical Support**: Licensor shall provide technical support in accordance with the agreed Service Level Agreement (SLA) via [Support Email / Portal].

---

### 6. WARRANTIES & LIMITATION OF LIABILITY

6.1. **Warranty**: Licensor warrants that the Software will perform substantially in accordance with the accompanying documentation under normal local operational conditions.

6.2. **Disclaimer**: EXCEPT AS EXPRESSLY PROVIDED HEREIN, THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED.

6.3. **Limitation of Liability**: IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, OR PUNITIVE DAMAGES, OR FOR LOSS OF PROFITS, DATA, OR BUSINESS INTERRUPTION, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. LICENSOR’S AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL BE LIMITED TO THE TOTAL FEES PAID BY LICENSEE IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

---

### 7. GOVERNING LAW & SIGNATURES

This Agreement shall be governed by and construed in accordance with the laws of [Jurisdiction].

**LICENSOR:**  
Name: _______________________________  
Title: ________________________________  
Signature: ____________________________  
Date: ________________________________  

**LICENSEE:**  
Name: _______________________________  
Title: ________________________________  
Signature: ____________________________  
Date: ________________________________  
