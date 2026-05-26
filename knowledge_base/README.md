# Review Evidence Knowledge Base

The prediction model is trained on the UCI Statlog German Credit Data dataset,
which describes European consumer-credit applications in Deutsche Mark values.
The retrieval corpus therefore uses official European consumer-credit guidance,
not India-specific rules.

## Indexed PDF Sources

| File | Authority | Why it is relevant |
| --- | --- | --- |
| `pdfs/EBA_GL_2020_06_Loan_Origination_and_Monitoring.pdf` | European Banking Authority | Credit granting, borrower creditworthiness, required evidence, model governance, and human credit decision-making. |
| `pdfs/Directive_EU_2023_2225_Consumer_Credit.pdf` | European Parliament and Council of the European Union | Consumer creditworthiness assessment, verifiable financial information, automated processing, and the right to human intervention. |

Metadata and official download URLs are recorded in `sources.json`. Retrieved
chunks include PDF title, issuing authority, page number, and official URL so
review notes can be audited against the source material.

Only relevant pages identified in `sources.json` are indexed. This keeps
retrieval focused on creditworthiness assessment, required information,
automated processing and human review rather than unrelated passages in the
long regulatory documents.

## Dataset Provenance And Limitation

Training-data provenance: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog).

The source dataset includes attributes encoding age, personal status and sex,
and foreign-worker status. Their presence in a historical teaching dataset does
not make them suitable for production lending decisions. A real deployment
would require legal assessment, feature exclusion or strict controls, and
formal fairness testing before model use.
