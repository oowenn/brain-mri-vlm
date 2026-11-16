# Midpoint report

This directory contains the written checkpoint report for COMS 6998 Team 32. The report follows the sections in the supplied requirement screenshot. The course separately asks for a PowerPoint-template submission; these Word/PDF files are the written report and do not constitute a submitted slide deck.

## Files

| File | Purpose |
| --- | --- |
| `report_template.md` | Editable prose and tables with named slots for verified experimental results. |
| `build_report.py` | Reads an explicit Git evidence snapshot, requires verified Step 8 results, fills the numerical tables and creates the Word document. It does not run training or inference. |
| `report.md` | Generated readable report source, including the filled tables and reference links. |
| `build-record.json` | Evidence commit, hashes of numerical source files, generation time and Word/source hashes. |
| `verification.json` | Passed final document/PDF checks, nine-page visual review, 28-link comparison, source/evidence checks and artifact hashes. |
| `../../output/docx/brain-mri-vlm-midpoint-report.docx` | Editable Word version. |
| `../../output/pdf/brain-mri-vlm-midpoint-report.pdf` | PDF exported from the same Word document. |

The builder reads the committed Step 5/6 summaries and sweep files plus the verified Step 8 summary. Historical evidence references use the supplied full Git commit, while executed source commits remain in each run directory. Model/method references point to original papers and official versioned documentation. The proposal is the repository's `COMS 6998 Proposal (5).pdf`.

## Rebuild and review

Use a document-generation environment containing Python and `python-docx==1.2.0`, separate from the cluster experiment environment. The original authoring used Codex's bundled Python/document dependencies and bundled LibreOffice. No cluster packages were changed for the report.

~~~bash
python reports/midpoint/build_report.py --evidence-commit <verified-evidence-commit>
~~~

Read the desired commit from `build-record.json`. Export the generated Word file to PDF using LibreOffice or Word, then visually inspect every page. The original review used the document skill's `render_docx.py --emit_pdf` to produce both a PDF and per-page PNGs; the final PDF is copied to `output/pdf/`. Inspect numeric tables, page breaks, citations and hyperlinks after any edits. Intermediate renders remain in ignored `tmp/midpoint/`.

Update the template rather than manually changing generated `report.md` when rebuilding. The Word file can also be edited directly for submission, but direct edits are not reflected in the template; re-export the PDF after such edits. Refresh `verification.json` and hashes for any revised deliverables.
