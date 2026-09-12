# EMPIAR-11280 Experimental Parameters

## Dataset

- EMPIAR: EMPIAR-11280
- EMDB: EMD-24949
- PDB: 7SAD
- Title: Memantine-bound GluN1a-GluN2B NMDA receptors
- Local input: /home/share/empiar/11280/data/*.tiff
- Local inventory: 3,657 TIFF movies, approximately 774.6 GB
- Each movie contains 30 frames.

## Parameters used as facts

| Field | Value | Evidence |
|---|---:|---|
| Pixel size | 0.856 A/pixel | EMD-24949 / wwPDB validation |
| Accelerating voltage | 300 kV | EMD-24949 / wwPDB validation |
| Total exposure dose | 57.6 e-/A2 | EMD-24949 / wwPDB validation |
| Detector | GATAN K3 BioQuantum | EMD-24949 / wwPDB validation |
| Reported resolution | 3.96 A | EMD-24949 / PDB 7SAD |
| Particles used | 131,384 | EMD-24949 / PDB 7SAD |

## Parameter requiring confirmation

Spherical aberration (Cs) was not found in the checked primary-paper and deposition records. It is therefore left as null in the test payload rather than silently treating the common Titan Krios value of 2.7 mm as an experimental fact. The model should request this input or use an explicitly approved operational default.

## Sources

- EMPIAR entry: https://empiar.pdbj.org/en/entry/11280/
- Primary paper: https://doi.org/10.1038/s41594-022-00772-0
- EMDB/PDB validation record: https://data.pdbj.org/pub/pdb/validation_reports/sa/7sad/7sad_full_validation.pdf
- PDB entry: https://www.rcsb.org/structure/7SAD
