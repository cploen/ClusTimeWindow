# Cluster Time Window Study

ROOT macros and supporting scripts for the NPS cluster time window (coincidence window) study.

## Layout
- macro/	ROOT macros
- csv/		generated tables (ignored)
- img/ 		generated plots (ignored)
- fig/ 	shared plots
- rootfiles/ 	input ROOT files (ignored)

### Formatting Images to Keep
 - $./make_montage <\stub name-of-file-before-extension>
 - $ convert top.png bottom.png -append stacked.png
 - $ convert clust_migration.png clust_migration.pdf
