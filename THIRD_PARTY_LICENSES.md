# Third-party licenses and citations

The `LICENSE` file at the root of this repository is Apache-2.0 and covers the
code, the `linear` checkpoint and the gene maps in `checkpoints/`. Everything
else below keeps the terms of its source. Other checkpoints and the datasets'
source studies are not redistributed under the repository license. The matrices in
`datasets/` are derived tables (selected genes, CPM, pseudonymized IDs), not copies
of the source studies.

| Resource | Kind | License / terms | Citation |
| --- | --- | --- | --- |
| `eva-60m`: Hub [`ScientaLab/eva-rna`](https://huggingface.co/ScientaLab/eva-rna) | Model | Scienta Lab EVA Model License (non-commercial research use, attribution; text in the Hub repository) | “When Transcriptomic Foundation Models Scale: Domain-Focused Pretraining for Drug Development in Immunology and Inflammation”, NeurIPS 2026 |
| `scgpt-human`: Hub [`wanglab/scGPT-human`](https://huggingface.co/wanglab/scGPT-human), from [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | Model | MIT ([upstream `LICENSE`](https://github.com/bowang-lab/scGPT/blob/main/LICENSE)); the Hub repository has no license file | Cui et al., Nat Methods 2024, [doi:10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0) |
| `bulkrnabert`: Hub [`InstaDeepAI/BulkRNABert`](https://huggingface.co/InstaDeepAI/BulkRNABert) at revision `7bcae6ac`, from [instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research) | Model | CC BY-NC-SA 4.0 ([upstream `LICENSE.md`](https://github.com/instadeepai/multiomics-open-research/blob/main/LICENSE.md)); the model card has no license tag | Gélard et al., Proc Mach Learn Res 259:384–400 (ML4H 2025), [proceedings page](https://proceedings.mlr.press/v259/gelard25a.html) |
| [Cell Taxonomy](https://ngdc.cncb.ac.cn/celltaxonomy) (CNCB-NGDC) | Annotation | Free for academic use only | Jiang et al., Nucleic Acids Res 2023, [doi:10.1093/nar/gkac816](https://doi.org/10.1093/nar/gkac816); [PMID 36161321](https://pubmed.ncbi.nlm.nih.gov/36161321/) |
| [Open Targets Platform](https://platform.opentargets.org) 25.12, `known_drug`, clinical phase 3+ | Annotation | CC0 1.0; built on ChEMBL (CC BY-SA 3.0) | Buniello et al., Nucleic Acids Res 2025, [doi:10.1093/nar/gkae1128](https://doi.org/10.1093/nar/gkae1128); Zdrazil et al., Nucleic Acids Res 2024, [doi:10.1093/nar/gkad1004](https://doi.org/10.1093/nar/gkad1004) |
| [Gene Ontology](https://geneontology.org) `goa_human.gaf`, release 2025-10-10 | Annotation | CC BY 4.0 | Ashburner et al., Nat Genet 2000, [doi:10.1038/75556](https://doi.org/10.1038/75556); GO Consortium, Nucleic Acids Res 2026, [doi:10.1093/nar/gkaf1292](https://doi.org/10.1093/nar/gkaf1292) |
| [Reactome](https://reactome.org) release 95, NCBI gene-to-pathway | Annotation | CC0 | Ragueneau et al., Nucleic Acids Res 2026, [doi:10.1093/nar/gkaf1223](https://doi.org/10.1093/nar/gkaf1223) |
| [WikiPathways](https://www.wikipathways.org), 18 immunology pathways | Annotation | CC0 (attribution requested) | Agrawal et al., Nucleic Acids Res 2024, [doi:10.1093/nar/gkad960](https://doi.org/10.1093/nar/gkad960) |
| [GSE54456](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54456) | Data | GEO default | [PMID 24441097](https://pubmed.ncbi.nlm.nih.gov/24441097/) |
| [GSE65408](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65408) | Data | GEO default | [PMID 26072395](https://pubmed.ncbi.nlm.nih.gov/26072395/) |
| [GSE73661](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73661) | Data | GEO default | [PMID 27802155](https://pubmed.ncbi.nlm.nih.gov/27802155/) |
| [GSE74697](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74697) | Data | GEO default | [PMID 27015450](https://pubmed.ncbi.nlm.nih.gov/27015450/); [PMID 27793094](https://pubmed.ncbi.nlm.nih.gov/27793094/) |
| [GSE140900](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140900) | Data | GEO default | [PMID 31838750](https://pubmed.ncbi.nlm.nih.gov/31838750/) |
| [GSE157194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157194) | Data | GEO default | [PMID 32615169](https://pubmed.ncbi.nlm.nih.gov/32615169/) |
| [GSE160501](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE160501) | Data | GEO default | [PMID 33218696](https://pubmed.ncbi.nlm.nih.gov/33218696/) |
| [GSE193309](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193309) | Data | GEO default | [PMID 36496193](https://pubmed.ncbi.nlm.nih.gov/36496193/) |
| [GSE193677](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193677) | Data | GEO default | [PMID 36109152](https://pubmed.ncbi.nlm.nih.gov/36109152/) |
| [GSE198520](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198520) | Data | GEO default | [PMID 35854416](https://pubmed.ncbi.nlm.nih.gov/35854416/) |
| [GSE199046](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE199046) | Data | GEO default | [PMID 35682818](https://pubmed.ncbi.nlm.nih.gov/35682818/) |
| [GSE211132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211132) | Data | GEO default | [PMID 37014697](https://pubmed.ncbi.nlm.nih.gov/37014697/) |
| [GSE213761](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213761) | Data | GEO default | [PMID 36689500](https://pubmed.ncbi.nlm.nih.gov/36689500/) |
| [GSE225731](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225731) | Data | GEO default | [PMID 37507201](https://pubmed.ncbi.nlm.nih.gov/37507201/) |
| [E-MTAB-6141](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-6141) | Data | EBI default | [PMID 31461658](https://pubmed.ncbi.nlm.nih.gov/31461658/); [PMID 30878974](https://pubmed.ncbi.nlm.nih.gov/30878974/) |
| [E-MTAB-7604](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-7604) | Data | EBI default | [PMID 30685385](https://pubmed.ncbi.nlm.nih.gov/30685385/) |
| [E-MTAB-7914](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-7914) | Data | EBI default | |
| [E-MTAB-11611](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11611) | Data | EBI default | [PMID 35589854](https://pubmed.ncbi.nlm.nih.gov/35589854/); [PMID 33485455](https://pubmed.ncbi.nlm.nih.gov/33485455/) |
| [E-MTAB-15225](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-15225) | Data | EBI default | [PMID 38073013](https://pubmed.ncbi.nlm.nih.gov/38073013/) |
| [GSE57945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57945) | Data | GEO default | [PMID 25003194](https://pubmed.ncbi.nlm.nih.gov/25003194/) |
| [GSE62207](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62207) | Data | GEO default | |
| [GSE63755](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63755) | Data | GEO default | [PMID 25500883](https://pubmed.ncbi.nlm.nih.gov/25500883/) |
| [GSE66207](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66207) | Data | GEO default | [PMID 26164662](https://pubmed.ncbi.nlm.nih.gov/26164662/) |
| [GSE67250](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67250) | Data | GEO default | [PMID 26973718](https://pubmed.ncbi.nlm.nih.gov/26973718/) |
| [GSE83645](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE83645) | Data | GEO default | [PMID 29031600](https://pubmed.ncbi.nlm.nih.gov/29031600/) |
| [GSE83687](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE83687) | Data | GEO default | [PMID 28892060](https://pubmed.ncbi.nlm.nih.gov/28892060/) |
| [GSE89725](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89725) | Data | GEO default | |
| [GSE93624](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93624) | Data | GEO default | [PMID 28805827](https://pubmed.ncbi.nlm.nih.gov/28805827/) |
| [GSE101794](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE101794) | Data | GEO default | [PMID 30542108](https://pubmed.ncbi.nlm.nih.gov/30542108/) |
| [GSE102746](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE102746) | Data | GEO default | [PMID 29983891](https://pubmed.ncbi.nlm.nih.gov/29983891/) |
| [GSE103489](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103489) | Data | GEO default | [PMID 29510191](https://pubmed.ncbi.nlm.nih.gov/29510191/) |
| [GSE107871](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107871) | Data | GEO default | [PMID 29273799](https://pubmed.ncbi.nlm.nih.gov/29273799/) |
| [GSE108743](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108743) | Data | GEO default | [PMID 30664808](https://pubmed.ncbi.nlm.nih.gov/30664808/) |
| [GSE114286](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114286) | Data | GEO default | [PMID 30341238](https://pubmed.ncbi.nlm.nih.gov/30341238/) |
| [GSE117405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117405) | Data | GEO default | [PMID 30054515](https://pubmed.ncbi.nlm.nih.gov/30054515/) |
| [GSE117875](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117875) | Data | GEO default | [PMID 30232290](https://pubmed.ncbi.nlm.nih.gov/30232290/) |
| [GSE122334](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122334) | Data | GEO default | [PMID 30700907](https://pubmed.ncbi.nlm.nih.gov/30700907/) |
| [GSE130038](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130038) | Data | GEO default | [PMID 31826065](https://pubmed.ncbi.nlm.nih.gov/31826065/) |
| [GSE134662](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134662) | Data | GEO default | [PMID 31393854](https://pubmed.ncbi.nlm.nih.gov/31393854/) |
| [GSE135170](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135170) | Data | GEO default | [PMID 32556182](https://pubmed.ncbi.nlm.nih.gov/32556182/) |
| [GSE137344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137344) | Data | GEO default | [PMID 32058087](https://pubmed.ncbi.nlm.nih.gov/32058087/) |
| [GSE139179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139179) | Data | GEO default | [PMID 33556098](https://pubmed.ncbi.nlm.nih.gov/33556098/) |
| [GSE139341](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139341) | Data | GEO default | [PMID 31852848](https://pubmed.ncbi.nlm.nih.gov/31852848/) |
| [GSE140227](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140227) | Data | GEO default | |
| [GSE140380](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140380) | Data | GEO default | |
| [GSE141571](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141571) | Data | GEO default | [PMID 32428543](https://pubmed.ncbi.nlm.nih.gov/32428543/) |
| [GSE142582](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142582) | Data | GEO default | [PMID 32381429](https://pubmed.ncbi.nlm.nih.gov/32381429/) |
| [GSE151243](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151243) | Data | GEO default | [PMID 37023004](https://pubmed.ncbi.nlm.nih.gov/37023004/) |
| [GSE151686](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151686) | Data | GEO default | [PMID 32866088](https://pubmed.ncbi.nlm.nih.gov/32866088/) |
| [GSE152999](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152999) | Data | GEO default | [PMID 33431859](https://pubmed.ncbi.nlm.nih.gov/33431859/) |
| [GSE153880](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153880) | Data | GEO default | [PMID 34750102](https://pubmed.ncbi.nlm.nih.gov/34750102/) |
| [GSE154200](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154200) | Data | GEO default | [PMID 38892346](https://pubmed.ncbi.nlm.nih.gov/38892346/); [PMID 37119871](https://pubmed.ncbi.nlm.nih.gov/37119871/) |
| [GSE154773](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154773) | Data | GEO default | [PMID 32853177](https://pubmed.ncbi.nlm.nih.gov/32853177/) |
| [GSE155702](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155702) | Data | GEO default | [PMID 33251217](https://pubmed.ncbi.nlm.nih.gov/33251217/) |
| [GSE157020](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157020) | Data | GEO default | [PMID 36521659](https://pubmed.ncbi.nlm.nih.gov/36521659/) |
| [GSE162550](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162550) | Data | GEO default | |
| [GSE164918](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164918) | Data | GEO default | [PMID 34118489](https://pubmed.ncbi.nlm.nih.gov/34118489/) |
| [GSE165512](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165512) | Data | GEO default | |
| [GSE166861](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166861) | Data | GEO default | [PMID 34164359](https://pubmed.ncbi.nlm.nih.gov/34164359/) |
| [GSE166863](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166863) | Data | GEO default | [PMID 33844442](https://pubmed.ncbi.nlm.nih.gov/33844442/) |
| [GSE168312](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168312) | Data | GEO default | |
| [GSE169146](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169146) | Data | GEO default | [PMID 35668129](https://pubmed.ncbi.nlm.nih.gov/35668129/) |
| [GSE171244](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171244) | Data | GEO default | [PMID 34113837](https://pubmed.ncbi.nlm.nih.gov/34113837/) |
| [GSE172372](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE172372) | Data | GEO default | [PMID 35660024](https://pubmed.ncbi.nlm.nih.gov/35660024/) |
| [GSE176279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176279) | Data | GEO default | |
| [GSE178197](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178197) | Data | GEO default | [PMID 34808239](https://pubmed.ncbi.nlm.nih.gov/34808239/); [PMID 30219209](https://pubmed.ncbi.nlm.nih.gov/30219209/) |
| [GSE178994](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178994) | Data | GEO default | [PMID 38945396](https://pubmed.ncbi.nlm.nih.gov/38945396/) |
| [GSE182424](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182424) | Data | GEO default | |
| [GSE183732](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183732) | Data | GEO default | [PMID 35563374](https://pubmed.ncbi.nlm.nih.gov/35563374/); [PMID 36928592](https://pubmed.ncbi.nlm.nih.gov/36928592/) |
| [GSE186063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186063) | Data | GEO default | [PMID 36536476](https://pubmed.ncbi.nlm.nih.gov/36536476/) |
| [GSE186117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186117) | Data | GEO default | [PMID 36928592](https://pubmed.ncbi.nlm.nih.gov/36928592/) |
| [GSE197698](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197698) | Data | GEO default | [PMID 36356046](https://pubmed.ncbi.nlm.nih.gov/36356046/) |
| [GSE199906](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE199906) | Data | GEO default | [PMID 36655602](https://pubmed.ncbi.nlm.nih.gov/36655602/) |
| [GSE205748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205748) | Data | GEO default | [PMID 37131254](https://pubmed.ncbi.nlm.nih.gov/37131254/); [PMID 37137278](https://pubmed.ncbi.nlm.nih.gov/37137278/) |
| [GSE205960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205960) | Data | GEO default | [PMID 37365156](https://pubmed.ncbi.nlm.nih.gov/37365156/) |
| [GSE209981](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209981) | Data | GEO default | [PMID 36167251](https://pubmed.ncbi.nlm.nih.gov/36167251/) |
| [GSE215068](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE215068) | Data | GEO default | [PMID 36727399](https://pubmed.ncbi.nlm.nih.gov/36727399/) |
| [GSE216645](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE216645) | Data | GEO default | [PMID 38357212](https://pubmed.ncbi.nlm.nih.gov/38357212/) |
| [GSE223799](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223799) | Data | GEO default | [PMID 36889660](https://pubmed.ncbi.nlm.nih.gov/36889660/) |
| [GSE224758](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE224758) | Data | GEO default | |
| [GSE226869](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226869) | Data | GEO default | |
| [GSE227747](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE227747) | Data | GEO default | [PMID 37727374](https://pubmed.ncbi.nlm.nih.gov/37727374/) |
| [GSE230113](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE230113) | Data | GEO default | [PMID 38997284](https://pubmed.ncbi.nlm.nih.gov/38997284/) |
| [GSE230200](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE230200) | Data | GEO default | [PMID 37742913](https://pubmed.ncbi.nlm.nih.gov/37742913/) |
| [GSE231692](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE231692) | Data | GEO default | [PMID 37867940](https://pubmed.ncbi.nlm.nih.gov/37867940/) |
| [GSE233900](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233900) | Data | GEO default | |
| [GSE234736](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE234736) | Data | GEO default | [PMID 37837660](https://pubmed.ncbi.nlm.nih.gov/37837660/) |
| [GSE235570](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235570) | Data | GEO default | [PMID 38598229](https://pubmed.ncbi.nlm.nih.gov/38598229/) |
| [GSE237690](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE237690) | Data | GEO default | [PMID 38358820](https://pubmed.ncbi.nlm.nih.gov/38358820/) |
| [GSE237920](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE237920) | Data | GEO default | |
| [GSE243634](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243634) | Data | GEO default | [PMID 39385749](https://pubmed.ncbi.nlm.nih.gov/39385749/) |
| [GSE245451](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245451) | Data | GEO default | [PMID 38183615](https://pubmed.ncbi.nlm.nih.gov/38183615/); [PMID 38534777](https://pubmed.ncbi.nlm.nih.gov/38534777/) |
| [GSE245608](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245608) | Data | GEO default | [PMID 38375886](https://pubmed.ncbi.nlm.nih.gov/38375886/) |
| [GSE245764](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245764) | Data | GEO default | |
| [GSE247742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE247742) | Data | GEO default | |
| [GSE249027](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249027) | Data | GEO default | [PMID 38069342](https://pubmed.ncbi.nlm.nih.gov/38069342/) |
| [GSE249936](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249936) | Data | GEO default | [PMID 38410434](https://pubmed.ncbi.nlm.nih.gov/38410434/); [PMID 40544015](https://pubmed.ncbi.nlm.nih.gov/40544015/) |
| [GSE261086](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261086) | Data | GEO default | |
| [GSE267132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267132) | Data | GEO default | [PMID 39391813](https://pubmed.ncbi.nlm.nih.gov/39391813/) |

## Terms referred to above

- **GEO default.** A GEO series carries no license of its own. NCBI places no
  restrictions on the use or distribution of GEO data and does not accept data
  when the submitter has requested restrictions. Submitters may still claim
  copyright or other rights, and NCBI cannot grant unrestricted permission.
  No record listed here states a restriction. See the
  [NCBI data usage policy](https://www.ncbi.nlm.nih.gov/home/about/policies/).
- **EBI default.** ArrayExpress and BioStudies records carry no dataset-level
  license. EMBL-EBI places no additional restrictions on use or redistribution
  and expects attribution. See the
  [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use).
