# Public data behind Fig. 1.1 and Fig. 6.2 (not in git)

Retrieved 2026-10-14 (UTC, between 14:55 and 15:13) with `curl`, from public NASA, USGS and Planetary Data System
(PDS) servers. Only public-domain science data is kept here; nothing is copied from the lroc.asu.edu website images,
which are copyrighted. The figure scripts read only the small reduced products in `figures/inputs/`; these raw
files are needed only to re-run `python figures/scripts/prepare_image_inputs.py {map|coords|nac|apollo}`.
Total on disk about 1.3 GB. `data/external/` is ignored by git except this file.

Credit lines used in the thesis: LROC images "NASA/GSFC/Arizona State University"; Apollo frame "NASA"; LOLA map
"NASA/GSFC"; feature coordinates "IAU Gazetteer of Planetary Nomenclature, USGS Astrogeology Science Center".
Terms: LROC's terms page (https://lroc.im-ldi.com/about/terms) says the LROC data products delivered to the PDS are
public domain and the images posted on the LROC website are copyrighted, so only PDS products are used. NASA
imagery is generally not copyrighted and asks for credit (https://www.nasa.gov/multimedia/guidelines/index.html).

| File (under data/external/) | Source URL | ID | Size (bytes) | sha256 | Used for |
|---|---|---|---|---|---|
| `lroc/M113941548LE.IMG` | https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/DATA/MAP/2009331/NAC/M113941548LE.IMG | LROC NAC EDR M113941548LE (nacl00015194), 2009-11-27, volume LROLRC_0001 | 264,467,400 | `a07d9ad9c20cda30b950920fab63f9afbc28d758ca4da79e627907982e4f896f` | Fig. 1.1a |
| `lroc/M1345757107LE.IMG` | https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0043C/DATA/ESM4/2020154/NAC/M1345757107LE.IMG | LROC NAC EDR M1345757107LE (nacl00241914), 2020-06-02, volume LROLRC_0043C | 264,467,400 | `be0c3b3e47f0063989b1dde1ee496101b8890c64af658f4a93e9c62f05880198` | Fig. 1.1b |
| `lroc/M1317564208LE.IMG` | https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0040A/DATA/ESM3/2019193/NAC/M1317564208LE.IMG | LROC NAC EDR M1317564208LE, 2019-07-12, volume LROLRC_0040A | 264,467,400 | `1328e79da4f50d4598e53be398e8dc09d06ac2e0631ca3aee5da99862f4300fe` | not used: another view of the same pit, downloaded to compare with M1345757107L (not clearer) |
| `lroc/INDEX_0001.LBL`, `INDEX_0043C.LBL` | `.../LROLRC_0001/INDEX/INDEX.LBL`, `.../LROLRC_0043C/INDEX/INDEX.LBL` (same server and path pattern) | PDS index labels (column definitions) | 31,645; 31,613 | | geometry columns |
| `lroc/row_*.txt` | rows for the product IDs copied from `.../INDEX/INDEX.TAB` of volumes LROLRC_0001, 0037C, 0040A, 0043A, 0043C | one line per EDR: resolution, incidence, north and sub-solar azimuth, corner latitudes | about 900 each | | pixel scale, sun and north directions |
| `lola/LDEM_64.IMG`, `LDEM_64.LBL` | https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/LDEM_64.IMG (and `.LBL`) | LOLA GDR LDEM_64, V3.1 (LRO-L-LOLA-4-GDR-V1.0), 64 pixel/degree, created 2019-03-15, PI D. E. Smith, GSFC | 530,841,600 (IMG); 5,128 (LBL) | `1c4958699d4cffd7e777d51421b6044bba300aa0bef286a544309efb082cdbd6` (IMG) | Fig. 6.2 hillshade |
| `gazetteer/MOON_nomenclature_center_pts.zip` | https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MOON_nomenclature_center_pts.zip (linked from https://planetarynames.wr.usgs.gov/Page/MOON/target) | IAU Gazetteer of Planetary Nomenclature, Moon, center points | 23,845,070 | `8670bd8beea6adf1cee8739b272182293d21ae5867160388dadccca733d25204` | Fig. 6.2 coordinates |
| `apollo/a15_h_85_11451.gif`, `a15_h_85_11451_caption.html` | https://nssdc.gsfc.nasa.gov/imgcat/hires/a15_h_85_11451.gif ; caption page https://nssdc.gsfc.nasa.gov/imgcat/html/ency_captions/a15_h_85_11451.html | Apollo 15 Hasselblad frame AS15-85-11451 (NSSDC scan, 648 x 590) | 312,205 | `a1062dcc277066c04a0ed373498327690974ba33564a07ae41fe5b2f8736d795` | Fig. 1.1c |

Gazetteer features used (IDs from the Gazetteer links): Rimae Doppelmayer 5104, Rima Cauchy 5049, Rima Oppolzer 5075,
Rimae Hypatia 5115, Rima Hadley 5064, and the maria used as labels (feature IDs in `figures/inputs/site_coordinates.csv`).
Blue Ghost Mission 1 is not in the Gazetteer; 18.56 N, 61.81 E is the center of its target landing region on the LROC
page https://www.lroc.im-ldi.com/images/1400, not a surveyed touchdown point.

Not used: the WAC 32 pixel/degree global mosaic (https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/WAC_GLOBAL_E000N0000_032P.IMG,
265 MB) exists and would also have worked; the LOLA hillshade was used instead. No magnetic-field model was downloaded.

Rate-limit note: a burst of several hundred HEAD requests while searching for the 2020 product made the PDS CDN answer
HTTP 429 for about 15 minutes. Later requests were single, spaced downloads.
