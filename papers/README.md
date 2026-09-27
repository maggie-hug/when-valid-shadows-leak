# Audited source papers

These are the three papers whose generation rules are implemented in
`scripts/check_cross_scheme_witnesses.py`. Source details were checked on
2026-09-28. The checks enumerate coefficient-stage privacy counterexamples;
the JPEG checks cover coefficient, regulation, and stability arithmetic.

## TCSVT21

Xuehu Yan, Yuliang Lu, Ching-Nung Yang, Xinpeng Zhang, and Shudong Wang.
**A Common Method of Share Authentication in Image Secret Sharing.**
*IEEE Transactions on Circuits and Systems for Video Technology*, 31(7),
2896-2908, 2021. DOI: [10.1109/TCSVT.2020.3025527](https://doi.org/10.1109/TCSVT.2020.3025527).

- [PDF in this repository](TCSVT21.pdf): unmodified publisher version, 13 pages.
- [Original publisher PDF](https://ieeexplore.ieee.org/ielx7/76/9471039/09201524.pdf).
- [Public full-text page](https://www.researchgate.net/publication/345333246_A_Common_Method_of_Share_Authentication_in_Image_Secret_Sharing).
- Source rules: Eq. (1), printed p. 2899 (PDF p. 4); Algorithm 1, Steps 2-4,
  printed p. 2900 (PDF p. 5). Step 2 samples the hidden pattern, while a
  rejected coefficient returns to Step 3.
- Privacy statement: the no-leakage property, printed p. 2906 (PDF p. 11).
- Repository checks: both coefficient domains, participants 1 and 2;
  `validation/tcsvt21_generator/` also implements the image-generation retry loop.

License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/),
as stated on the PDF's first page. This copy is shared unchanged for
noncommercial research use.

## MBE22

Yue Jiang, Xuehu Yan, Jia Chen, Jingwen Cheng, and Jianguo Zhang.
**Meaningful secret image sharing for JPEG images with arbitrary quality factors.**
*Mathematical Biosciences and Engineering*, 19(11), 11544-11562, 2022.
DOI: [10.3934/mbe.2022538](https://doi.org/10.3934/mbe.2022538).

- [PDF in this repository](MBE22.pdf): unmodified publisher version, 19 pages.
- [Publisher article page](https://www.aimspress.com/article/doi/10.3934/mbe.2022538).
- [Original publisher PDF](https://www.aimspress.com/aimspress-data/mbe/2022/11/PDF/mbe-19-11-538.pdf).
- Source rules: Eq. (2.7), printed p. 11549 (PDF p. 6); Algorithm 1,
  Steps 4-11, printed p. 11551 (PDF p. 8), covering prefix constraints,
  accepted coefficients, splicing, and JPEG encoding.
- Privacy argument: Sec. 4.1, printed p. 11553 (PDF p. 10).
- Repository checks: participant-1 witnesses under the 8- and 9-bit
  coefficient representations.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as stated on the PDF's final page: copyright 2022 the authors, licensee AIMS Press.

## SBC24

Yue Jiang, Kejiang Chen, Wei Yan, Xuehu Yan, Guozheng Yang, and Kai Zeng.
**Robust Secret Image Sharing Resistant to JPEG Recompression Based on Stable Block Condition.**
*IEEE Transactions on Multimedia*, 26, 10446-10461, 2024.
DOI: [10.1109/TMM.2024.3407694](https://doi.org/10.1109/TMM.2024.3407694).

- [Publisher article page](https://ieeexplore.ieee.org/document/10542386/).
- [Author's public full-text page](https://www.researchgate.net/publication/381022466_Robust_Secret_Image_Sharing_Resistant_to_JPEG_Recompression_Based_on_Stable_Block_Condition).
- [Author-version PDF download](https://www.researchgate.net/publication/381022466_Robust_Secret_Image_Sharing_Resistant_to_JPEG_Recompression_Based_on_Stable_Block_Condition/fulltext/6659f840479366623a33da9c/Robust-Secret-Image-Sharing-Resistant-to-JPEG-Recompression-Based-on-Stable-Block-Condition.pdf).
- Source rules: Algorithm 1 and Eqs. (10), (13)-(20), including parameter
  derivation, prefix matching, regulation, stability screening, and recovery.
  In the published version, Algorithm 1 is on printed p. 10450 (PDF p. 5),
  and Eqs. (17)-(19) are on printed p. 10451 (PDF p. 6).
- Privacy argument: Sec. IV-C1. Equation numbers and section labels are
  preferable locators for the author version, whose page numbering differs.
- Repository checks: participant-2 witnesses under the 8- and 9-bit
  representations, including regulated outputs and authorized recovery.

The publicly posted author version identifies itself as an accepted manuscript
and contains a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) notice.
Its file could not be downloaded during this update, so the repository provides
the original download and full-text links. The local publisher copy requires
IEEE permission for redistribution and is not included here.

## Third-party licenses

The repository's MIT license applies to its code. The papers retain their
own licenses and attribution; their PDFs have not been edited or re-exported.
