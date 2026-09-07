# Final Experimental Results

Higher utility Rank-1/mAP is better. Lower source attack, adaptive attack, donor leakage and FID are preferable for their respective interpretations.

| method         |   utility_rank1 |   utility_mAP |   source_attack_rank1 |   source_attack_mAP |   adaptive_attack_rank1 |   donor_rank1 |   feature_cosine |   ssim |   psnr_db |    fid |
|:---------------|----------------:|--------------:|----------------------:|--------------------:|------------------------:|--------------:|-----------------:|-------:|----------:|-------:|
| Original       |           14.73 |          7.84 |                 14.73 |                7.84 |                  nan    |        nan    |           1      | 1      |    nan    |   0    |
| Resize control |           14.76 |          7.86 |                 14.58 |                7.85 |                  nan    |        nan    |           0.9998 | 0.9986 |     51.01 |   0.13 |
| Blur r=2       |           11.37 |          6.38 |                 10.63 |                6.26 |                  nan    |        nan    |           0.935  | 0.6887 |     24.02 | nan    |
| Blur r=4       |            8.05 |          3.81 |                  4.72 |                3.36 |                  nan    |        nan    |           0.7718 | 0.4832 |     20.62 | nan    |
| Blur r=8       |            5.31 |          2.37 |                  1.93 |                1.78 |                  nan    |        nan    |           0.6203 | 0.3734 |     18.24 | 393.34 |
| Pixel 32x64    |           14.58 |          7.74 |                 14.07 |                7.78 |                  nan    |        nan    |           0.9977 | 0.8744 |     27.67 | nan    |
| Pixel 16x32    |           13.84 |          7.42 |                 13.33 |                7.54 |                  nan    |        nan    |           0.9858 | 0.6395 |     22.88 | nan    |
| Pixel 8x16     |           11.64 |          6.41 |                 11.61 |                6.67 |                  nan    |        nan    |           0.9628 | 0.4188 |     19.85 | 406.41 |
| PIS α=0.00     |            4.04 |          2.17 |                  2.67 |                2.09 |                    8.36 |          0.06 |           0.6831 | 0.3496 |     16.82 | 291.4  |
| PIS α=0.25     |            4.25 |          2.09 |                  1.78 |                1.8  |                    5.87 |          0.15 |           0.6517 | 0.3466 |     16.75 | 297.95 |
| PIS α=0.50     |            6.98 |          2.79 |                  1.1  |                1.2  |                    2.19 |          0.21 |           0.5703 | 0.3346 |     16.33 | 299.25 |
| PIS α=0.75     |           22.68 |          8.7  |                  0.39 |                0.57 |                    1.3  |          1.66 |           0.4739 | 0.3134 |     15.45 | 291.8  |
| PIS α=1.00     |           45.25 |         21.39 |                  0.24 |                0.4  |                    0.71 |          3.38 |           0.4258 | 0.295  |     14.66 | 284.11 |

## Protocol note

Adaptive privacy results use a separate held-out 375-identity target split and should not be treated as directly numerically equivalent to the full 750-identity fixed-attack protocol.
