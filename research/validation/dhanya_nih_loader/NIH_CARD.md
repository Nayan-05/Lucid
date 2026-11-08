# NIH ChestX-ray14 card

- Source: Data_Entry_2017_v2020.csv (NIH ChestX-ray14 metadata), from https://nihcc.app.box.com/v/ChestXray-NIHCC
- Downloaded: 18-10-2026 (downloaded again on 08-11-2026; same size, same counts)
- Size: 9,003,496 bytes
- Rows: 112,120 (one row per image)
- Patients: 30,805
- Licence: research use only. Must attribute the NIH Clinical Center and cite Wang et al. (CVPR 2017). Terms: https://nihcc.app.box.com/v/ChestXray-NIHCC

## First look

- Finding Labels, most common: No Finding 60,361; Infiltration 9,547; Atelectasis 4,215; Effusion 3,955; Nodule 2,705; Pneumothorax 2,194; Mass 2,139.
- Pneumonia is rare (about 1% of images): 1,431 images from 1,008 patients.
- Patient Age: min 0, max 95, mean 46.6. Impossible ages (> 100) in this copy: none (impossible_ages_dropped = 0).
- Children (age 18 or under) with pneumonia: 76 patients.

## Label rule used

Positive: the pipe-separated Finding Labels contain exactly "Pneumonia" ("Pneumothorax" does not count). Negative: the cell is exactly "No Finding". Everything else is skipped.

## Warning

The labels are text-mined from radiology reports and are noisy (weak labels). Images are frontal views (PA and AP) from one US hospital; results here may not transfer to other hospitals or to children's X-rays.
