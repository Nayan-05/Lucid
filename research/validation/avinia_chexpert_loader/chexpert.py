from pathlib import Path, PurePosixPath

import pandas as pd


def read_labels(json_path):
	"""Read the labels as JSON Lines, falling back to a regular JSON list."""
	try:
		return pd.read_json(json_path, lines=True)
	except ValueError:
		return pd.read_json(json_path)


def map_chexpert(meta, labels, image_root, n_pos=500, n_neg=500, seed=42, ext=None):
	"""Map CheXpert metadata and labels into balanced records and audit counts."""
	counts = {}

	# Start the audit trail with the total metadata row count.
	counts["rows"] = len(meta)

	# Keep frontal images only.
	frontal = meta.loc[meta["frontal_lateral"] == "Frontal"].copy()
	counts["frontal"] = len(frontal)

	# Join the labels to the frontal metadata by image path.
	label_columns = labels[["path_to_image", "Pneumonia", "No Finding"]]
	joined = frontal.merge(label_columns, on="path_to_image", how="inner")
	counts["frontal_with_labels"] = len(joined)

	# Store each patient ID as text for the shared record format.
	joined["patient"] = joined["deid_patient_id"].astype(str)

	# A positive has Pneumonia = 1.0.
	positive_mask = joined["Pneumonia"] == 1.0

	# A negative has No Finding = 1.0, with no positive or uncertain pneumonia.
	negative_mask = (
		(joined["No Finding"] == 1.0)
		& ~joined["Pneumonia"].isin([1.0, -1.0])
	)

	# Count uncertain pneumonia and rows that are neither positive nor negative.
	counts["uncertain_pneumonia_dropped"] = int(
		(joined["Pneumonia"] == -1.0).sum()
	)
	counts["not_pos_not_neg_dropped"] = (
		len(joined) - int(positive_mask.sum()) - int(negative_mask.sum())
	)

	# Keep the first positive image per patient.
	positives = (
		joined.loc[positive_mask]
		.drop_duplicates(subset="patient", keep="first")
		.copy()
	)

	# Exclude from the negative class anyone who has a positive image.
	positive_patients = set(positives["patient"])
	negative_candidates = joined.loc[negative_mask].copy()
	negative_candidates = negative_candidates.loc[
		~negative_candidates["patient"].isin(positive_patients)
	]

	# Keep the first remaining negative image per patient.
	negatives = negative_candidates.drop_duplicates(
		subset="patient", keep="first"
	).copy()

	counts["positives_available"] = len(positives)
	counts["negatives_available"] = len(negatives)

	# Use the same number from each class, limited by availability and requests.
	n = min(n_pos, n_neg, len(positives), len(negatives))
	positive_sample = positives.sample(n=n, random_state=seed)
	negative_sample = negatives.sample(n=n, random_state=seed)

	counts["final_per_class"] = n
	counts["final_total"] = 2 * n

	# Build forward-slash paths and change the extension if requested.
	root = str(image_root).replace("\\", "/").rstrip("/")
	records = []

	for label, sample in [(1, positive_sample), (0, negative_sample)]:
		for _, row in sample.iterrows():
			relative_path = PurePosixPath(
				str(row["path_to_image"]).replace("\\", "/")
			)

			if ext is not None:
				new_extension = ext if ext.startswith(".") else f".{ext}"
				relative_path = relative_path.with_suffix(new_extension)

			relative_text = relative_path.as_posix()
			image_path = f"{root}/{relative_text}" if root else relative_text

			age_value = row["age"]
			age = None if pd.isna(age_value) else int(age_value)

			view_value = row["ap_pa"]
			view = None if pd.isna(view_value) or view_value == "" else str(view_value)

			records.append(
				{
					"path": image_path,
					"label": label,
					"patient": str(row["patient"]),
					"source": "chexpert",
					"view": view,
					"age": age,
				}
			)

	return records, counts


def load_chexpert(
	csv_path,
	image_root,
	n_pos=500,
	n_neg=500,
	seed=42,
	labels_path=None,
	ext=None,
):
	"""Load the two files, map them, and return only the records."""
	csv_path = Path(csv_path)

	# By default, look for impression_fixed.json beside the CSV.
	if labels_path is None:
		labels_path = csv_path.with_name("impression_fixed.json")

	meta = pd.read_csv(csv_path)
	labels = read_labels(labels_path)

	records, _ = map_chexpert(
		meta,
		labels,
		image_root,
		n_pos=n_pos,
		n_neg=n_neg,
		seed=seed,
		ext=ext,
	)
	return records
