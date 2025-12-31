import os
import numpy as np
from unet_model import SimpleUNet, extract_features_unet
from sklearn.semi_supervised import LabelPropagation

# Run vanilla UNet (from unet_model) and extract features weighted by sampling location
model = SimpleUNet(in_channels=1, base_channels=16, out_channels=64)
model.eval()

# deterministic color map from label ID → RGB
def colorize_labels(label_img, bg_label=0):
    max_lab = int(label_img.max())
    lut = np.zeros((max_lab + 1, 3), dtype=np.uint8)

    for lab in np.unique(label_img):
        if lab == bg_label:
            lut[lab] = (0, 0, 0)  # background black
        else:
            rng = np.random.default_rng(seed=lab)  # seed by label id
            lut[lab] = rng.integers(0, 256, size=3, dtype=np.uint8)

    return lut[label_img]
    
def generate_image_label(test_nissl, nissl, anno):
    features_fixed = extract_features_unet(model, nissl)
    features_test_nissl = extract_features_unet(model, test_nissl)

    # Mask zero values outside the histo stack, then filter features and labels based on the masks
    mask_nissl = (nissl.flatten() != 0)
    mask_test_nissl = (test_nissl.flatten() != 0)

    features_nissl_masked = features_fixed[mask_nissl]
    features_test_nissl_masked = features_test_nissl[mask_test_nissl]
    labels_nissl = anno.flatten()  # Provided labels for nissl
    labels_nissl_masked = labels_nissl[mask_nissl]

    # For nissl, we have known labels; for test_nissl, set labels to -1 (unlabeled).
    unlabeled_test_nissl = -np.ones(features_test_nissl_masked.shape[0], dtype=int)

    # Combine features and labels from both images.
    features_combined = np.concatenate([features_nissl_masked, features_test_nissl_masked], axis=0)
    labels_combined = np.concatenate([labels_nissl_masked, unlabeled_test_nissl], axis=0)

    # Label propgation
    lp_model = LabelPropagation(kernel='knn', n_neighbors=50)
    lp_model.fit(features_combined, labels_combined)

    propagated_labels_test_nissl = lp_model.transduction_[features_nissl_masked.shape[0]:]

    # Assign labels to new annotation image
    anno2_flat = np.zeros(test_nissl.size, dtype=int)
    # Fill in only the valid (nonzero) positions with propagated labels
    anno2_flat[mask_test_nissl] = propagated_labels_test_nissl
    anno2 = anno2_flat.reshape(test_nissl.shape)

    #np.save("../data/anno2.npy", anno2)
    return anno2

