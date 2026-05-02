# Barnsley Fern Models

These models learn to predict which affine transform generated each fern point.

| Model | Train accuracy | Test accuracy |
| --- | ---: | ---: |
| Nearest centroid | 0.800 | 0.803 |
| Gaussian naive Bayes | 0.947 | 0.946 |
| K-nearest neighbors (k=7) | 1.000 | 0.998 |

The labels come directly from the four Barnsley fern transforms, so this is a clean supervised learning setup.