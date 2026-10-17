# Dataset acquisition and provenance

## Static alphabet recognition

Public benchmark: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). Download it yourself under its stated license. Keep train, validation and test images distinct, preferably by collection session/signer; random image splits may inflate scores. J and Z involve motion and cannot be reliably evaluated as full signs from single still frames.

## Isolated-sign sequence recognition

[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) is recommended for signer-independent isolated-sign research. Obtain data and official partitions through the dataset provider. Construct a CSV with columns `path,label,split,signer` from the dataset's metadata; do not guess signers or splits. Run `python -m asl.video_features --manifest path/to/manifest.csv --output data/temporal`. Train with `python -m asl.temporal --data data/temporal --out outputs/temporal`.

Do not combine the static alphabet and isolated-sign tests into one number. Preserve licenses and do not commit raw videos. WLASL is another separately licensed research benchmark; check terms before use.
