# Contributing to SignBridge

Contributions are welcome when they are **testable, reproducible and properly sourced**.

1. Open an issue describing the problem and the proposed change.
2. Keep static alphabet and isolated-sign benchmarks separate; identify the exact task and dataset.
3. Work on a feature branch and open a pull request. Avoid committing large binary datasets, videos, checkpoints or personally identifying recordings.
4. Add focused tests for bug fixes. Run `python -m pytest -q` and `python -m compileall -q asl apps`.
5. For numerical improvements, include the data protocol, seed, environment, validation procedure, uncertainty estimates and a comparable baseline. Do not present validation-selected scores as untouched test results.
6. Document any license restrictions for third-party data or weights.

Do not submit generated benchmark numbers, fake model weights, speculative claims of translation performance, or personally identifiable material without documented authorization.

Please be respectful of Deaf and hard-of-hearing communities and distinguish isolated sign recognition from general sign-language translation.
