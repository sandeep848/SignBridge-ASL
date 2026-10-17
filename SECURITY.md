# Security and privacy

This repository provides research software, **not a hardened public inference service**. The FastAPI app is intended for use on a trusted local machine bound to 127.0.0.1.

Do not expose it publicly without authentication, rate limiting, request-size enforcement at the proxy, request timeouts, worker isolation for video decoders, model loading controls, and a privacy/data-retention assessment.

Report suspected vulnerabilities privately to the repository maintainers through GitHub security advisories or another non-public contact mechanism. Do not open a public issue with an exploitable vulnerability or private signer recordings.

Webcam input is processed locally by default. Do not collect, store, or share user video without explicit consent and a documented retention policy.
