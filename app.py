import os

import modal

ENVIRONMENT_NAME = "alessio-dev"

# Volume holding the production traces replayed by the experiment runner. The
# public default is ``GORGO-completions``; the traces behind the paper live on a
# private volume with a different name, so allow an override rather than
# hardcoding it. Without this, ``create_if_missing=True`` silently creates an
# EMPTY ``GORGO-completions`` and every replay fails with FileNotFoundError on
# its trace path, which looks like a bad manifest rather than a missing mount.
COMPLETIONS_VOLUME_NAME = os.getenv("GORGO_COMPLETIONS_VOLUME", "GORGO-completions")

app = modal.App(name="GORGO")
replicas = modal.Dict.from_name(
    "GORGO-replicas", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
proxies = modal.Dict.from_name(
    "GORGO-proxies", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
completions_volume = modal.Volume.from_name(
    COMPLETIONS_VOLUME_NAME, create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
# Output destination for ``proxy/workload.py`` runs (one JSON doc per run
# under ``/results``).
bench_results_volume = modal.Volume.from_name(
    "GORGO-bench-results", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
hf_datasets_volume = modal.Volume.from_name(
    "GORGO-hf-datasets", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
# HF ``save_to_disk`` for lmsys/lmsys-chat-1m (e.g. ``…/lmsys-chat-1m/train/*.arrow``).
lmsys_chat_1m_volume = modal.Volume.from_name(
    "GORGO-lmsys-chat-1m", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
