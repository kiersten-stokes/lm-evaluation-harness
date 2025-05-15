import json

import datasets


def process_docs(
    dataset: datasets.Dataset, contexts: tuple[list[str]]
) -> datasets.Dataset:
    def _process_doc(doc, ctx):
        out_doc = {**doc, "source": ctx[0], "source_ids": ctx[1]}
        return out_doc

    return list(map(_process_doc, dataset, contexts))


def postprocess_docs(predictions, references):
    predictions_context = {
        "answer": predictions,
        "contexts": references["source"],
        "context_ids": references["source_ids"],
    }
    return json.dumps(predictions_context)
