"""
In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.

Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt-Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively.
"""

import asyncio
import importlib.util
import json
import re
from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Optional

import datasets

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.utils import simple_parse_args_string


_CITATION = """
@misc{bandel2024unitxt,
      title={Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI},
      author={Elron Bandel and Yotam Perlitz and Elad Venezian and Roni Friedman-Melamed and Ofir Arviv and Matan Orbach and Shachar Don-Yehyia and Dafna Sheinwald and Ariel Gera and Leshem Choshen and Michal Shmueli-Scheuer and Yoav Katz},
      year={2024},
      eprint={2401.14019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


def assert_unitxt_installed():
    if importlib.util.find_spec("unitxt") is None:
        raise Exception(
            "Please install unitxt via 'pip install unitxt'. For more information see: https://www.unitxt.ai/"
        )

    from unitxt import __version__ as unitxt_version

    # Function argument change due to https://github.com/IBM/unitxt/pull/1564
    unitxt_version = tuple(map(int, (unitxt_version.split("."))))
    if unitxt_version < (1, 17, 2):
        raise Exception(
            "Please install a more recent version of unitxt via 'pip install --upgrade unitxt' to avoid errors due to breaking changes"
        )


def assert_mcp_installed():
    if importlib.util.find_spec("mcp") is None:
        raise Exception("Please install mcp via 'pip install mcp'")


class UnitxtFactory:
    def __init__(self, config: Optional[dict] = None) -> None:
        pass

    def __new__(cls, config):
        if config.get("rag"):
            config["class"] = UnitxtEnd2EndRAG
            return UnitxtEnd2EndRAG(config)
        config["class"] = Unitxt
        return Unitxt(config)


class Unitxt(ConfigurableTask):
    VERSION = 0

    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}
        assert "recipe" in config, "Unitxt task must have a 'recipe' string."

        unitxt_config = {
            "metadata": {"version": self.VERSION},
            "dataset_name": config["recipe"],
        }

        if config.get("process_docs"):
            unitxt_config["process_docs"] = config.get("process_docs")
        if config.get("process_results"):
            unitxt_config["process_results"] = config.get("process_results")

        super().__init__(config=unitxt_config)
        self.image_decoder = datasets.Image()
        self.metrics = self.dataset["test"][0]["metrics"]

    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        assert_unitxt_installed()
        from unitxt import load_dataset

        self.dataset = load_dataset(self.DATASET_NAME, use_cache=True)

    def has_training_docs(self):
        return "train" in self.dataset

    def has_validation_docs(self):
        return "validation" in self.dataset

    def has_test_docs(self):
        return "test" in self.dataset

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["source"]

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        return doc["target"]

    def get_arguments(self, doc, ctx):
        return (ctx, {"until": ["\n"]})

    def fewshot_context(
        self,
        doc: str,
        num_fewshot: int,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        gen_prefix: Optional[str] = None,
    ) -> str:
        if isinstance(self.doc_to_text(doc), list):
            if apply_chat_template:
                formated_source = chat_template(self.doc_to_text(doc))
                return formated_source
            else:
                raise Exception(
                    "Got chat template format from Unitxt, but apply_chat_template is false. Add '--apply_chat_template' to command line."
                )
        else:
            return super().fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                system_instruction=system_instruction,
                apply_chat_template=apply_chat_template,
                fewshot_as_multiturn=fewshot_as_multiturn,
                chat_template=chat_template,
                gen_prefix=gen_prefix,
            )

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        kwargs.pop("apply_chat_template", False)  # Not used by unitxt
        kwargs.pop("chat_template", False)  # Not used by unitxt
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=self.get_arguments(doc, ctx),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """

        continuation = results[0]

        predictions = continuation

        references = doc
        return {
            metric.replace("metrics.", ""): (predictions, references)
            for metric in self.metrics
        }

    def score(self, items, metric):
        predictions, references = zip(*items)
        assert_unitxt_installed()
        from unitxt import evaluate

        for reference in references:
            reference["metrics"] = [metric]

        results = evaluate(predictions, references)

        return results[0]["score"]["global"]["score"]

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            metric.replace("metrics.", ""): partial(self.score, metric=metric)
            for metric in self.metrics
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {metric.replace("metrics.", ""): True for metric in self.metrics}


images_regex = r'<img\s+src=["\'](.*?)["\']\s*/?>'
image_source_regex = r'<img\s+src=["\'](.*?)["\']'


def extract_images(text, instance):
    image_sources = re.findall(image_source_regex, text)
    images = []
    for image_source in image_sources:
        current = instance
        for key in image_source.split("/"):
            if key.isdigit():
                key = int(key)
            current = current[key]
        images.append(current)
    return images


class UnitxtMultiModal(Unitxt):
    MULTIMODAL = True

    def doc_to_text(self, doc):
        return re.sub(images_regex, "<image>", doc["source"])

    def doc_to_image(self, doc):
        images = extract_images(doc["source"], doc)
        return [self.image_decoder.decode_example(image) for image in images]

    def get_arguments(self, doc, ctx):
        return (ctx, {"until": ["\n"]}, {"visual": self.doc_to_image(doc)})


class UnitxtEnd2EndRAG(Unitxt):
    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}

        session_args, request_args = self.validate_rag_and_return(config)
        self.contexts = None

        super().__init__(config)

        assert_mcp_installed()
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        retrieved_contexts = []

        async def retrieve_contexts():
            async with streamablehttp_client(**session_args) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_name = request_args.pop("tool")
                    query_field = request_args.pop("query_field")
                    context_field = request_args.pop("context_field")
                    id_field = request_args.pop("id_field")

                    for query in self.queries:
                        req_args = {query_field: query, **request_args}
                        tool_result = await session.call_tool(tool_name, req_args)
                        contexts = [
                            item
                            for sublist in [
                                json.loads(res.text)[context_field]
                                for res in tool_result.content
                            ]
                            for item in sublist
                        ]
                        context_ids = [
                            json.loads(res.text)[id_field]
                            for res in tool_result.content
                        ]
                        retrieved_contexts.append((contexts, context_ids))

        asyncio.run(retrieve_contexts())

        self.contexts = retrieved_contexts

    @staticmethod
    def validate_rag_and_return(config):
        assert (rag_args := config.get("rag")), "Task yaml missing the 'rag' section"

        assert (session_args := simple_parse_args_string(rag_args.get("session"))), (
            "RAG task yaml missing 'session' section for MCP connection arguments"
        )

        assert (request_args := simple_parse_args_string(rag_args.get("request"))), (
            "RAG task yaml missing 'request' section with retrieval arguments"
        )
        assert request_args.get("tool"), (
            "RAG task retrieval request arguments must include 'tool' name"
        )
        assert request_args.get("query_field"), (
            "RAG task retrieval request arguments must include 'query_field' name"
        )
        assert request_args.get("context_field"), (
            "RAG task retrieval request arguments must include 'context_field' name"
        )
        assert request_args.get("id_field"), (
            "RAG task retrieval request arguments must include 'id_field' name"
        )
        return session_args, request_args

    @property
    def queries(self):
        return [json.loads(doc["task_data"])["question"] for doc in self.eval_docs]

    def test_docs(self):
        if self.config.process_docs is not None and self.contexts is not None:
            return self.config.process_docs(self.dataset["test"], self.contexts)
        return self.dataset["test"]

    def validation_docs(self):
        if self.config.process_docs is not None and self.contexts is not None:
            return self.config.process_docs(self.dataset["validation"], self.contexts)
        return self.dataset["validation"]

    def training_docs(self):
        if self.config.process_docs is not None and self.contexts is not None:
            return self.config.process_docs(self.dataset["train"], self.contexts)
        return self.dataset["train"]

    def doc_to_text(self, doc):
        contexts = doc.get("contexts", None)
        if contexts is not None and isinstance(doc["contexts"], list):
            return "Use the following context to inform your answer: " + " ".join(
                contexts
            )
        return doc["source"]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        predictions, references = results[0], doc
        if callable(self.config.process_results):
            prediction = self.config.process_results(predictions, references)
        else:
            # Use the default Unitxt `outputs` structure:
            #  https://www.unitxt.ai/en/stable/docs/rag_support.html#rag-task-definition
            prediction = {
                "answer": predictions,
                "contexts": references["contexts"],
                "context_ids": references["context_ids"],
            }

        return {
            metric.replace("metrics.", ""): (prediction, doc) for metric in self.metrics
        }
