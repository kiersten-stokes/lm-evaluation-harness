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
        doc["target"]

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
        source = self.doc_to_text(doc)
        if isinstance(source, list):
            if apply_chat_template:
                formated_source = chat_template(self.doc_to_text(doc))
                return formated_source
            else:
                raise Exception(
                    "Got chat template format from Unitxt, but apply_chat_template is false. Add '--apply_chat_template' to command line."
                )
        else:
            return source

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


class UnitxtRAG(Unitxt):
    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}

        assert (rag_args := config.get("rag")), "Not a RAG task"

        assert (session_args := simple_parse_args_string(rag_args.get("session"))), (
            "RAG task missing MCP session connection arguments"
        )
        session_args.pop("arg1")  # remove example values

        assert (request_args := simple_parse_args_string(rag_args.get("request"))), (
            "RAG task missing retrieval request arguments"
        )
        assert (tool := request_args.pop("tool")), (
            "RAG task retrieval request arguments must include `tool` name"
        )
        assert (query_field := request_args.pop("query_field")), (
            "RAG task retrieval request arguments must include `query_field` name"
        )
        # TODO do better error checking for correct formatting

        self.contexts = None

        super().__init__(config)  # will load dataset

        queries = [json.loads(data["task_data"])["question"] for data in self.eval_docs]
        # TODO add error checking to ensure the query structure is always the same

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self.contexts = []

        async def retrieve_contexts():
            # Connect to a streamable HTTP server
            async with streamablehttp_client(**session_args) as (
                read_stream,
                write_stream,
                _,
            ):
                # Create a session using the client streams
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize connection
                    await session.initialize()
                    for query in queries:
                        req_args = {query_field: query, **request_args}
                        tool_result = await session.call_tool(tool, req_args)
                        contexts = [
                            item
                            for sublist in [
                                json.loads(res.text)["contexts"]
                                for res in tool_result.content
                            ]
                            for item in sublist
                        ]
                        context_ids = [
                            json.loads(res.text)["id"] for res in tool_result.content
                        ]
                        self.contexts.append((contexts, context_ids))

        asyncio.run(retrieve_contexts())

        # contexts = [[json.loads(data["target"])["answer"]] for data in self.eval_docs]  # workaround for testing
        # context_ids = [
        #    json.loads(data["references"][0])["context_ids"] for data in self.eval_docs
        # ]
        # self.contexts = list(zip(contexts, context_ids))

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
        source = self.doc_to_text(doc)
        if isinstance(source, list):
            if apply_chat_template:  # TODO better way to handle chat_template??
                chat_history = [
                    {
                        "role": "user",
                        "content": "Use the following context to inform your answer: "
                        + " ".join(source),
                    }
                ]
                formated_source = chat_template(chat_history)
                return formated_source
            else:
                raise Exception(
                    "Got chat template format from Unitxt, but apply_chat_template is false. Add '--apply_chat_template' to command line."
                )
        else:
            return source

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
            # Use the default `outputs` structure:
            #  https://www.unitxt.ai/en/stable/docs/rag_support.html#rag-task-definition
            prediction = {
                "answer": predictions,
                "contexts": references["source"],
                "context_ids": references["source_ids"],
            }

        return {
            metric.replace("metrics.", ""): (prediction, doc) for metric in self.metrics
        }
