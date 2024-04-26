# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import Iterator
from cog import BasePredictor, ConcatenateIterator, Input
from llama_cpp import Llama


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.llm = Llama(
            "./lama3-8b-hikikomori-v0.3-GGUF-unsloth.Q5_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=4096,
            seed=-1,
        )

    def predict(
        self,
        prompt: str = Input(
            description="The prompt to generate text from.",
            default="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What's the largest planet in the solar system?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.",
            default=64,
        ),
        temperature: float = Input(
            description="The temperature to use for sampling.", default=0.8
        ),
        top_p: float = Input(
            description="The nucleus sampling probability.", default=0.95
        ),
        frequency_penalty: float = Input(
            description="The frequency penalty to use.", default=0.0
        ),
        presence_penalty: float = Input(
            description="The presence penalty to use.", default=0.0
        ),
        stop: str = Input(
            description="The stop sequence to use.",
            default="\n",
        ),
    ) -> ConcatenateIterator[str]: # type: ignore
        """Run a single prediction on the model"""
        stream = self.llm(
            stream=True,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        if isinstance(stream, Iterator):
            for output in stream:
              result = output["choices"][0]["text"]
              yield result # type: ignore
