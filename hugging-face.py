"""This file demonstrates how to use Modal to host a HuggingFace-based HTTP
inference API.

We demonstrate the following:

- Programmatically defining and constructing the image via Modal's SDK,
  including logic to actually pull and download the model;

- Hooking HF-based inference up to a REST endpoint


> modal deploy hugging-face.py
> curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"msg": "I am Jonathan and I ", "verbose": true}' \
    https://jinnovation--jjin-hf-modal-poc-model-complete.modal.run
{"prediction":"I am Jonathan and I  have two daughters. I have made this blog because I am a momma to four children, a wife wife to three daughters. And, we just have five kids, and I am not ready to"}
"""

import modal
from pydantic import BaseModel
from fastapi.responses import JSONResponse

img = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers==4.39.3",
        "huggingface_hub==0.22.2",
        "torch==2.1.2",
    )
)

app = modal.App("jjin-hf-modal-poc")

class Request(BaseModel):
    verbose: bool = True
    msg: str

@app.cls(
    # NB(@jinnovation): This is also where we can specify the GPU type if needed
    # gpu = "A100",
    timeout = 60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=img,
)
class Model:
    @modal.enter()
    def preload_model(self):
        """This method runs once on container initialization, amortizing the
        model-loading cost to zero over time.

        Note that this pattern -- encapsulating expensive initialization logic
        behind a @modal.enter-decorated method -- generalizes well to, for
        example, downloading datasets, pre-loading them into memory, etc.

        """
        from transformers import pipeline

        self.pipeline = pipeline("text-generation", model="openai-community/gpt2")

    @modal.web_endpoint(method="POST")
    async def complete(self, req: Request):
        if req.verbose:
            print("PREDICTING!!")

        return JSONResponse({"prediction": self.pipeline(req.msg)[0]["generated_text"]})
