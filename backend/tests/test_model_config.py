import asyncio

import prompt_ninja.model_config as model_config


def test_available_models_keeps_text_outputs_without_requiring_structured_output(
    monkeypatch,
):
    async def fake_catalogue():
        return (
            {
                "id": "deepseek/deepseek-v4-flash",
                "name": "DeepSeek V4 Flash",
                "pricing": {},
                "supported_parameters": [],
                "output_modalities": ["text"],
            },
            {
                "id": "openai/gpt-5.6-terra",
                "name": "GPT-5.6 Terra",
                "pricing": {},
                "supported_parameters": ["structured_outputs"],
                "output_modalities": ["text"],
            },
        )

    monkeypatch.delenv("OPENROUTER_ALLOWED_MODELS", raising=False)
    monkeypatch.setattr(model_config, "model_catalogue", fake_catalogue)

    models = asyncio.run(model_config.available_models())

    assert [model["id"] for model in models] == [
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5.6-terra",
    ]


def test_normalise_model_rejects_models_that_cannot_output_text():
    image_only = {
        "id": "example/image-model",
        "name": "Image model",
        "architecture": {"output_modalities": ["image"]},
    }

    assert model_config._normalise_model(image_only) is None
