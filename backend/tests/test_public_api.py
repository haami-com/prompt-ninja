import re

import prompt_ninja


def test_public_package_exports_the_primary_api():
    assert re.fullmatch(r"\d+\.\d+\.\d+", prompt_ninja.__version__)
    assert prompt_ninja.PromptNinja.__name__ == "PromptNinja"
    assert prompt_ninja.PromptCollection.__name__ == "PromptCollection"
    assert prompt_ninja.OpenRouterPromptClient.__name__ == "OpenRouterPromptClient"
    assert prompt_ninja.TokenUsageCostHook.__name__ == "TokenUsageCostHook"
