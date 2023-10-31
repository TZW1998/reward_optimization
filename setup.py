from setuptools import setup, find_packages

setup(
    name="reward_opt",
    version="0.0.1",
    packages=["reward_opt"],
    python_requires="==3.11.5",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]",
        "accelerate",
        "wandb",
        "torchvision",
        "inflect",
        "pydantic",
        "transformers",
    ],
)
