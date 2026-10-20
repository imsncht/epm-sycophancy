from setuptools import setup, find_packages

setup(
    name="sycocomp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.50.0",
        "openai>1.0.0",
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
        "accelerate"
    ],
    entry_points={
        "console_scripts": [
            "run-syco=sycocomp.run_syco:main",
            "get-gpt-responses=sycocomp.get_responses_gpt:main",
            "run-logit-lens=sycocomp.logit-lens:main",
            "run-scorers=sycocomp.scorers:main",
            "run-analysis=sycocomp.analysis:main",
            "run-dpo-pairs=sycocomp.dpo_alignment:main",
            "run-multiturn=sycocomp.multiturn_eval:main",
            "run-aita-bench=sycocomp.aita_benchmark:main",
            "run-convergence-classifier=sycocomp.convergence_classifier:main"
        ],
    },
    author="researcher",
    description="coding modules for LLM sycopahncy and behavioral evaluation research",
    python_requires=">=3.10",
)