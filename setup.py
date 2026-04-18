from setuptools import setup, find_packages

setup(
    name="tmg-q",
    version="1.0.0",
    description="Tanh-Nonlinear Mixed-Precision Genetic Quantization Framework",
    author="Abdullah Salem Saleh Al-Faqeer",
    packages=find_packages(where="TMG-Q"),
    package_dir={"": "TMG-Q"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35",
        "numpy>=1.24"
    ],
    scripts=[
        "TMG-Q/scripts/Chat_GPT2_V2.py",
        "TMG-Q/scripts/Compress_GPT2_V2.py"
    ]
)
