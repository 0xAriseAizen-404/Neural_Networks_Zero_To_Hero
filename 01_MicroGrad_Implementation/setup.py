import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd_aa",
    version="0.1.0",
    author="Arise Aizen",
    author_email="maheshmahesh6336.6336@gmail.com",
    description="A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0xAriseAizen-404/Neural_Networks_Zero_To_Hero/tree/master/01_MicroGrad_Implementation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)