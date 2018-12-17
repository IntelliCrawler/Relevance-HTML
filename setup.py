import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="relevance_html",
    version="0.0.1",
    author="Shuochi Huang, Cheng Ding, Tianyu Ao",
    author_email="author@example.com",
    description="Relevance-HTML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IntelliCrawler/Relevance-HTML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
