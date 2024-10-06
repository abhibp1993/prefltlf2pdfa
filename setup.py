from setuptools import setup, find_packages

# Read the content of the README file (optional)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prefltlf2pdfa",  # Replace with your project's name
    version="1.0",  # Version of the package
    author="Abhishek N. Kulkarni",  # Your name
    author_email="abhi.bp1993@gmail.com",  # Your email
    description="`prefltlf2pdfa` is a tool for expressing PrefLTLf formulas and converting them to preference automata."
                " PrefLTLf formulas enable expressing preferences over LTLf formulas.",  # Short project description
    long_description=long_description,  # Long description (optional)
    long_description_content_type="text/markdown",  # Content type for the long description
    url="https://github.com/abhibp1993/prefltlf2pdfa",  # URL to your project's repository
    packages=find_packages(),  # Automatically find and include all packages in the project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # License of your project
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Minimum Python version requirement
    install_requires=[
        # List your project's dependencies here, for example:
        # 'requests>=2.25.1',
    ],
    # entry_points={
    #     'console_scripts': [
    #         # Add terminal commands, for example:
    #         # 'your_script_name=your_package.module:function',
    #     ],
    # },
)
