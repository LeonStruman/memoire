# Datagotchi Health

Datagotchi Health is an algorithm designed to predict mental health outcomes based on lifestyle behaviors and provide recommendations for improving mental health.

## Table of Contents
1. [Prerequisites / Installation](#prerequisites--installation)
2. [Getting Started](#getting-started)
3. [Repository Structure](#repository-structure)

## Prerequisites / Installation

Before you begin, ensure you have met the following requirements:
- Python 3.9
- R
- Poetry (for Python virtual environment)
- Make (to run the Makefile)

To install the required dependencies, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/datagotchi-health.git
    ```
2. Navigate to the project directory:
    ```sh
    cd datagotchi-health
    ```
3. Install the dependencies using Poetry:
    ```sh
    poetry install
    ```

## Getting Started

To get started with the project, follow these steps:

1. Request access to the data by contacting the repository owner.
2. Create a `.env` file in the project root directory with the `DATA_PATH` environment variable set to the location of your data:
    ```env
    DATA_PATH='path/to/data'
    ```
3. Edit the `config.py` file to suit your specific needs.
4. Run the ML pipeline : 
- Step 1 : create features : 
    ```sh
    make create-features
    ```
- Step 2 : select features : 
    ```sh
    make select-features
    ```
- Step 3 : Run the cross-validation:
    ```sh
    make run-crossval
    ```

## Repository Structure

The repository is organized as follows:
code

```markdown
code/
├── cleaning/
│   └── (various cleaning functions)
├── eda/
│   └── (exploratory data analysis scripts)
app/
├── ml/
    └── (machine learning workflow from raw data to evaluated predictions)
``````
- **cleaning**: Contains scripts for data cleaning.
- **eda**: Contains scripts for exploratory data analysis to explore and understand the data.
- **ml**: Contains scripts for the machine learning workflow, including data processing, model training, and evaluation.
