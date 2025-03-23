> Paper code for the paper "LLMs as Proxy Survey Participants With RAG" by Elias Torjani, Airidas Brikas, and Daniel Hardt (a continuation of our bachelor thesis)

### FAQ
- What is this? 
- Why did we make it? 
- What can it be used for?

^Check out our paper on it: [Market research via persona-induced Large Language Models](https://url.com)


## How to reproduce our experiments with your own data
1. Export your chat messages from Facebook, Instagram, and/or WhatsApp (links to instructions below)
2. Take the surveys to constitute target responses, for the LLMs proxying you in the same surveys.
3. Clone this repository
4. Download [Ollama](https://ollama.com/) and your models of choice, to run inference locally. 
   - *API-cloud is discouraged due to possible sensitive information leakage.*


> [!IMPORTANT]
> **Exporting chat messages from Facebook, Instagram, and/or WhatsApp**
> This is a manual process, and Meta will take a few days to a week.
> - [Facebook](https://accountscenter.facebook.com/info_and_permissions/dyi) --> Account settings --> Download your information
>   - Get both Facebook and Instagram
>   - Date range = All time
>   - Format = JSON
> - WhatsApp --> Settings --> Account --> Request account info --> Account information

>[!NOTE] <details> <summary>High-level</summary>
>```
>└── Data Analysis Pipeline
>    ├── 1. Initial Setup
>    │   ├── Load simulation files
>    │   ├── Configure directories
>    │   └── Import dependencies
>    │
>    ├── 2. Data Processing
>    │   ├── Extract run numbers
>    │   ├── Infer survey types
>    │   ├── Map simulations to base cases
>    │   └── Clean invalid values
>    │
>    ├── 3. Analysis
>    │   ├── Single simulation evaluation
>    │   ├── Multi-simulation aggregation
>    │   └── Base simulation comparison
>    │
>    └── 4. Visualization
>        ├── Model comparison plots
>        ├── Hyperparameter analysis
>        └── Correlation studies
>
>```


------------------------------------
## Structure
> [!NOTE]
> <details> <summary>High-level</summary>
>     ```└── Data Analysis Pipeline
>         ├── 1. Initial Setup
>         │   ├── Load simulation files
>         │   ├── Configure directories
>         │   └── Import dependencies
>         │
>         ├── 2. Data Processing
>         │   ├── Extract run numbers
>         │   ├── Infer survey types
>         │   ├── Map simulations to base cases
>         │   └── Clean invalid values
>         │
>         ├── 3. Analysis
>         │   ├── Single simulation evaluation
>         │   ├── Multi-simulation aggregation
>         │   └── Base simulation comparison
>         │
>         └── 4. Visualization
>             ├── Model comparison plots
>             ├── Hyperparameter analysis
>             └── Correlation studies
>     ```
> </details>
> <details>
> <summary>Lower-level with key functions</summary>
> ```
> └── Data Analysis Pipeline
>    ├── 1. Setup & Configuration
>    │   ├── Import Dependencies
>    │   ├── Constants Definition
>    │   └── Directory Configuration
>    │
>    ├── 2. Data Loading & Validation
>    │   ├── Simulation File Indexing
>    │   ├── Column Validation
>    │   └── Data Type Verification
>    │
>    ├── 3. Data Preprocessing
>    │   ├── Answer Cleaning
>    │   │   ├── Text Normalization
>    │   │   ├── Pattern Matching
>    │   │   └── Invalid Answer Detection
>    │   │
>    │   ├── Data Mapping
>    │   │   ├── Base Simulation Mapping
>    │   │   ├── Subject Inference
>    │   │   └── Answer Remapping
>    │   │
>    │   └── Data Enrichment
>    │       ├── Survey Type Detection
>    │       └── Answer Integration
>    │
>    ├── 4. Analysis
>    │   ├── Correlation Analysis
>    │   ├── Error Calculation
>    │   └── Statistical Measures
>    │
>    └── 5. Visualization & Reporting
>        ├── Performance Metrics
>        ├── Comparison Plots
>        └── Summary Statistics
> ```
> </details

