> Code for the paper, "LLMs as Proxy Survey Participants With RAG", by Elias Torjani, Airidas Brikas, and Daniel Hardt (our BSc thesis)

Check out our [abstract-length] paper on it: [Market research via persona-induced Large Language Models](https://sltc2024.github.io/abstracts/torjani.pdf), or see our poster below as a TL;DR

![Poster](Poster.png)

## How to reproduce our experiments with your own data
1. Export your chat messages from Facebook, Instagram, and/or WhatsApp (instructions below)
2. Take the surveys to constitute target responses, for the LLMs proxying you in the same surveys.
3. Clone this repository
4. Download [Ollama](https://ollama.com/download/) and your models of choice, to run inference locally. 
   - *Any cloud provider is discouraged to mitigate leakage-risk of sensitive information.*


> [!IMPORTANT]
> **Exporting chat messages from Facebook, Instagram, and/or WhatsApp**
> This is a relatively manual process, and Meta will take about a week.
> 1. [Facebook](https://accountscenter.facebook.com/info_and_permissions/dyi) incl. Instagram --> Account settings --> Download your information --> Download or transfer information --> pick account[s] (incl. Instagram) --> Specific types of information --> choose "Messages"
>   - Get "All time", and in JSON format
> 2. WhatsApp --> Settings --> Chats -->  Export chat --> pick your 1-on-1 chats to export "Without media"
> 3. Optional: Use Beeper's API to continuosly export new messages, but be aware of our experiment is a snapshot in time.

------------------------------------
## Structure of repository

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
>
></details>
>
