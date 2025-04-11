# DrugXplorer Web Application

## Installation required packages
The package requirements are available in the `requirements.txt` file in the repo. 
```bash
python -m venv venv 
source venv/Scripts/activate
pip install -r requirements.txt
```

## Configuring Backend
Following cloning this repository, head to the `base_code.py` file in the Web application folder. We adopt the backend LLM framework from [Mistral AI](https://mistral.ai/). The instructions to generate your own API key is available in this [link](https://docs.mistral.ai/getting-started/quickstart/). Locate to the following code snippet in this file and paste the API key in the required lines. 

```python
os.environ["MISTRAL_API_KEY"] = "" #Insert API key here. 
```

You shall have the option to select your choice of LLM which can be customized in the following code snippet: 

```python
llm = LLM(
    model="mistral-small-latest", #customize to your preference. 
    api_base="https://api.mistral.ai/v1",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2
)
```
Base can be changed to the LLM of your choice inclusing OpenAI, Llama, etc. 

Following these changes, `Base_Code.py` can be run 
```bash
python Base_Code.py
```

## Cite this Work
Coming Soon

## Authors: 
Ashwin Sivakumar, Rishi Senthil Kumar, Jagadeshwar Muthukumaran, Dr. Reena Monica P, Dr. Sheena Chrisabel Pravin 
This work was carried out at MEDxAI Innovations and the Vellore Institute of Technology, Chennai.
