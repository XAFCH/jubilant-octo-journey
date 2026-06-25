import json
import os
import pandas as pd
import openai
from openai import AzureOpenAI
import unicodedata

def preprocess_text(text):
    # Normalize Unicode characters to NFC form
    text = unicodedata.normalize('NFC', text)

    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

    # Optional: Standardize punctuation
    text = text.translate({
        ord('â€?): ord("'"),
        ord('â€?): ord("'"),
        ord('â€?): ord('"'),
        ord('â€?): ord('"'),
        ord('â€?): ord('-'),
        ord('â€?): ord('-'),
    })

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def generate_path_description(abstract, label_path):

    # Construct the label path string
    label_path_str = '\n'.join([f"- Level {i+1}: {label}" for i, label in enumerate(label_path)])
    
    # Create the prompt
    prompt = f"""Abstract:"{abstract}"

Hierarchical label path:
{label_path_str}

Task:
Generate a detailed description that:
- Defines each label in the hierarchy, and the relationship between each level.
- Shows how the abstract's content is connected to each label. Avoid introducing unrelated information.
- Uses clear and professional language appropriate for an academic audience.
- Focus solely on the information provided in the abstract and labels. Do not include personal opinions or external information.
- Is approximately 200 words in length.
"""
    
    # Call the GPT-4 API
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key= os.getenv("AZURE_OPENAI_KEY")
    deployment_name = 'gpt-4o'
    api_version = '2024-02-01'

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )

    response = client.chat.completions.create(
        model = deployment_name,
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 300,
        temperature = 0.7,
    )
    
    # Extract and return the generated description
    description = response.choices[0].message.content
    return description

def get_new_dataset():
    f = open('data/BGC/bgc_test_raw.json', 'r', encoding='utf-8')
    data = f.readlines()
    f.close()
    print('Finish loading dataset!')
    # new_data = []
    with open('data/BGC/bgc_test_new.json', 'w') as f_out:
        for i, line in enumerate(data):
            line = json.loads(line)
            abstract = line['token']
            abstract = preprocess_text(abstract)
            label_path = line['path']
            line['path_description'] = []
            for path in label_path:
                try:
                    description = generate_path_description(abstract, path)
                except openai.BadRequestError as e:
                    print(f"Example {i+1} cannot be processed!")
                    print(f"BadRequestError: {e}")
                    break
                line['path_description'].append(description)
            write_line = json.dumps(line)
            f_out.write(write_line + '\n')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["AZURE_OPENAI_KEY"] = os.getenv("AZURE_OPENAI_KEY", "")
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai.azure.com/"
    get_new_dataset()

