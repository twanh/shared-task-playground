import argparse
import json
import re
from typing import NamedTuple, List, Dict, Any

from openai import OpenAI
from vllm import LLM, SamplingParams


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8192/v1"

def parse_args():

    parser = argparse.ArgumentParser(description="Baseline for SharedTask")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model"
    )

    parser.add_argument('--prompt',
                        type=str,
                        help='Input prompt for text generation')

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input file containing the syllogisms'
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Path to the output file to save the results'
    )

    return parser.parse_args()

def create_prompt(prompt_name, **kwargs):

    with open(prompt_name, 'r') as f:
        prompt = f.read()

    return prompt.format(**kwargs)


class DataRow(NamedTuple):
    id: str
    syllogism: str
    validity: bool
    plausibility: bool

def read_input_file(input_file: str) -> List[DataRow]:
    """Read and parse the JSON input file containing syllogisms."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return [DataRow(
        id=item['id'],
        syllogism=item['syllogism'],
        validity=item['validity'],
        plausibility=item['plausibility']
    ) for item in data]

def extract_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON object from model response."""
    # Look for JSON object in the response
    json_match = re.search(r'\{[^{}]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to parse the entire response as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Return a default response if parsing fails
        return {"validity": False, "explanation": "Failed to parse response"}

def process_syllogism(client: OpenAI, prompt_template: str, syllogism_data: DataRow) -> Dict[str, Any]:
    """Process a single syllogism and return the model's prediction."""
    # Create the full prompt with the syllogism
    full_prompt = prompt_template + f"\n\nSyllogism: {syllogism_data.syllogism}\nID: {syllogism_data.id}\n"
    
    try:
        response = client.completions.create(
            model="llama",  # This will be replaced by the actual model path
            prompt=full_prompt,
            max_tokens=150,
            temperature=0.0,
            stop=["\n\n"]
        )
        
        response_text = response.choices[0].text.strip()
        result = extract_json_response(response_text)
        
        # Ensure required fields are present
        result["id"] = syllogism_data.id
        if "validity" not in result:
            result["validity"] = False
        if "explanation" not in result:
            result["explanation"] = "No explanation provided"
            
        return result
        
    except Exception as e:
        return {
            "id": syllogism_data.id,
            "validity": False,
            "explanation": f"Error: {str(e)[:50]}"
        }

def main():
    args = parse_args()

    # Initialize OpenAI client for vLLM
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Load the prompt template
    prompt_template = create_prompt(args.prompt)
    
    # Read input data
    print(f"Reading syllogisms from {args.input_file}...")
    syllogisms = read_input_file(args.input_file)
    print(f"Loaded {len(syllogisms)} syllogisms")
    
    # Process each syllogism
    results = []
    for i, syllogism_data in enumerate(syllogisms):
        if i % 10 == 0:
            print(f"Processing syllogism {i+1}/{len(syllogisms)}")
        
        result = process_syllogism(client, prompt_template, syllogism_data)
        results.append(result)
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate accuracy if we have ground truth
    correct = sum(1 for i, result in enumerate(results) 
                  if result.get("validity") == syllogisms[i].validity)
    accuracy = correct / len(results)
    print(f"Accuracy: {accuracy:.3f} ({correct}/{len(results)})")

if __name__ == "__main__":
    main()

