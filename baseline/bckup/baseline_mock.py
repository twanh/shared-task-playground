#!/usr/bin/env python3
"""
Mock version of baseline.py for testing without vLLM server.
This version simulates the LLaMA responses for development/testing.
"""

import argparse
import json
import random
from typing import List, Dict, Any
from baseline import DataRow, read_input_file, create_prompt

def mock_llm_response(syllogism: str) -> Dict[str, Any]:
    """Mock LLM response for testing purposes."""
    # Simple heuristic: check for common invalid patterns
    syllogism_lower = syllogism.lower()
    
    # Very basic pattern matching for demonstration
    if "all" in syllogism_lower and "therefore" in syllogism_lower:
        if "are" in syllogism_lower and syllogism_lower.count("all") >= 2:
            validity = random.choice([True, False])  # Random for demo
        else:
            validity = True
    else:
        validity = random.choice([True, False])
    
    explanations = [
        "Valid by universal instantiation",
        "Invalid due to undistributed middle",
        "Valid by modus ponens",
        "Invalid affirming the consequent",
        "Valid syllogistic form",
        "Invalid existential fallacy"
    ]
    
    return {
        "validity": validity,
        "explanation": random.choice(explanations)
    }

def process_syllogism_mock(prompt_template: str, syllogism_data: DataRow) -> Dict[str, Any]:
    """Process a single syllogism with mock LLM."""
    result = mock_llm_response(syllogism_data.syllogism)
    result["id"] = syllogism_data.id
    return result

def main():
    parser = argparse.ArgumentParser(description="Mock Baseline for SharedTask")
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt file')
    parser.add_argument('input_file', type=str, help='Path to input JSON file')
    parser.add_argument('output_file', type=str, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    # Load prompt template
    prompt_template = create_prompt(args.prompt)
    
    # Read input data
    print(f"Reading syllogisms from {args.input_file}...")
    syllogisms = read_input_file(args.input_file)
    print(f"Loaded {len(syllogisms)} syllogisms")
    
    # Process each syllogism with mock LLM
    results = []
    for i, syllogism_data in enumerate(syllogisms):
        if i % 100 == 0:
            print(f"Processing syllogism {i+1}/{len(syllogisms)}")
        
        result = process_syllogism_mock(prompt_template, syllogism_data)
        results.append(result)
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate accuracy
    correct = sum(1 for i, result in enumerate(results) 
                  if result.get("validity") == syllogisms[i].validity)
    accuracy = correct / len(results)
    print(f"Mock Accuracy: {accuracy:.3f} ({correct}/{len(results)})")

if __name__ == "__main__":
    main()