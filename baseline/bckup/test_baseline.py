#!/usr/bin/env python3
"""
Test script for the baseline syllogism classifier.
This script tests the baseline system with a small subset of data.
"""

import json
import subprocess
import sys
import os

def create_test_data():
    """Create a small test dataset."""
    test_data = [
        {
            "id": "test-1",
            "syllogism": "All cats are animals. Fluffy is a cat. Therefore, Fluffy is an animal.",
            "validity": True,
            "plausibility": True
        },
        {
            "id": "test-2", 
            "syllogism": "All birds can fly. Penguins are birds. Therefore, penguins can fly.",
            "validity": True,
            "plausibility": False
        },
        {
            "id": "test-3",
            "syllogism": "All cats are animals. All dogs are animals. Therefore, all cats are dogs.",
            "validity": False,
            "plausibility": False
        }
    ]
    
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return 'test_data.json'

def main():
    print("Creating test data...")
    test_file = create_test_data()
    
    print("Testing baseline system...")
    
    # Check if vLLM server is running (this will fail gracefully)
    cmd = [
        sys.executable, 'baseline.py',
        '--model', 'llama',
        '--prompt', 'prompt1.prompt',
        test_file,
        'test_results.json'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Baseline system executed successfully")
            print("Output:", result.stdout)
            
            # Check if results file was created
            if os.path.exists('test_results.json'):
                with open('test_results.json', 'r') as f:
                    results = json.load(f)
                print(f"✓ Generated {len(results)} predictions")
                
                # Show sample results
                for result in results[:2]:
                    print(f"  ID: {result.get('id')}")
                    print(f"  Validity: {result.get('validity')}")
                    print(f"  Explanation: {result.get('explanation', 'N/A')}")
                    print()
            else:
                print("✗ Results file not created")
                
        else:
            print("✗ Baseline system failed")
            print("Error:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out (likely vLLM server not running)")
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
    
    # Cleanup
    for file in ['test_data.json', 'test_results.json']:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()