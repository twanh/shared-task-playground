import argparse
import os
import json
from typing import NamedTuple
from vllm import LLM, SamplingParams

PROMPTS_DIR = './prompts/'

def print_outputs(outputs):
    print("\nGenerated Outputs:\n" + "-" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\n")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)


def create_argparser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("data_file", help="Path to the data file (json)")

    parser.add_argument(
        '--model',
        help="Model name",
        default="meta-llama/Meta-Llama-3-20B-Instruct"
    )


    parser.add_argument('--system-prompt', help="The name of the system prompt to use")
    parser.add_argument('--prompt', help="The name (of the file) of the prompt to use (inside prompts dir)", default="prompt1")


    return parser.parse_args()


def create_prompt(
    prompt_name: str,
    prompts_dir: str = PROMPTS_DIR,
    **kwargs
) -> str:

    if not prompt_name.endswith('.prompt'):
        prompt_name = prompt_name + '.prompt'

    prompt_path = os.path.join(prompts_dir, prompt_name)
    with open(prompt_path, 'r') as prompt_file:
        prompt = prompt_file.read()

    try:
        formated_prompt = prompt.format(**kwargs)
    except KeyError as e:
        print(e)
        # Do not format the prompt but return the orignal tempalte
        formated_prompt = prompt
    return formated_prompt





def create_conversation(
    system_prompt: str|None,
    user_prompt: str|None,
    history: list|None = None
) -> list[dict[str, str]]:

    conv: list[dict[str,str]] = []

    if system_prompt is not None:
        conv.append({
            "role": "system",
            "content": system_prompt
        })

    if user_prompt is not None:
        conv.append({
            "role": "user",
            "content": user_prompt
        })

    if history is not None:
        new_conv = history.copy() # Avoid modifying the original list passed in
        if system_prompt is not None:
            new_conv.append({"role": "system", "content": system_prompt})
        if user_prompt is not None:
            new_conv.append({"role": "user", "content": user_prompt})
        return new_conv
    else:
        # Your original logic for creating a new conversation
        conv: list[dict[str, str]] = []
        if system_prompt is not None:
            conv.append({"role": "system", "content": system_prompt})
        if user_prompt is not None:
            conv.append({"role": "user", "content": user_prompt})
        return conv


class DataRow(NamedTuple):

    id_: str
    syllogism: str
    validity: bool
    plausibility: bool

def load_data(file: str) -> list[DataRow]:

    data: list[DataRow] = []

    with open(file, 'r') as data_file:
        content = json.load(data_file)

    for row in content:
        # TODO: Think about the defaults/error handling
        data_row = DataRow(
            id_ = row.get('id', 'NO-ID'),
            syllogism=row.get('syllogism', 'NO-SYLLOGISM'),
            validity=row.get('validity', False),
            plausibility=row.get('plausibility', False),
        )
        data.append(data_row)


    return data


def parse_output(output: str) -> bool:

    output = output.lower()
    resp = json.loads(output)
    return resp.get('validity', None)

class ResultRow(NamedTuple):

    id_: str
    syllogism: str
    validity: bool
    plausibility: bool
    predicted_validity: bool|None = None


def main():

    args = create_argparser()

    data = load_data(args.data_file)

    print(f"Loaded {len(data)} data rows")

    system_prompt = None
    if args.system_prompt:
        system_prompt = create_prompt(args.system_prompt)


    llm = LLM(model=args.model)

    results = []
    correct = 0
    total = 0

    for i, row in enumerate(data):
        user_prompt = create_prompt(
            args.prompt,
            syllogism=row.syllogism,
        )

        conv = create_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        #sampling_params = SamplingParams(temperature=0, max_tokens=512)

        outputs = llm.chat(
            conv,
        #    sampling_params=sampling_params
        )

        print(f'=== Data row {i+1}/{len(data)} ===')
        print(f"Data row ID: {row.id_}")
        print(f"Syllogism: {row.syllogism}")
        print(f"Validity: {row.validity}")
        print("Prompt: ", end="")
        print(outputs[0].prompt)

        result = parse_output(outputs[0].outputs[0].text)
        print(f"Predicted validity: {result}")

        result_row = ResultRow(
            id_=row.id_,
            syllogism=row.syllogism,
            validity=row.validity,
            plausibility=row.plausibility,
            predicted_validity=result
        )
        results.append(result_row)

        if result == row.validity:
            correct += 1

        total += 1


    print(f"Correct: {correct}/{total}")
    accuracy = correct/total
    print(f"Accuracy: {accuracy}")

    # Save results
    # TODO: Make output file configurable
    results_file = args.data_file.replace('.json', '_results.json')
    with open(results_file, 'w') as out_file:
        json.dump([r._asdict() for r in results], out_file, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":

    raise SystemExit(main())
