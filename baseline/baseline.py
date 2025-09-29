import os
import argparse
from vllm import LLM, SamplingParams

PROMPTS_DIR = './prompts/'


def create_argparser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help="Model name",
        default="meta-llama/Meta-Llama-3-8B-Instruct"
    )


    parser.add_argument('--prompt-name', help="The name (of the file) of the prompt to use (inside prompts dir)", default="prompt1")

    return parser.parse_args()


def create_prompt(prompt_name: str, prompts_dir: str = PROMPTS_DIR, **kwargs) -> str:

    if not prompt_name.endswith('.prompt'):
        prompt_name = prompt_name + '.prompt'

    prompt_path = os.path.join(prompts_dir, prompt_name)
    with open(prompt_path, 'r') as prompt_file:
        prompt = prompt_file.read()

    return prompt.format(**kwargs)



def main():

    args = create_argparser()

    prompt = create_prompt(args.prompt_name)

    llm = LLM(model=args.model)




if __name__ == "__main__":

    raise SystemExit(main())
