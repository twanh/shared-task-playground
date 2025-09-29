import os
import argparse
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





def create_chat_template(
    system_prompt: str|None,
    user_prompt: str|None,
    history: list|None = None
) -> list[dict[str, str]]:

    conv: list[dict[str,str]] = []

    if system_prompt is not None:
        conv.append({"system": system_prompt})

    if user_prompt is not None:
        conv.append({"user": user_prompt})

    if history is not None:
        return history.append(conv)

    return conv


def main():

    args = create_argparser()


    system_prompt = None
    if args.system_prompt:
        system_prompt = create_prompt(args.system_prompt)
    user_prompt = create_prompt(args.prompt, syllogism="test syllo")


    llm = LLM(model=args.model)

    conversation = create_chat_template(system_prompt, user_prompt)

    outputs = llm.chat(conversation, use_tqdm=True)

    print_outputs(outputs)


if __name__ == "__main__":

    raise SystemExit(main())
