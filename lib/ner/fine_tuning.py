import yaml
import json
import argparse

def yaml_to_jsonl(yaml_file: str, jsonl_output_file: str):
    with open(yaml_file, 'r') as yaml_in, open(jsonl_output_file, 'w') as json_out:
        yaml_data = yaml.safe_load(yaml_in)
        
        for item in yaml_data:
            prompt = f"context: {item.get('context')}\n" if item.get("context") else ""
            prompt += f"text: {item.get('text')}"
            prompt += "\n\n###\n\n"
            
            completion = item.get('entities')
            completion = "\n".join(f"{item['name']}: {item['value']}" for item in completion)
            completion = completion if len(completion) > 0 else "No entities"
            completion = f" {completion}\n"
            
            json_line = {
                "prompt": prompt,
                "completion": completion
            }

            json_out.write(json.dumps(json_line) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YAML file to JSONL file")
    parser.add_argument("input", help="Path to the input YAML file")
    parser.add_argument("output", help="Path to the output JSONL file")
    
    args = parser.parse_args()

    yaml_to_jsonl(args.input, args.output)

