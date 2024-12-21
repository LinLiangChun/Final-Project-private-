import re
import faiss
import random
import numpy as np
from base import Agent
from colorama import Fore, Style
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import warnings
from transformers import logging as transformers_logging

from utils import RAG, strip_all_lines

# Ignore warning messages from transformers
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

class AdaptiveRAG:
    def __init__(self, rag_config: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(rag_config["embedding_model"])
        self.embed_model = AutoModel.from_pretrained(rag_config["embedding_model"]).eval()
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.index = faiss.IndexFlatL2(self.embed_dim)
        self.id2evidence = {}
        self.insert_acc = 0
        self.top_k = rag_config["top_k"]
        
        self.default_weight = 1.0
        
        self.retrieve_count = {}
        self.insert_order = {}

    def encode_data(self, text: str) -> np.ndarray:
        # tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # with torch.no_grad():
        #     embeddings = self.embed_model(**tokens).last_hidden_state[:, 0, :]
        # return embeddings.squeeze().numpy()

        # Tokenize the sentence
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0]
        norm = np.linalg.norm(feature)
        return feature / norm

    def insert(self, key: str, value: str):
        embedding = self.encode_data(key).astype("float32")
        self.index.add(embedding[np.newaxis, :])
        self.id2evidence[str(self.insert_acc)] = value
        self.retrieve_count[str(self.insert_acc)] = 0
        self.insert_order[str(self.insert_acc)] = self.insert_acc
        self.insert_acc += 1

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        embedding = self.encode_data(query).astype("float32")
        distances, indices = self.index.search(embedding[np.newaxis, :], self.top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            evidence = self.id2evidence.get(str(idx), None)
            if evidence:
                results.append((evidence, 1.0 / (dist + 1e-5)))
                self.retrieve_count[str(idx)] = self.retrieve_count.get(str(idx), 0) + 1
        
        return results

    def adjust_weights(self, retrieval_scores):
        max_score = max(retrieval_scores) if retrieval_scores else 1.0
        weights = [score / max_score for score in retrieval_scores]
        
        return weights

    def update_memory(self, top_k: int) -> None:
        sorted_examples = sorted(
            self.id2evidence.keys(),
            key=lambda x: (self.retrieve_count.get(x, 0), self.insert_order.get(x, 0)),
            reverse=True
        )
        keep_indices = set(sorted_examples[:top_k])
        
        self.id2evidence = {k: v for k, v in self.id2evidence.items() if k in keep_indices}
        self.retrieve_count = {k: v for k, v in self.retrieve_count.items() if k in keep_indices}
        self.insert_order = {k: v for k, v in self.insert_order.items() if k in keep_indices}
        
        self.index.reset()
        for idx, evidence in self.id2evidence.items():
            embedding = self.encode_data(evidence).astype("float32")
            self.index.add(embedding[np.newaxis, :])
            
        print('Memory update!')

class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    @staticmethod
    def get_system_prompt() -> str:
        '''
        system_prompt = """\
        Act as a professional medical doctor that can diagnose the patient based on the patient profile.
        Provide your diagnosis in the following format: <number>. <diagnosis>""".strip()
        '''
        
        system_prompt = """\
        Act as a professional medical doctor that can diagnose the patient based on the patient profile and provide the reasoning process concisely.
        Provide your diagnosis and reasoning concisely in the following format (within 100 words):
        Diagnosis: <number>. <diagnosis>
        Reasoning: <concise reasoning>""".strip()
        
        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(option_text: str, text: str) -> str:
        '''
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the following patient profile:
        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Now, directly provide the diagnosis for the patient in the following format: <number>. <diagnosis>""".strip()
        '''
        
        prompt = f"""\ 
        Act as a medical doctor and diagnose the patient based on the following patient profile:
        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Provide your diagnosis and reasoning concisely in the following format (within 100 words):
        Diagnosis: <number>. <diagnosis>
        Reasoning: <concise reasoning>""".strip()
        
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        '''
        prompt = f"""\
        {{question}}
        Diagnosis: {{answer}}"""
        '''
        
        prompt = f"""\ 
        Patient Profile: {{question}}
        Diagnosis: {{answer}}
        Reasoning: {{reasoning}}"""
        
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(option_text: str, text: str,) -> str:
        '''
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the provided patient profile.
        
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Here are some example cases.
        
        {{fewshot_text}}
        
        Now it's your turn.
        
        {text}        
        
        Now provide the diagnosis for the patient in the following format: <number>. <diagnosis>"""
        '''
        
        prompt = f"""\ 
        Act as a medical doctor and diagnose the patient based on the provided patient profile.
        
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Here are some example cases.
        
        {{fewshot_text}}
        
        Now it's your turn.
        
        {text} 

        Provide your diagnosis and reasoning concisely in the following format (within 100 words):
        Diagnosis: <number>. <diagnosis>
        Reasoning: <concise reasoning>"""
        
        return strip_all_lines(prompt)

    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        '''
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        '''
        
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        diagnosis = re.search(r"Diagnosis:\s*(\d+\..*?)(?=\s*Reasoning:|$)", generated_text, re.S)
        reasoning = re.search(r"Reasoning:\s*(.*)", generated_text, re.S)

        return {
            "diagnosis": diagnosis.group(1).strip() if diagnosis else "No Diagnosis",
            "reasoning": reasoning.group(1).strip() if reasoning else "No Reasoning"
        }

    @staticmethod
    def extract_label(response: dict, label2desc: dict[str, str]) -> tuple:
        pred_text = response.get("diagnosis", "")
        reasoning = response.get("reasoning", "")
        
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)
        
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        
        return str(prediction), reasoning

    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        
        # TODO
        super().__init__(config)
        self.llm_config = config
        if config['use_8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        
        self.rag = AdaptiveRAG(config["rag"])
        
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        self.reasoning_logs = None
        
        self.model.eval()

    def __call__(self, label2desc: dict[str, str], text: str) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        
        # TODO
        self.reset_log_info()
        option_text = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(option_text, text)
        prompt_fewshot = self.get_fewshot_template(option_text, text)
        
        '''
        shots = self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        '''
        
        retrieval_results = self.rag.retrieve(query=text)
        docs, scores = zip(*retrieval_results) if retrieval_results else ([], [])

        weights = self.rag.adjust_weights(scores)
        shots = [f"[Weight: {weight:.2f}] {doc}" for doc, weight in zip(docs, weights)]

        if self.rag.insert_acc >= 50:
            if len(shots) > 0:
                fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
                try:
                    prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                    print(Fore.RED + error_msg + Fore.RESET)
                    prompt = prompt_zeroshot
            else:
                print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
                prompt = prompt_zeroshot
        else:
            prompt = prompt_zeroshot

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.generate_response(messages)
        prediction, reasoning = ClassificationAgent.extract_label(response, label2desc)

        '''
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(system_prompt + prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": response,
        })
        
        self.inputs.append(text)
        self.self_outputs.append(f"{str(prediction)}. {label2desc[int(prediction)]}")
        '''
        
        self.reasoning_logs = {
            "input": text,
            "reasoning": reasoning,
            "diagnosis": f"{str(prediction)}. {label2desc[int(prediction)]}"
        }
        
        return prediction
    
    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        
        # TODO
        '''
        if correctness:
            question = self.inputs[-1]
            answer = self.self_outputs[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)
        '''
        
        if correctness and self.reasoning_logs:
            question = self.reasoning_logs["input"]
            reasoning = self.reasoning_logs["reasoning"]
            diagnosis = self.reasoning_logs["diagnosis"]
            chunk = f"{question}\nDiagnosis: {diagnosis}"
            self.rag.insert(key=question, value=chunk)

            print(reasoning)
            
            return True
        return False

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        super().__init__(config)
        self.llm_config = config
        if config['use_8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        
        self.rag = RAG(config["rag"])
        
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        self.reasoning_logs = None
        
        self.model.eval()

    @staticmethod
    def get_system_prompt() -> str:
        system_prompt = """\
        Act as a professional programmer.
        You will be given a table schema and a user query, and you need to generate the correct SQL code to answer the user query in the following format:
        ```sql\n<your_SQL_code>\n```"""
        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(table_schema: str, user_query: str) -> str:
        prompt = f"""\
        {table_schema}
        
        -- Using valid SQLite, answer the following question for the tables provided above.
        -- Question: {user_query}
        
        Now, generate the correct SQL code directly in the following format:
        ```sql\n<your_SQL_code>\n```"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        Question: {{question}}
        {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(table_schema: str, user_query: str) -> str:
        prompt = f"""\
        You are performing the text-to-SQL task. Here are some examples:
        
        {{fewshot_text}}
        
        Now it's your turn.
        
        -- SQL schema: {table_schema}
        -- Using valid SQLite, answer the following question for the SQL schema provided above.
        -- Question: {user_query}
        
        Now, generate the correct SQL code directly in the following format:
        ```sql\n<your_SQL_code>\n```"""
        return strip_all_lines(prompt)

    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    @staticmethod
    def parse_sql(pred_text: str) -> str:
        """
        Parse the SQL code from the LLM's response.
        """
        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code
        else:
            print(Fore.RED + "No SQL code found in the response" + Style.RESET_ALL)
            sql_code = pred_text
        return sql_code

    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        self.reset_log_info()
        prompt_zeroshot = self.get_zeroshot_prompt(table_schema, user_query)
        prompt_fewshot = self.get_fewshot_template(table_schema, user_query)
        
        shots = self.rag.retrieve(query=user_query, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []

        # retrieval_results = self.rag.retrieve(query=user_query) if (self.rag.insert_acc > 0) else None
        # docs, scores = zip(*retrieval_results) if retrieval_results else ([], [])

        # weights = self.rag.adjust_weights(scores)
        # shots = [f"{doc}" for doc, weight in zip(docs, weights)]

        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Style.RESET_ALL)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        pred_text = self.generate_response(messages)
        sql_code = self.parse_sql(pred_text)
        
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(self.get_system_prompt() + prompt)),
            "num_output_tokens": len(self.tokenizer.encode(pred_text)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": pred_text,
        })

        self.inputs.append(user_query)
        self.self_outputs.append(f"```sql\n{sql_code}\n```")
        return sql_code

    def update(self, correctness: bool) -> bool:
        """
        Update the agent based on the correctness of its output.
        """
        if correctness:
            question = self.inputs[-1]
            answer = self.self_outputs[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)
            return True
        return False
        
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--output_path', type=str, default=None, help='path to save csv file for kaggle submission')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        max_tokens = 128
        agent_name = ClassificationAgent
        EMBEDDING = 'sentence-transformers/all-mpnet-base-v2'
    elif args.bench_name.startswith("sql_generation"):
        max_tokens = 512
        agent_name = SQLGenerationAgent
        EMBEDDING = 'BAAI/bge-large-en-v1.5'
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    llm_config = {
        # TODO: specify your configs for the agent here
        'model_name': args.model_name,
        'exp_name': f'adaptive_rag_{args.bench_name}_{args.model_name}',
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': max_tokens,
        'do_sample': False,
        'device': args.device,
        'use_8bit': args.use_8bit,
        'rag': {
            #'embedding_model': 'BAAI/bge-base-en-v1.5',
            'embedding_model': EMBEDDING,
            'seed': 42,
            'top_k': 16,
            'order': 'similar_at_top',
        }
    }
    agent = agent_name(llm_config)
    main(agent, bench_cfg, debug=args.debug, use_wandb=args.use_wandb, wandb_name=llm_config["exp_name"], wandb_config=llm_config)
