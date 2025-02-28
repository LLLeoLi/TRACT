## List of files that need to be changed
- `src/llamafactory/train/tuner.py`: 
    - Add `from .cot_with_raft import run_cot_raft`
    - Add the following block in run_exp:
    ```
    elif finetuning_args.stage == "cot_raft":
        run_cot_raft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    ```
- `LLaMA-Factory/src/llamafactory/data/preprocess.py`:
    - Add `from .processros.custom_processor import preprocess_cot_raft_dataset`
    - Add 
    ```
    elif stage == "cot_raft":
        preprocess_func = partial(
            preprocess_cot_raft_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    ```