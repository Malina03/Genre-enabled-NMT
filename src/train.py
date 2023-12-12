'''Fine-tune a pre-trained model from Huggingface on a new dataset.'''

from transformers import AutoTokenizer, EarlyStoppingCallback, AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoConfig
from utils import get_args, get_train_args, load_data, compute_metrics, train_tokenizer, update_model_config
import wandb
from functools import partial
import os

if __name__ == "__main__":
    
    args = get_args()
    if args.wandb:
        # only log the training process 
        if args.genre:
            wandb_name = f"{args.train_file.split('/')[-1].split('.')[1]}_{args.exp_type}_{args.model_type}_{args.genre}"
        else:
            wandb_name = f"{args.train_file.split('/')[-1].split('.')[1]}_{args.exp_type}_{args.model_type}"
        # Initialize wandb
        wandb.init(project="genre_NMT", name=wandb_name, config=args)

    
    # Load the data
   
    if args.train_tokenizer:
        print("Training tokenizer")
        tokenizer = train_tokenizer(args)
        print("Tokenizer trained.")
    elif args.use_costum_tokenizer:
        print("Using costum tokenizer from:", args.tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    else:
        print("Using pretrained-tokenizer from:", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if "genre_aware_token" in args.model_type:
            if args.old_tokens or "old_tokens" in args.train_file:
                tags = ['>>info<<', '>>promo<<', '>>news<<', '>>law<', '>>other<<', '>>arg<<', '>>instr<<', '>>lit<<', '>>forum<<']
            else:
                tags = ['<info>', '<promo>', '<news>', '<law>', '<other>', '<arg>', '<instr>', '<lit>', '<forum>']
            tokenizer.add_special_tokens({'additional_special_tokens': tags})
            print("Added genre tokens to tokenizer")


    train_dataset = load_data(args.train_file, args, tokenizer=tokenizer)
    dev_dataset= load_data(args.dev_file, args, tokenizer=tokenizer)

    if args.eval or args.predict:
        test_dataset = load_data(args.test_file, args, tokenizer=tokenizer)

 
    # Load the model
    if args.checkpoint is None:
        # config = AutoConfig.from_pretrained(args.model_name)
        if "from_scratch" in args.exp_type:
            print("Training from scratch")
            config = AutoConfig.from_pretrained(args.model_name)
            model = AutoModelForSeq2SeqLM.from_config(config)
            if args.train_tokenizer:
                model.resize_token_embeddings(len(tokenizer))
                config = update_model_config(config, tokenizer, args)
        else:
            # load the pretrained model to fine-tune it
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        if "genre_aware_token" in args.model_type:
            model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True)

    # Set the training arguments
    training_args = get_train_args(args)

  
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping, early_stopping_threshold=args.early_stopping_threshold)
    ]
    
    # Instantiate the trainer
    trainer =  Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=16),
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer, args=args),
        callbacks=callbacks
    )

    if args.eval:
        if args.predict:
            output = trainer.predict(test_dataset=test_dataset)
            preds = output.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            decode_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if args.use_costum_tokenizer or args.train_tokenizer:
                # clean up the SentencePiece tokenization
                decode_preds = [pred.replace("▁", " ") for pred in decode_preds]
                # remove leading and trailing spaces
            predictions = [pred.strip() for pred in decode_preds]
            if args.genre:
                logging_dir = os.path.join(args.root_dir, "eval", args.exp_type, args.model_type, args.genre)
            else:
                logging_dir = os.path.join(args.root_dir, "eval", args.exp_type, args.model_type)
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir)
            eval_corpus = args.test_file.split("/")[-1].split(".")[0]
            with open(os.path.join(logging_dir, f'{eval_corpus}_predictions.txt'), "w") as f:
                for pred in predictions:
                    f.write(pred + "\n")
            print("\nInfo:\n", output.metrics, "\n")
            print("Tested on:", args.test_file)
            print('Predictions saved to:', os.path.join(logging_dir, f'{eval_corpus}_predictions.txt'))
            if args.checkpoint is not None:
                print("Model from:", args.checkpoint)
            else:
                print("Baseline model:", args.model_name)
     
        else:
            metrics = trainer.evaluate()
            print("\nInfo:\n", metrics, "\n")

    else:
        ## evaluate the baseline model before training
        if args.eval_baseline:
            metrics = trainer.evaluate()
            print("\nBaseline metrics:\n", metrics, "\n")
        metrics = trainer.train()
        print("\nInfo:\n", metrics, "\n")