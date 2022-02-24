import torch
import numpy as np
import random
import torch.nn.functional as F
import argparse
import os

from tqdm import trange
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)

from datasets import load_from_disk
from torch.utils.data import DataLoader

from dataset import (
    BiEncoder_Dataset_Original,
    BiEncoder_Dataset_Overflow,
    CrossEncoder_Dataset,
)

from utils import CustomSampler

from encoder import (
    BertEncoder_For_CrossEncoder,
    RoBertaEncoder_For_CrossEncoder,
    BertEncoder_For_BiEncoder,
)


def set_seed(random_seed):
    """
    Random number fixed
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


def biencoder_train(
    args,
    queries,
    passages,
    tokenizer,
    p_encoder,
    q_encoder,
    sampler=None,
    overflow=True,
):
    """
    In-batch Negative BiEncoder Train

    Arg:
        queires: List
        passages: List
        tokenizer: BertTokenizer
        p_encoder: BertEncoder_For_BiEncoder
        q_encoder: BertEncoder_For_BiEncoder
        sampler: Sampler
            you can use the CustomSampler
            if you don't want to use CustomSampler,
             you have to insert 'shuffle=True' in your DataLoader
        overflow: bool
            If you want data with overflow technique,
             keep overflow as true, and if you want to use data
             that simply cut passage into max_length, use False.
    """
    if overflow == True:
        overflow_biencoder = BiEncoder_Dataset_Overflow(
            queries, passages, tokenizer)
        biencoder_dataset = overflow_biencoder._return_train_dataset()
    else:
        overflow_biencoder = BiEncoder_Dataset_Original(
            queries, passages, tokenizer)
        biencoder_dataset = overflow_biencoder._return_train_dataset()

    if sampler is not None:
        sampler = sampler(biencoder_dataset, args.per_device_train_batch_size)
        train_dataloader = DataLoader(
            biencoder_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=sampler,
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            biencoder_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in p_encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in p_encoder.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in q_encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in q_encoder.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        # eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    p_encoder.zero_grad()
    q_encoder.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    q_encoder.train()
    p_encoder.train()
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        loss_value = 0  # Use it when you use accumulation.
        losses = 0
        for step, batch in enumerate(epoch_iterator):
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

            # Calculate the similarity & loss score for "in batch negative".
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element
            # targets = torch.arange(0, args.per_device_train_batch_size).long()
            targets = torch.arange(0, len(p_inputs["input_ids"])).long()

            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)

            loss = F.nll_loss(sim_scores, targets)

            ########################No ACCUMULATION#########################
            losses += loss.item()
            if step % 100 == 0:
                print(f"{epoch}epoch loss: {losses/(step+1)}")

            q_encoder.zero_grad()
            p_encoder.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ################################################################

            # #############################ACCUMULATION#########################
            # loss.backward()
            # if (step+1) % args.gradient_accumulation_steps == 0 :
            #     optimizer.step()
            #     scheduler.step()
            #     self.q_encoder.zero_grad()
            #     self.p_encoder.zero_grad()

            # losses += loss.item()
            # if (step+1) % 100 == 0 :
            #     train_loss = losses / 100
            #     print(f'training loss: {train_loss:4.4}')
            #     losses = 0
            # ##################################################################

            del p_inputs, q_inputs

    return p_encoder, q_encoder


def crossencoder_train(args, queries, passages, tokenizer, cross_encoder, sampler=None):
    """
    In-batch Negative CrossEncoder Train

    Arg:
        queries: List
        passages: List
        tokenizer: BertTokenizer or RoBertaTokenizer
        cross_encoder: BertEncoder_For_CrossEncoder or RoBertaEncoder_For_CrossEncoder
        sampler: Sampler
            you can use the CustomSampler
            if you don't want to use CustomSampler,
             you have to insert 'shuffle=True' in your DataLoader
    """
    crossencoder_dataset = CrossEncoder_Dataset(queries, passages, tokenizer)
    train_dataset = crossencoder_dataset._return_train_dataset()

    if sampler is not None:
        sampler = sampler(train_dataset, args.per_device_train_batch_size)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=sampler,
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in cross_encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in cross_encoder.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        # eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    cross_encoder.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    cross_encoder.train()

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        losses = 0
        
        for step, batch in enumerate(epoch_iterator):
            cross_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # 'token_type_ids' : batch[2] # When you use BertModel, Unannotate it
            }
            for k in cross_inputs.keys():
                cross_inputs[k] = cross_inputs[k].tolist()

            # -- Make In-Batch Negative Sampling
            new_input_ids = []
            new_attention_mask = []
            # new_token_type_ids = [] # When you use BertModel, Unannotate it
            
            for i in range(len(cross_inputs["input_ids"])):
                sep_index = cross_inputs["input_ids"][i].index(tokenizer.sep_token_id)  # [SEP] token의 index

                for j in range(len(cross_inputs["input_ids"])):
                    
                    # -- Make Negative Samples => i_th query with j_th passage
                    # positive: i_th query + i_th query
                    # negative: i_th query + j_th query
                    # Note: Since multiple passages can be obtained for one query, the i_th query and j_th passage can be positive samples. Because of this, Sampling is performed in prepraration for this case. However, there is no significant difference in performance when shuffle is used as sampling
                    
                    query_id = cross_inputs["input_ids"][i][:sep_index]
                    query_att = cross_inputs["attention_mask"][i][:sep_index]
                    # query_tok = cross_inputs['token_type_ids'][i][:sep_index] # When you use BertModel, Unannotate it

                    context_id = cross_inputs["input_ids"][j][sep_index:]
                    context_att = cross_inputs["attention_mask"][j][sep_index:]
                    # context_tok = cross_inputs['token_type_ids'][j][sep_index:] # When you use BertModel, Unannotate it
                    
                    query_id.extend(context_id)
                    query_att.extend(context_att)
                    # query_tok.extend(context_tok) # When you use BertModel, Unannotate it
                    
                    new_input_ids.append(query_id)
                    new_attention_mask.append(query_att)
                    # new_token_type_ids.append(query_tok) # When you use BertModel, Unannotate it

            new_input_ids = torch.tensor(new_input_ids)
            new_attention_mask = torch.tensor(new_attention_mask)
            # new_token_type_ids = torch.tensor(new_token_type_ids) # When you use BertModel, Unannotate it
            
            if torch.cuda.is_available():
                new_input_ids = new_input_ids.to("cuda")
                new_attention_mask = new_attention_mask.to("cuda")
                # new_attention_mask = new_attention_mask.to('cuda') # When you use BertModel, Unannotate it

            change_cross_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                # 'token_type_ids' : new_token_type_ids # When you use BertModel, Unannotate it
            }

            cross_output = cross_encoder(**change_cross_inputs) 
            cross_output = cross_output.view(-1, args.per_device_train_batch_size) # (batch_size, emb_dim)
            
            # only i_th element is accepted as positive
            targets = torch.arange(0, args.per_device_train_batch_size).long()

            if torch.cuda.is_available():
                targets = targets.to("cuda")

            score = F.log_softmax(cross_output, dim=1)
            loss = F.nll_loss(score, targets)
            ########################No ACCUMULATION#########################
            losses += loss.item()
            if step % 100 == 0:
                print(f"{epoch}epoch loss: {losses/(step+1)}")

            cross_encoder.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ################################################################

            # #############################ACCUMULATION#########################
            # loss.backward()
            # if (step+1) % args.gradient_accumulation_steps == 0 :
            #     optimizer.step()
            #     scheduler.step()
            #     cross_encoder.zero_grad()

            # losses += loss.item()
            # if (step+1) % 100 == 0 :
            #     train_loss = losses / 100
            #     print(f'training loss: {train_loss:4.4}')
            #     losses = 0
            # ##################################################################

    return cross_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # -- mode
    parser.add_argument('--encoder', type=str, default='cross', help='Biencoder can be used as the instruction "bi" and crossencoder can be used as the instruction "cross".')
    parser.add_argument('--model', type=str, default='klue/bert-base', help='You can insert "klue/bert-base" or "klue/roberta-base" or "klue/roberta-base"')

    # -- training arguments
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate (default: 1e-5)")
    parser.add_argument('--train_batch_size', type=int, default=4, help="train batch size (default: 4)")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="strength of weight decay (default: 0.01)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient accumulation steps (default: 1)")
    
    # -- save
    parser.add_argument('--output_directory', type=str, default='./save_directory/', help='Put in your save directory')
    parser.add_argument('--input_directory', type=str, default='./_data/', help='Enter input_directory containing Encoder.')

    sub_args = parser.parse_args()

    args = TrainingArguments(
        output_dir=sub_args.output_directory,
        evaluation_strategy="epoch",
        learning_rate=sub_args.lr,
        # if you use bi-encoder, More batch size may be input than crossencoder.
        per_device_train_batch_size=sub_args.train_batch_size,
        gradient_accumulation_steps=sub_args.gradient_accumulation_steps,
        num_train_epochs=sub_args.epochs,
        weight_decay=sub_args.weight_decay,
    )

    set_seed(42)  # magic number :)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_from_disk(
        os.path.join(sub_args.input_directory, 'train_dataset')
    )  # put in your data path, dataset have train/valid dataset
    train_dataset = dataset["train"]

    if sub_args.encoder == "cross":
        # you can use 'klue/bert-base' model, and you have to change the code above.
        model_checkpoint = sub_args.model

        if model_checkpoint.split("/")[1].split("-")[0] == "roberta":
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            cross_encoder = RoBertaEncoder_For_CrossEncoder.from_pretrained(
                model_checkpoint
            )
        elif model_checkpoint.split("/")[1].split("-")[0] == "bert":
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            cross_encoder = BertEncoder_For_CrossEncoder.from_pretrained(
                model_checkpoint
            )

        if torch.cuda.is_available():
            cross_encoder = cross_encoder.to("cuda")

        c_encoder = crossencoder_train(
            args,
            train_dataset["question"],
            train_dataset["context"],
            tokenizer,
            cross_encoder,
            sampler=CustomSampler,
        )

        torch.save(
            c_encoder, os.path.join(sub_args.output_directory, 'c_encoder.pt')
        )

    elif sub_args.encoder == "bi":
        # in this code, you just can use 'klue/bert-base' in bi-encoder because I jsut make bertmodel in bi-encoder
        model_checkpoint = sub_args.model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        passage_encoder = BertEncoder_For_BiEncoder.from_pretrained(
            model_checkpoint)
        question_encoder = BertEncoder_For_BiEncoder.from_pretrained(
            model_checkpoint)

        if torch.cuda.is_available():
            passage_encoder = passage_encoder.to("cuda")
            question_encoder = question_encoder.to("cuda")

        p_encoder, q_encoder = biencoder_train(
            args,
            train_dataset["question"],
            train_dataset["context"],
            tokenizer,
            passage_encoder,
            question_encoder,
            sampler=CustomSampler,
            overflow=True,
        )

        torch.save(
            p_encoder, os.path.join(sub_args.output_directory, 'p_encoder.pt')
        )
        torch.save(
            q_encoder, os.path.join(sub_args.output_directory, 'q_encoder.pt')
        )
