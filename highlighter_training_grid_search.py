import torch
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer, AutoModel, BertForTokenClassification
import random
import os
import json
import argparse
import numpy as np
import pickle
import logging
import warnings
import statistics

# Suppress specific warnings using the warnings module
warnings.filterwarnings('ignore', message="Some weights of .* were not initialized from the model checkpoint")

# Set up logging to suppress all warnings and only show errors
logging.getLogger("transformers").setLevel(logging.ERROR)

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embed_texts_contriever(text, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
    
    return emb

def embed_texts_contriever2(text, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
    
    return emb



def main():

    # Argument parser setup
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--trials_for_same_setting', type=int, default=10)
    parser.add_argument('--epochs_range', type=int, default=10)
    parser.add_argument('--retrieval_method', type=str, default='None', choices=['None', 'top_1', 'top_k'])

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cik_to_name = {}
    
    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)

    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            if tenK[:4] == "2022":
                file_path = os.path.join(directory3, tenK)
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if ('company_name' in data) and ('cik' in data):
                            firm_name = data['company_name']
                            cik = data['cik']
                            cik_to_name[cik] = firm_name
                        
    
    # Load annotated results
    training_annotated_results = []
    with open('fin.rag/annotation/annotated_result/all/aggregate_train.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            id_text_label = [
                data['sample_id'],
                data['text'],
                data['tokens'],
                data['naive_aggregation']['label'],
                data['highlight_probs']
            ]
            training_annotated_results.append(id_text_label)

    testing_annotated_results = []
    with open('fin.rag/annotation/annotated_result/all/aggregate_test.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            id_text_label = [
                data['sample_id'],
                data['text'],
                data['tokens'],
                data['naive_aggregation']['label'],
                data['highlight_probs']
            ]
            testing_annotated_results.append(id_text_label)

    # Load paragraph information
    with open('para_info/para_info_contriever_firm.pkl', 'rb') as f:
        para_info = pickle.load(f)




    for i in range(args.epochs_range):
        f1_for_each_epoch = []
        for k in range(args.trials_for_same_setting):

            # Extract embeddings and texts
            final_embeddings = np.vstack([item[2] for item in para_info]).astype('float32')
            final_texts = [item[1] for item in para_info]
            final_ids = [item[3] for item in para_info]

            # Initialize tokenizer and retriever model
            tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            retriever = AutoModel.from_pretrained('facebook/contriever')
            retriever.to(device).eval()

            # Initialize classification model
            model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device).train()

            # Setup optimizer and loss criterion
            optimizer1 = torch.optim.Adam(
                list(model.parameters()), lr=args.lr
            )
            criterion1 = torch.nn.BCELoss()
            
            # Convert embeddings to tensors
            all_doc_embeddings = torch.tensor(final_embeddings).to(device)  # Shape: (num_docs, embedding_dim)
            for j in range(i + 1):
                count = 0
                
                for training_element in training_annotated_results:

                    query_firm_name = cik_to_name[training_element[0].split('_')[2]]
                    concat_text = f"{query_firm_name} {training_element[1]}"
                    
                    # Embed the query text
                    query_embedding = embed_texts_contriever(concat_text, retriever, tokenizer, device)
                    query_embedding = F.normalize(query_embedding, p=2, dim=1)
                    
                    query_embedding = query_embedding.to(device)  # Shape: (1, embedding_dim)

                    # Compute similarities (dot product)
                    similarities = torch.matmul(all_doc_embeddings, query_embedding.t()).squeeze()  # Shape: (num_docs,)

                    # Retrieve top-k documents
                    topk_values, topk_indices = torch.topk(similarities, args.epochs_range + 1)
                    
                    # Tokenize training element
                    tokenized_ids = tokenizer.convert_tokens_to_ids(training_element[2])
                    tokenized_stringA = torch.tensor(tokenized_ids).unsqueeze(0).to(device)
                    if args.retrieval_method == 'top_1':
                        tokenized_stringB = tokenizer.encode(final_texts[topk_indices[1]], add_special_tokens=False, return_tensors="pt").to(device)
                    elif args.retrieval_method == 'top_k':
                        tokenized_stringB = tokenizer.encode(final_texts[topk_indices[j + 1]], add_special_tokens=False, return_tensors="pt").to(device)
                    
                    # Truncate sequences if necessary
                    tokenized_stringA = tokenized_stringA[:, :250] if tokenized_stringA.size(1) > 250 else tokenized_stringA
                    if args.retrieval_method in ['top_1', 'top_k']:
                        tokenized_stringB = tokenized_stringB[:, :250] if tokenized_stringB.size(1) > 250 else tokenized_stringB
                    
                    # Combine tokenized strings with separator token
                    sep_token_id = tokenizer.sep_token_id
                    sep_token_tensor = torch.tensor([[sep_token_id]]).to(device)
                    if args.retrieval_method == 'None':
                        combined_tokenized_string = torch.cat((tokenized_stringA, sep_token_tensor), dim=1).to(device)
                    elif args.retrieval_method in ['top_1', 'top_k']:
                        combined_tokenized_string = torch.cat((tokenized_stringA, sep_token_tensor, tokenized_stringB), dim=1).to(device)
                    
                    # Create attention mask
                    attention_mask = torch.ones(combined_tokenized_string.shape, dtype=torch.long).to(device)
                    
                    inputs = {
                        "input_ids": combined_tokenized_string,
                        "attention_mask": attention_mask
                    }
                    
                    # Get model outputs
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Find the separator token index
                    sep_index = (combined_tokenized_string == sep_token_id).nonzero(as_tuple=True)[1].item()
                    logits_before_sep = logits[:, :sep_index, :]
                    
                    # Compute softmax probabilities
                    probabilities_label_1 = torch.sigmoid(logits_before_sep)[0, :, 1]
                    
                    # Get the true label tensor and truncate if necessary
                    true_label_tensor = torch.tensor(training_element[3]).float().to(device)
                    if len(true_label_tensor) > 250:
                        true_label_tensor = true_label_tensor[:250]
                    
                    # Prepare for loss computation
                    optimizer1.zero_grad()
                    
                    # Compute and print loss
                    loss = criterion1(probabilities_label_1, true_label_tensor)
                    
                    # Backpropagation and optimization step
                    loss.backward()
                    optimizer1.step()

            
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            count = 0

            retriever.eval()
            model.eval()

            for testing_element in testing_annotated_results:
                count += 1
                #print(f'{count}')

                query_firm_name = cik_to_name[testing_element[0].split('_')[2]]
                concat_text = f"{query_firm_name} {testing_element[1]}"

                # Embed the query text
                query_embedding = embed_texts_contriever2(concat_text, retriever, tokenizer, device)
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
                
                query_embedding = query_embedding.to(device)  # Shape: (1, embedding_dim)
                all_doc_embeddings = all_doc_embeddings.to(device)

                # Compute similarities (dot product)
                similarities = torch.matmul(all_doc_embeddings, query_embedding.t()).squeeze()  # Shape: (num_docs,)

                topk_values, topk_indices = torch.topk(similarities, 2)

                tokenized_ids = tokenizer.convert_tokens_to_ids(testing_element[2])
                tokenized_stringA = torch.tensor(tokenized_ids).unsqueeze(0).to(device)
                if args.retrieval_method in ['top_1', 'top_k']:
                    tokenized_stringB = tokenizer.encode(final_texts[topk_indices[1]], add_special_tokens=False, return_tensors="pt").to(device)

                # Truncate sequences if necessary
                tokenized_stringA = tokenized_stringA[:, :250] if tokenized_stringA.size(1) > 250 else tokenized_stringA
                if args.retrieval_method in ['top_1', 'top_k']:
                    tokenized_stringB = tokenized_stringB[:, :250] if tokenized_stringB.size(1) > 250 else tokenized_stringB
                
                # Prepare combined input for the model
                sep_token_id = tokenizer.sep_token_id
                sep_token_tensor = torch.tensor([[sep_token_id]]).to(device)
                if args.retrieval_method == 'None':
                    combined_tokenized_string = torch.cat((tokenized_stringA, sep_token_tensor), dim=1).to(device)
                elif args.retrieval_method in ['top_1', 'top_k']:
                    combined_tokenized_string = torch.cat((tokenized_stringA, sep_token_tensor, tokenized_stringB), dim=1).to(device)
                attention_mask = torch.ones(combined_tokenized_string.shape, dtype=torch.long).to(device)
                
                with torch.no_grad():
                    # Get model outputs
                    inputs = {"input_ids": combined_tokenized_string, "attention_mask": attention_mask}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Calculate probabilities before the separator token
                    sep_index = (combined_tokenized_string == sep_token_id).nonzero(as_tuple=True)[1].item()
                    logits_before_sep = logits[:, :sep_index, :]
                    probabilities_label_1 = torch.sigmoid(logits_before_sep)[0, :, 1]
                    
                    binary_tensor = (probabilities_label_1 >= 0.5).float()
                    true_label_tensor = torch.tensor(testing_element[3]).to(device)[:250]
                    
                    # Calculate True Positives (TP)
                    TP = torch.sum((binary_tensor == 1) & (true_label_tensor == 1)).item()

                    # Calculate False Positives (FP)
                    FP = torch.sum((binary_tensor == 1) & (true_label_tensor == 0)).item()

                    # Calculate False Negatives (FN)
                    FN = torch.sum((binary_tensor == 0) & (true_label_tensor == 1)).item()

                    # Calculate Precision
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

                    # Calculate Recall
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                    # Calculate F1 Score
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1_score

            f1_for_each_epoch.append(total_f1 / len(testing_annotated_results))
        
        averageF1 = round(sum(f1_for_each_epoch) / len(f1_for_each_epoch), 4)
        stdF1 = round(statistics.stdev(f1_for_each_epoch), 4)
        print(f'Train for {i + 1} epochs: Average F1 is {averageF1}, SD is {stdF1}')


if __name__ == "__main__":
    main()
