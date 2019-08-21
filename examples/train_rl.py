import torch
from pytorch_transformers import *
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange as trange
from torch import nn
import random
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import run_squad as rs
# from utils_squad import (read_squad_examples, convert_examples_to_features,
#                          RawResult, write_predictions,
#                          RawResultExtended, write_predictions_extended)
import utils_squad as us
import nltk
from nltk import word_tokenize
from rouge import Rouge
import json
import logging
import math
import collections
from io import open
from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
import os
import argparse

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def init_model(args, load_fine_tuned = True):

    MODELS = [(BertForQuestionAnswering,       BertTokenizer,      'bert-large-uncased')]
    for model_class, tokenizer_class, pretrained_weights in MODELS:
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        config = BertConfig.from_pretrained(pretrained_weights)
    
    if load_fine_tuned:
        path = '/home/ubuntu/rl_testing/large-uncased/pytorch_model.bin'
        model.load_state_dict(torch.load(path))
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=0.9)

    model.to(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # prefix = ''
    # output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    # output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    # if args.version_2_with_negative:
    #         output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    # else:
    #         output_null_log_odds_file = None

    return model, optimizer, tokenizer

def load_dataset(args, tokenizer, number_of_examples = None, batch_size = 4):
    ## For Train
    # dataset = rs.load_and_cache_examples(args, tokenizer)
    dataset, examples, features = rs.load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=True, number_of_examples = number_of_examples)
    train_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    return dataset, examples, features, train_sampler, train_dataloader

def train_with_rewards(args, model, tokenizer, optimizer, epochs = 2, number_of_examples = None, batch_size = 4):
    
    logger.info('Loading SQuAD dataset ....')
    dataset, examples, features, train_sampler, train_dataloader = load_dataset(args, tokenizer, number_of_examples=number_of_examples, batch_size=batch_size)
    logger.info('Loaded dataset')

    train_iterator = trange(int(epochs), desc="Epoch", disable=-1 not in [-1, 0])
    model.zero_grad()
    logger.info('Starting Training ....')
    for tr_iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=-1 not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            torch.cuda.empty_cache()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
#             break
            outputs = model(**form_inputs(args, batch))
            loss = outputs[0]
            
            results, used_features = get_all_results(args, batch, outputs, features)
            used_examples = get_required_examples(used_features, examples)
            all_predictions = write_predictions(used_examples, used_features, results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
            
            rewards = calc_rewards(all_predictions, used_examples)
            loss = loss + rewards
            if step % 1000 == 0:
                logger.info('reward = %s', rewards)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if args.local_rank in [-1, 0] and args.save_steps > 0 and step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
        logger.info('The loss in step %s is %s',tr_iter, loss)
    logger.info('Training complete ...')

def form_inputs(args, batch):
    inputs = {}
    inputs = {'input_ids': batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  None if args.model_type == 'xlm' else batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
    return inputs

def get_all_results(args, batch, outputs, features):
    example_indices = batch[3]
    all_results = []
    used_features = []
    for i, example_index in enumerate(example_indices):
                # logger.info(example_index)
                eval_feature = features[example_index.item()]
                used_features.append(eval_feature)
                unique_id = int(eval_feature.unique_id)
                if args.model_type in ['xlnet', 'xlm']:
                    # XLNet uses a more complex post-processing procedure
                    result = us.RawResultExtended(unique_id            = unique_id,
                                               start_top_log_probs  = outputs[0][i].tolist(),
                                               start_top_index      = (outputs[1][i]).tolist(),
                                               end_top_log_probs    = (outputs[2][i]).tolist(),
                                               end_top_index        = (outputs[3][i]).tolist(),
                                               cls_logits           = (outputs[4][i]).tolist())
                else:
                    result = us.RawResult(unique_id    = unique_id,
                                       start_logits = outputs[1][i].tolist(),
                                       end_logits   = outputs[2][i].tolist())
                    # logger.info(unique_id)
                all_results.append(result)
    return all_results, used_features

def get_required_examples(used_features, all_examples):
    used_examples = []
    for f in used_features:
        used_examples.append(all_examples[f.example_index])
    return used_examples

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    count_f = 0
    for feature in all_features:
        example_index_to_features[count_f].append(feature)
        count_f += 1

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            
            #  if feature.unique_id == 1000000075:
                
            result = unique_id_to_result[feature.unique_id]
            start_indexes = us._get_best_indexes(result.start_logits, n_best_size)
            end_indexes = us._get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = us.get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = us._compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    # with open(output_prediction_file, "w") as writer:
    #     writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # if version_2_with_negative:
    #     with open(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def calc_rewards(all_predictions, used_examples):
    ## Convert all_predictions into list
    pred_list = list(all_predictions.items()) ## List of tuples
    ans_list = []
    bleu_loss = []
    rougel_f_scores = []
    rouge = Rouge()
    for id_x, ans in pred_list:
        ans_list.append(word_tokenize(ans)) ## List of all the answers. shape - (batch_size,)
    
    actual_answers = []
    
    for ex in used_examples:
        st = ex.start_position
        end = ex.end_position
        actual_ans = ex.doc_tokens[st:end]
        actual_answers.append(actual_ans) ## List of actual answers from train set
   
    for i in range(len(ans_list)):
        bleu_loss.append(1 - nltk.translate.bleu_score.sentence_bleu([actual_answers[i]], ans_list[i]))
        try:
            scores = rouge.get_scores(" ".join(ans_list[i]), " ".join(actual_answers[i]))
            rougel_f_scores.append(1 - scores[0]['rouge-l']['f'])
        except:
            rougel_f_scores.append(0.0)
        # logger.info(scores)
        
    bleu_avg = sum(bleu_loss)/len(bleu_loss)
    rouge_avg = sum(rougel_f_scores)/len(rougel_f_scores)
    
    total_loss = bleu_avg + rouge_avg
        
    return total_loss

def get_args():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    # parser.add_argument("--predict_file", default=None, type=str, required=True,
    #                     help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    #                     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    return args

class args_list:
    def __init__(self):
        self.local_rank = -1
        self.n_gpu = 1
        self.train_file = '/home/ubuntu/question_generation/data/train-v2.0.json'
        self.evaluate = 0
        self.predict_file = '/home/ubuntu/question_generation/data/dev-v2.0.json'
        self.eval_batch_size = 1
        self.model_type = 'bert'
        self.model_name_or_path = 'bert-base-uncased'
        self.output_dir = './outputs/'
        self.tokenizer_name = 'BertTokenizer'
        self.max_seq_length = 384
        self.version_2_with_negative = True
        self.doc_stride = 128
        self.max_query_length = 64
        self.device = torch.device('cuda')
        self.overwrite_cache = False
        self.null_score_diff_threshold = 0.0
        self.n_best_size = 20
        self.max_answer_length = 30
        self.verbose_logging = False
        self.do_lower_case = True
        self.no_cuda = False
        self.fp16 = False


def main():

    # args = get_args()
    # args = args_list()

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    # parser.add_argument("--predict_file", default=None, type=str, required=True,
    #                     help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--n_gpu', type=int, default=1, help="default num of GPUs")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 2
        logger.info('number of GPUs is %s', args.n_gpu)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # logger.info('------------------------------------NOT SUPPOSED TO BE HERE------------------------------------')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 2
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    model, optimizer, tokenizer = init_model(args, load_fine_tuned = True)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    train_with_rewards(args, model, tokenizer, optimizer, epochs = args.num_train_epochs, number_of_examples = None, batch_size = args.per_gpu_train_batch_size)

    # ------------ run_squad main: (use if in prod) ----------------

    # # Training
    # if args.do_train:
    #     train_dataset = rs.load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    #     global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    #     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # # Save the trained model and the tokenizer
    # if args.local_rank == -1 or torch.distributed.get_rank() == 0:
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)

    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)

    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    #     model.to(args.device)

    # # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)

    #     for checkpoint in checkpoints:
    #         # Reload the model
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)

    #         # Evaluate
    #         result = evaluate(args, model, tokenizer, prefix=global_step)

    #         result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
    #         results.update(result)

    # logger.info("Results: {}".format(results))

    # return results


if __name__ == "__main__":
    main()
