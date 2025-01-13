import re


def compute_num_pads(list_bboxes):
    max_len = -1
    for bboxes in list_bboxes:
        num_bboxes = len(bboxes)
        if num_bboxes > max_len:
            max_len = num_bboxes
    num_pad_vector = []
    for bboxes in list_bboxes:
        num_pad_vector.append(max_len - len(bboxes))
    return num_pad_vector


def remove_punctuations(sentences):
    punctuations = ["''", "'", "``", "`", ".", "?", "!", ",", ":", "-", "--", "...", ";"]
    res_sentences_list = []
    for i in range(len(sentences)):
        res_sentence = []
        for word in sentences[i].split(' '):
            if word not in punctuations:
                res_sentence.append(word)
        res_sentences_list.append(' '.join(res_sentence))
    return res_sentences_list


def lowercase_and_clean_trailing_spaces(sentences):
    return [(sentences[i].lower()).rstrip() for i in range(len(sentences))]


def add_space_between_non_alphanumeric_symbols(sentences):
    return [re.sub(r'([^\w0-9])', r" \1 ", sentences[i]) for i in range(len(sentences))]


def tokenize(list_sentences):
    res_sentences_list = []
    for i in range(len(list_sentences)):
        sentence = list_sentences[i].split(' ')
        while '' in sentence:
            sentence.remove('')
        res_sentences_list.append(sentence)
    return res_sentences_list

def convert_vector_word2idx(sentence, word2idx_dict):
    return [word2idx_dict[word] for word in sentence]


def convert_allsentences_word2idx(sentences, word2idx_dict):
    return [convert_vector_word2idx(sentences[i], word2idx_dict) for i in range(len(sentences))]


def convert_vector_idx2word(sentence, idx2word_list):
    return [idx2word_list[idx] for idx in sentence]


def convert_allsentences_idx2word(sentences, idx2word_list):
    return [convert_vector_idx2word(sentences[i], idx2word_list) for i in range(len(sentences))]


def tokens2description(tokens, idx2word_list, sos_idx, eos_idx):
    desc = []
    for tok in tokens:
        if tok == sos_idx:
            continue
        if tok == eos_idx:
            break
        desc.append(tok)
    desc = convert_vector_idx2word(desc, idx2word_list)
    desc[-1] = desc[-1] + '.'
    pred = ' '.join(desc).capitalize()
    return pred


# import torchvision
from PIL import Image as PIL_Image


# def preprocess_image(pil_image, img_size):
#     transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
#     transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                 std=[0.229, 0.224, 0.225])])

#     # pil_image = PIL_Image.open(image_path)
#     if pil_image.mode != 'RGB':
#         pil_image = PIL_Image.new("RGB", pil_image.size)
#     preprocess_pil_image = transf_1(pil_image)
#     image = torchvision.transforms.ToTensor()(preprocess_pil_image)
#     image = transf_2(image)
#     return image.unsqueeze(0)

import torch

def create_pad_mask(mask_size, pad_row, pad_column):
    bs, out_len, in_len = mask_size
    pad_row_tens = torch.tensor([out_len]) - torch.tensor(pad_row).unsqueeze(-1).repeat(1, out_len)
    pad_col_tens = torch.tensor([in_len]) - torch.tensor(pad_column).unsqueeze(-1).repeat(1, in_len)
    arange_on_columns = (torch.arange(in_len).unsqueeze(0).repeat(bs, 1) < pad_col_tens).type(torch.int32)
    arange_on_rows = (torch.arange(out_len).unsqueeze(0).repeat(bs, 1) < pad_row_tens).type(torch.int32)
    mask = torch.matmul(arange_on_rows.unsqueeze(-1), arange_on_columns.unsqueeze(-2))
    return mask


def create_no_peak_and_pad_mask(mask_size, num_pads):
    block_mask = create_pad_mask(mask_size, num_pads, num_pads)
    bs, seq_len, seq_len = mask_size
    triang_mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.float),
                             diagonal=0).unsqueeze(0).repeat(bs, 1, 1)
    return torch.mul(block_mask, triang_mask)